//! Register-pressure microbenchmark.
//!
//! Fast-iteration tool for locating Apple Silicon's register-budget
//! tier cliffs. Runs a minimal fragment shader whose per-thread state
//! size we control, measures submitted_done_ms per frame, and sweeps
//! the state size to find the step-change that indicates crossing a
//! tier boundary.
//!
//! Why this exists: the real ray-march shader has many confounding
//! factors (memory patterns, branching, etc.). A synthetic shader with
//! a single parameter lets us isolate "what happens when per-thread
//! state grows by N bytes" cleanly.
//!
//! Usage:
//!     cargo run --bin reg_bench --release -- [--resolution WxH]
//!                                             [--frames N]
//!                                             [--iter N]
//!                                             [--stacks N1,N2,...]
//!                                             [--ambient N]
//!
//! - `--ambient N` adds N vec3<f32> "live scalar" variables declared
//!   at function scope and read/written every iteration. Simulates
//!   the ambient register load of `cur_node_origin`, `cur_side_dist`,
//!   `inv_dir`, etc. in march_cartesian. Each vec3 is 12 B of declared
//!   state; the compiler's intermediate-register overhead on top of
//!   that is what pushes real shaders past the tier boundary at lower
//!   declared-byte counts than this synthetic's own cliff.
//!
//! Defaults: 2560x1440, 30 frames per config, 64 inner iterations per
//! pixel, stack sizes 1,2,4,8,16,32,64,128, ambient=0.
//!
//! Output: for each stack size, mean/std submitted_done_ms. A step
//! change in mean between adjacent sizes indicates a tier cliff.
//!
//! The shader's per-thread state grows with STACK_N: a `array<u32, N>`
//! plus per-iteration scalars. Each DDA-shaped inner iteration does
//! read-modify-write on `s[depth]` with a variable `depth`, forcing
//! the compiler to treat the array as dynamically-indexed (the same
//! property that forces thread-private allocation in march_cartesian).

use std::time::Duration;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // --- arg parsing (tiny, hand-rolled) ---
    let mut resolution = (2560u32, 1440u32);
    let mut frames = 30u32;
    let mut iter = 64u32;
    let mut stacks: Vec<u32> = vec![1, 2, 4, 8, 16, 32, 64, 128];
    let mut ambient: u32 = 0;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--resolution" => {
                let v = &args[i + 1];
                let (w, h) = v.split_once('x').expect("resolution should be WxH");
                resolution = (w.parse().unwrap(), h.parse().unwrap());
                i += 2;
            }
            "--frames" => {
                frames = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--iter" => {
                iter = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--stacks" => {
                stacks = args[i + 1]
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect();
                i += 2;
            }
            "--ambient" => {
                ambient = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    let (w, h) = resolution;
    println!(
        "reg_bench: resolution={w}x{h} frames={frames} iter={iter} stacks={stacks:?} ambient={ambient}"
    );

    // --- wgpu setup ---
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("no adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("reg_bench"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("no device");

    let info = adapter.get_info();
    println!("reg_bench: adapter={} backend={:?}", info.name, info.backend);

    // --- fixed offscreen target ---
    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // --- uniforms: pixel bounds + iter count ---
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Uniforms {
        width: u32,
        height: u32,
        n_iter: u32,
        _pad: u32,
    }
    let uniforms = Uniforms {
        width: w,
        height: h,
        n_iter: iter,
        _pad: 0,
    };
    let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

    // --- sink buffer: prevents the compiler from DCE'ing the whole loop ---
    // Every pixel writes one u32 via atomicAdd. Without this the compiler
    // could observe that `s` is never read into the output and drop the
    // entire array + loop.
    let sink_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sink"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&sink_buf, 0, &[0u8; 16]);

    // --- bind-group layout (shared across all stack sizes) ---
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sink_buf.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    // --- sweep stack sizes ---
    println!();
    println!(
        "{:>10} {:>14} {:>12} {:>12} {:>12} {:>12}",
        "stack_N", "state_bytes", "mean_ms", "p50_ms", "p90_ms", "std_ms"
    );
    println!("{}", "-".repeat(80));

    let mut results: Vec<(u32, f64)> = Vec::new();
    for &stack_n in &stacks {
        let shader_src = shader_with_stack_size(stack_n, ambient);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reg_bench_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("reg_bench_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Warm-up frame (not measured) — first frame includes shader
        // compilation latency in wgpu backends that lazy-compile.
        render_one_frame(&device, &queue, &pipeline, &bg, &color_view, true);

        // Measured frames.
        let mut samples: Vec<f64> = Vec::with_capacity(frames as usize);
        for _ in 0..frames {
            let t = render_one_frame(&device, &queue, &pipeline, &bg, &color_view, true);
            samples.push(t);
        }

        // state_bytes: 4 B per u32 in the stack. Other per-thread
        // scalars (iter counter, acc, depth) are small and constant
        // across configurations; baseline for comparison.
        let state_bytes = (stack_n as usize) * 4;

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let p50 = samples[n / 2];
        let p90 = samples[(n * 9 / 10).min(n - 1)];
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt();

        println!(
            "{:>10} {:>14} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
            stack_n, state_bytes, mean, p50, p90, std
        );
        results.push((stack_n, mean));
    }

    // Step-change detection: print ratio between adjacent mean times.
    println!();
    println!("Step-change detection (ratio = mean[i+1] / mean[i]):");
    println!(
        "{:>10} {:>10} {:>10} {:>10}",
        "stack_N", "next_N", "ratio", "delta_ms"
    );
    println!("{}", "-".repeat(44));
    for w in results.windows(2) {
        let (n0, m0) = w[0];
        let (n1, m1) = w[1];
        let ratio = m1 / m0;
        let marker = if ratio > 1.3 {
            "  *** tier cliff likely ***"
        } else {
            ""
        };
        println!(
            "{:>10} {:>10} {:>10.3} {:>10.3}{}",
            n0,
            n1,
            ratio,
            m1 - m0,
            marker
        );
    }
}

/// Render one frame, return submitted-done latency in ms.
/// Uses the `on_submitted_work_done` callback (the authoritative
/// Apple Silicon signal, per the main renderer's doc comments).
fn render_one_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    view: &wgpu::TextureView,
    _wait: bool,
) -> f64 {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("frame"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fs"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }

    let done_slot: std::sync::Arc<std::sync::Mutex<Option<std::time::Instant>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    let submit_start = std::time::Instant::now();
    queue.submit(Some(encoder.finish()));
    {
        let done_slot = std::sync::Arc::clone(&done_slot);
        queue.on_submitted_work_done(move || {
            let mut slot = done_slot.lock().unwrap();
            *slot = Some(std::time::Instant::now());
        });
    }
    let _ = device.poll(wgpu::PollType::Wait);
    let done = done_slot.lock().unwrap();
    let elapsed = done
        .map(|t| t.duration_since(submit_start))
        .unwrap_or(Duration::ZERO);
    elapsed.as_secs_f64() * 1000.0
}

/// Build the shader source with the requested stack size + ambient
/// vec3 count inlined. The shader does a read-modify-write loop on
/// a dynamically-indexed array to mimic the register-pressure pattern
/// of `s_cell[depth]` in `march_cartesian`. AMBIENT vec3 scalars
/// declared at function scope and read+written every iteration mimic
/// ambient register load from `cur_node_origin`, `cur_side_dist`,
/// `inv_dir`, etc. The sink atomicAdd at the end prevents DCE.
fn shader_with_stack_size(stack_n: u32, ambient: u32) -> String {
    // Generate ambient scalar declarations, initialisation, per-
    // iteration updates, and a final "feed-back into acc" that
    // prevents DCE.
    let mut amb_decls = String::new();
    let mut amb_updates = String::new();
    let mut amb_sink = String::new();
    for i in 0..ambient {
        amb_decls.push_str(&format!(
            "    var amb_{i}: vec3<f32> = vec3<f32>(f32(px + {i}u), f32(py ^ {i}u), f32({i}u) * 0.3);\n"
        ));
        // Use all three components of the previous iteration's amb_i
        // in the next update — ensures the full 12 B stays live.
        amb_updates.push_str(&format!(
            "        amb_{i} = amb_{i} * vec3<f32>(1.001, 0.999, 1.0003) + \
             vec3<f32>(f32(acc & 255u) * 0.01, amb_{i}.z * 0.5, 0.1);\n"
        ));
        amb_sink.push_str(&format!(
            "        acc = acc ^ u32(amb_{i}.x * 100.0) ^ u32(amb_{i}.y * 100.0) ^ u32(amb_{i}.z * 100.0);\n"
        ));
    }

    format!(
        r#"
struct Uniforms {{
    width: u32,
    height: u32,
    n_iter: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> sink: array<atomic<u32>, 4>;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {{
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {{
    // Per-thread state: STACK_N-element u32 array indexed dynamically.
    var s: array<u32, {stack_n}u>;

    let px = u32(frag_coord.x);
    let py = u32(frag_coord.y);

    // Ambient register load — vec3 scalars, live across all iterations.
{amb_decls}
    // Seed the stack with per-pixel values.
    for (var i = 0u; i < {stack_n}u; i = i + 1u) {{
        s[i] = px * 31u + py * 41u + i * 137u;
    }}

    var acc: u32 = px ^ py;
    var depth: u32 = 0u;
    for (var iter = 0u; iter < u.n_iter; iter = iter + 1u) {{
        // Ambient updates: keep all amb_* live and evolving.
{amb_updates}
        // Dynamic-index stack op: fixed 4-element cycle regardless
        // of STACK_N so access pattern is constant.
        depth = (depth + (acc & 1u) + 1u) & 3u;
        let v = s[depth];
        acc = acc ^ v;
        s[depth] = v + acc;

        // Feed ambient values back into acc so the compiler can't DCE them.
{amb_sink}
    }}

    atomicAdd(&sink[0], acc);
    return vec4<f32>(f32(acc & 0xFFu) / 255.0, 0.0, 0.0, 1.0);
}}
"#
    )
}

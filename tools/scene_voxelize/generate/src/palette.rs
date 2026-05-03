use std::{collections::BinaryHeap, time::Instant};

pub struct Palette {
    pub rgba: Vec<glam::Vec4>,
    pub lut: Vec<u32>,
}

impl Palette {
    /// Target palette size.
    ///
    /// 65 000 leaves headroom inside the game's 65 536-entry u16
    /// palette for the builtin blocks and any other imported models
    /// registered in the same `ColorRegistry`.
    pub const SIZE: u32 = 65_000;

    /// Somewhat based on the article https://blog.pkh.me/p/39-improving-color-quantization-heuristics.html
    ///
    /// Does median cut to find the approximate palette given the samples
    ///
    /// Everything happens in oklab space here. the author mentioned with small palettes luminance cuts sort of dominate, however
    ///
    /// Also, creates an srgb `(0-255)^3` -> palette index LUT
    pub fn from_samples(samples: &mut [glam::Vec3]) -> Self {
        let timer = Instant::now();
        // median cut phase
        // here we just choose the axis with the widest range
        // article I read suggested also using the MSE (per-axis)
        // to determine the axis to cut on
        fn split(samples: &mut [glam::Vec3]) -> (&mut [glam::Vec3], &mut [glam::Vec3]) {
            let mut lab_min = glam::Vec3::INFINITY;
            let mut lab_max = glam::Vec3::NEG_INFINITY;
            samples.iter().for_each(|sample| {
                lab_min = sample.min(lab_min);
                lab_max = sample.max(lab_max);
            });
            let split_component = (lab_max - lab_min).max_position();
            samples.sort_unstable_by(|a, b| {
                let a_c = a[split_component];
                let b_c = b[split_component];
                a_c.total_cmp(&b_c)
            });
            samples.split_at_mut(samples.len() / 2)
        }

        struct Bucket<'a> {
            data: &'a mut [glam::Vec3],
            mean: glam::Vec3,
            mse: f32,
        }
        impl<'a> Bucket<'a> {
            fn new(data: &'a mut [glam::Vec3]) -> Self {
                let inv_len = 1.0 / (data.len() as f32);
                let mean = data
                    .iter()
                    .fold(glam::Vec3::ZERO, |acc, &x| acc + x * inv_len);
                let mse = data
                    .iter()
                    .fold(0.0, |acc, &x| acc + (x - mean).length_squared() * inv_len);
                Self { data, mean, mse }
            }
        }
        impl<'a> PartialEq for Bucket<'a> {
            fn eq(&self, other: &Self) -> bool {
                self.data.len() == other.data.len() && self.mse == other.mse
            }
        }
        impl<'a> Eq for Bucket<'a> {}
        impl<'a> PartialOrd for Bucket<'a> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                match (self.data.len() < 2, other.data.len() < 2) {
                    (true, false) => {
                        return Some(core::cmp::Ordering::Less);
                    }
                    (false, true) => {
                        return Some(core::cmp::Ordering::Greater);
                    }
                    _ => {}
                }
                self.mse.partial_cmp(&other.mse)
            }
        }
        impl<'a> Ord for Bucket<'a> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                match (self.data.len() < 2, other.data.len() < 2) {
                    (true, false) => {
                        return core::cmp::Ordering::Less;
                    }
                    (false, true) => {
                        return core::cmp::Ordering::Greater;
                    }
                    _ => {}
                }
                self.mse.total_cmp(&other.mse)
            }
        }

        let mut buckets = BinaryHeap::with_capacity(Self::SIZE as usize);
        buckets.push(Bucket::new(samples));

        // Start with one bucket (the whole sample set) and split
        // until we have `SIZE` buckets total.
        for _ in 0..(Self::SIZE - 1) {
            let bucket = buckets.pop().unwrap();
            let pair = split(bucket.data);
            buckets.push(Bucket::new(pair.0));
            buckets.push(Bucket::new(pair.1));
        }

        let mut palette_rgba = Vec::with_capacity(Self::SIZE as usize);
        let mut palette_oklab = Vec::with_capacity(Self::SIZE as usize);

        // Every bucket is a real cluster — no more reserved slot 0.
        // `Child::Empty` is the only empty sentinel in the tree, so
        // palette slot 0 is free to carry whatever cluster k-means
        // emits there.
        for bucket in buckets {
            let lab = bucket.mean;
            let rgba = oklab_to_linear_rgb(lab).extend(1.0);
            palette_rgba.push(rgba);
            palette_oklab.push(lab.to_array());
        }

        let lab_tree = kd_tree::KdIndexTree::build_by_ordered_float(&palette_oklab);

        let mut lut = vec![0; 256 * 256 * 256];
        let mut mse = 0.0;

        for r in 0..256 {
            for g in 0..256 {
                for b in 0..256 {
                    let srgb = glam::uvec3(r, g, b).as_vec3() / 255.0;
                    let linear = srgb_to_linear(srgb);
                    let lab = linear_rgb_to_oklab(linear);

                    let nearest = lab_tree.nearest(&lab.to_array()).unwrap();
                    mse += nearest.squared_distance as f64;
                    let palette_index = *nearest.item as u32;

                    lut[(r | (g << 8) | (b << 16)) as usize] = palette_index;
                }
            }
        }
        mse /= 256u32.pow(3) as f64;
        println!("palette MSE: {:.4}", mse);
        println!("palette generation (CPU): {:#?}", timer.elapsed());

        Self {
            rgba: palette_rgba,
            lut,
        }
    }
}

pub fn srgb_to_linear(srgb: glam::Vec3) -> glam::Vec3 {
    srgb.map(|x| {
        if x < 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    })
}

pub fn oklab_to_linear_rgb(lab: glam::Vec3) -> glam::Vec3 {
    const M1: glam::Mat3 = glam::Mat3::from_cols_array(&[
        1.000000000,
        1.000000000,
        1.000000000,
        0.396337777,
        -0.105561346,
        -0.089484178,
        0.215803757,
        -0.063854173,
        -1.291485548,
    ]);
    const M2: glam::Mat3 = glam::Mat3::from_cols_array(&[
        4.076724529,
        -1.268143773,
        -0.004111989,
        -3.307216883,
        2.609332323,
        -0.703476310,
        0.230759054,
        -0.341134429,
        1.706862569,
    ]);

    let lms = M1 * lab;
    return M2 * (lms * lms * lms);
}

pub fn linear_rgb_to_oklab(rgb: glam::Vec3) -> glam::Vec3 {
    const M1: glam::Mat3 = glam::Mat3::from_cols_array(&[
        0.4121656120,
        0.2118591070,
        0.0883097947,
        0.5362752080,
        0.6807189584,
        0.2818474174,
        0.0514575653,
        0.1074065790,
        0.6302613616,
    ]);

    const M2: glam::Mat3 = glam::Mat3::from_cols_array(&[
        0.2104542553,
        1.9779984951,
        0.0259040371,
        0.7936177850,
        -2.4285922050,
        0.7827717662,
        -0.0040720468,
        0.4505937099,
        -0.8086757660,
    ]);

    let lms = M1 * rgb;
    return M2 * (lms.signum() * lms.abs().powf(1.0 / 3.0));
}

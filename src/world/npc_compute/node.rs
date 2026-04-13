//! Render graph node that dispatches the NPC compute shader.

use bevy::prelude::*;
use bevy::render::{
    render_graph,
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

use super::pipeline::{NpcComputeBindGroup, NpcComputePipeline};

const WORKGROUP_SIZE: u32 = 64;

pub enum NpcComputeState {
    Loading,
    Ready,
}

pub struct NpcComputeNode {
    state: NpcComputeState,
}

impl Default for NpcComputeNode {
    fn default() -> Self {
        Self {
            state: NpcComputeState::Loading,
        }
    }
}

impl render_graph::Node for NpcComputeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<NpcComputePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        match self.state {
            NpcComputeState::Loading => {
                if let bevy::render::render_resource::CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id)
                {
                    self.state = NpcComputeState::Ready;
                }
            }
            NpcComputeState::Ready => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if !matches!(self.state, NpcComputeState::Ready) {
            return Ok(());
        }

        let Some(bind_group_res) = world.get_resource::<NpcComputeBindGroup>() else {
            return Ok(());
        };
        if bind_group_res.npc_count == 0 {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<NpcComputePipeline>();

        let Some(compute_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &bind_group_res.bind_group, &[]);
        pass.set_pipeline(compute_pipeline);

        let workgroups = (bind_group_res.npc_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }
}

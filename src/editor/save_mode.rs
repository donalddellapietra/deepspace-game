//! Save mode: hover a cell at a non-leaf view layer and stash the
//! underlying subtree as a reusable mesh.
//!
//! When `SaveMode.active == true`, the usual white outline gizmo is
//! suppressed and the hovered view cell is overlaid with a neon
//! translucent mesh of the node beneath it. The overlay re-uses the
//! same `NodeId`-keyed baked sub-mesh cache that `world::render`
//! fills in for the renderer — no mesh is baked twice.
//!
//! Pressing the save key (`G`) records the hovered `NodeId` into
//! `SavedMeshes`, which `inventory` displays alongside the block
//! grid. Only view layers `L <= MAX_LAYER - 2` are eligible: any
//! higher and one view cell is smaller than a tree leaf, so there's
//! no clean node to save.

use bevy::ecs::hierarchy::ChildOf;
use bevy::prelude::*;

use crate::camera::CursorLocked;
use crate::interaction::TargetedBlock;
use crate::inventory::InventoryState;
use crate::world::position::LayerPos;
use crate::world::render::RenderState;
use crate::world::tree::{
    slot_index, NodeId, BRANCH_FACTOR, EMPTY_NODE, MAX_LAYER,
    NODE_VOXELS_PER_AXIS,
};
use crate::world::view::{
    bevy_origin_of_layer_pos, cell_size_at_layer, scale_for_layer, WorldAnchor,
};
use crate::world::{CameraZoom, WorldState};

/// Extra size factor on the highlight parent so the overlay sits a
/// sliver outside the world block. Without this the translucent
/// neon material fights the opaque block's depth buffer and drops
/// out across most of the surface.
const OVERLAY_SCALE_BIAS: f32 = 1.02;

// -------------------------------------------------------------- resources

#[derive(Resource, Default)]
pub struct SaveMode {
    pub active: bool,
}

/// A saved subtree, referenced by its content-addressed `NodeId`.
///
/// Capturing a mesh bumps the library refcount on `node_id`
/// (see `save_on_click`), so the subtree survives in the library
/// even after the live world edits its original location away.
/// Without that pin, placing a captured mesh after even small
/// unrelated edits would panic in `install_subtree` — the captured
/// id would dangle.
#[derive(Clone, Debug)]
pub struct SavedMesh {
    pub node_id: NodeId,
    /// Tree layer of the saved node. Used later if we want to place
    /// the saved mesh back into the world at a matching zoom.
    pub layer: u8,
}

#[derive(Resource, Default)]
pub struct SavedMeshes {
    pub items: Vec<SavedMesh>,
}

#[derive(Resource)]
pub struct SaveHighlightMaterial {
    pub handle: Handle<StandardMaterial>,
}

/// Tracks the currently-spawned highlight entity so we only rebuild
/// it when the hovered NodeId changes.
#[derive(Resource, Default)]
pub struct SaveHighlightState {
    current: Option<(NodeId, Entity)>,
}

#[derive(Component)]
pub struct SaveHighlight;

// ------------------------------------------------------ material init

pub fn init_save_highlight_material(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let handle = materials.add(StandardMaterial {
        base_color: Color::srgba(0.25, 1.0, 0.85, 0.45),
        emissive: LinearRgba::new(0.6, 2.8, 2.4, 1.0),
        alpha_mode: AlphaMode::Blend,
        metallic: 0.7,
        perceptual_roughness: 0.15,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.insert_resource(SaveHighlightMaterial { handle });
}

// --------------------------------------------------------------- toggle

/// V → flip save mode on/off. Gated on cursor locked and inventory
/// closed so it doesn't fire while the player is in a menu.
pub fn toggle_save_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    locked: Res<CursorLocked>,
    mut save_mode: ResMut<SaveMode>,
) {
    if inv.open || !locked.0 {
        return;
    }
    if keyboard.just_pressed(KeyCode::KeyV) {
        save_mode.active = !save_mode.active;
    }
}

// ---------------------------------------------------------- save click

/// Left-click while in save mode → record the currently-hovered
/// subtree into `SavedMeshes`, fire a toast, and drop back into
/// normal mode. `place_block` and `remove_block` both early-return
/// when `SaveMode.active`, so the same click doesn't also edit the
/// world.
pub fn save_on_click(
    mouse: Res<ButtonInput<MouseButton>>,
    locked: Res<CursorLocked>,
    inv: Res<InventoryState>,
    mut save_mode: ResMut<SaveMode>,
    targeted: Res<TargetedBlock>,
    zoom: Res<CameraZoom>,
    mut world: ResMut<WorldState>,
    mut saved: ResMut<SavedMeshes>,
    mut commands: Commands,
) {
    if inv.open || !save_mode.active || !locked.0 || locked.is_changed() {
        return;
    }
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if !save_mode_eligible(zoom.layer) {
        return;
    }
    let Some(lp) = targeted.hit_layer_pos.as_ref() else {
        return;
    };
    let Some((node_id, layer)) = resolve_node_at_lp(&world, lp) else {
        return;
    };

    // No-op if we already have this exact subtree saved, but still
    // flash a toast + drop out of save mode so the click gives
    // visible feedback. First-time saves pin the node in the
    // library so it survives any future edits that would otherwise
    // have evicted it.
    let already = saved.items.iter().any(|s| s.node_id == node_id);
    if !already {
        world.library.ref_inc(node_id);
        saved.items.push(SavedMesh { node_id, layer });
    }

    let idx = saved
        .items
        .iter()
        .position(|s| s.node_id == node_id)
        .expect("just pushed / existed");
    let message = if already {
        format!("Already saved (#{}, L{})", idx, layer)
    } else {
        format!("Saved mesh #{} (L{})", idx, layer)
    };
    super::toast::show_toast(&mut commands, message);

    save_mode.active = false;
}

// ----------------------------------------------------- highlight system

pub fn update_save_highlight(
    mut commands: Commands,
    save_mode: Res<SaveMode>,
    targeted: Res<TargetedBlock>,
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    world: Res<WorldState>,
    material: Option<Res<SaveHighlightMaterial>>,
    mut state: ResMut<SaveHighlightState>,
    mut render_state: ResMut<RenderState>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Some(material) = material else { return };

    let want: Option<(NodeId, u8, Vec3)> =
        if save_mode.active && save_mode_eligible(zoom.layer) {
            targeted.hit_layer_pos.as_ref().and_then(|lp| {
                let (node_id, layer) = resolve_node_at_lp(&world, lp)?;
                let origin = bevy_origin_of_layer_pos(lp, &anchor);
                Some((node_id, layer, origin))
            })
        } else {
            None
        };

    let Some((node_id, node_layer, origin)) = want else {
        despawn_current(&mut commands, &mut state);
        return;
    };

    // The baked mesh has vertices in `[0, 25]` on each axis, so the
    // node at `node_layer` lives at a Transform scale of
    // `scale_for_layer(node_layer)` (= `cell_size_at_layer(L) / 25`).
    // We want the overlay slightly bigger than the world block and
    // centred on it, so the translucent surface doesn't z-fight
    // against the opaque world render. Bias the scale up, then shift
    // the min corner back by half the extra width.
    let base_scale = scale_for_layer(node_layer);
    let scale = base_scale * OVERLAY_SCALE_BIAS;
    let cell = cell_size_at_layer(zoom.layer);
    let overlay_origin =
        origin - Vec3::splat((OVERLAY_SCALE_BIAS - 1.0) * cell * 0.5);
    let overlay_transform = Transform::from_translation(overlay_origin)
        .with_scale(Vec3::splat(scale));
    // `NODE_VOXELS_PER_AXIS` is the mesh's raw vertex extent; kept
    // here as a referenced constant so the relationship between
    // `base_scale` and `cell` is obvious to the reader.
    let _ = NODE_VOXELS_PER_AXIS;

    // Reuse the existing entity if the hovered node hasn't changed.
    if let Some((cur_id, entity)) = state.current {
        if cur_id == node_id {
            commands.entity(entity).insert(overlay_transform);
            return;
        }
        commands.entity(entity).despawn();
        state.current = None;
    }

    // Spawn a fresh overlay. Uses the cached bake from `RenderState`,
    // so the second and later hovers of the same node cost nothing.
    let baked = render_state
        .get_or_bake(&world, node_id, &mut meshes)
        .to_vec();

    let parent = commands
        .spawn((
            SaveHighlight,
            overlay_transform,
            Visibility::Visible,
        ))
        .id();

    for sub in &baked {
        commands.spawn((
            Mesh3d(sub.mesh.clone()),
            MeshMaterial3d(material.handle.clone()),
            Transform::default(),
            Visibility::Inherited,
            ChildOf(parent),
        ));
    }

    state.current = Some((node_id, parent));
}

fn despawn_current(
    commands: &mut Commands,
    state: &mut SaveHighlightState,
) {
    if let Some((_, entity)) = state.current.take() {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.despawn();
        }
    }
}

// ------------------------------------------------------------- helpers

/// True when the current view layer cleanly maps one view cell to one
/// tree node. At `L > MAX_LAYER - 2` a view cell is smaller than a
/// leaf, so there is no subtree to save — the user's "anything but
/// the highest layer" rule.
pub fn save_mode_eligible(view_layer: u8) -> bool {
    view_layer + 2 <= MAX_LAYER
}

/// Resolve a hovered `LayerPos` to the `NodeId` of the subtree that
/// one view cell covers. The cell `(cx, cy, cz)` at view layer `L`
/// decomposes into `slot_a = (c / 5)` (child of the layer-`L` node)
/// and `slot_b = (c % 5)` (child of the layer-`(L + 1)` node),
/// landing on a layer-`(L + 2)` subtree. Mirrors the rule in
/// `world::edit::edit_at_layer_pos` that makes bulk edits work at
/// the same view cell.
fn resolve_node_at_lp(
    world: &WorldState,
    lp: &LayerPos,
) -> Option<(NodeId, u8)> {
    if !save_mode_eligible(lp.layer) {
        return None;
    }

    let mut id = world.root;
    for &slot in &lp.path {
        let node = world.library.get(id)?;
        let children = node.children.as_ref()?;
        id = children[slot as usize];
        if id == EMPTY_NODE {
            return None;
        }
    }

    let b = BRANCH_FACTOR as u8;
    let slot_a = slot_index(
        (lp.cell[0] / b) as usize,
        (lp.cell[1] / b) as usize,
        (lp.cell[2] / b) as usize,
    );
    let node = world.library.get(id)?;
    let children = node.children.as_ref()?;
    id = children[slot_a];
    if id == EMPTY_NODE {
        return None;
    }

    let slot_b = slot_index(
        (lp.cell[0] % b) as usize,
        (lp.cell[1] % b) as usize,
        (lp.cell[2] % b) as usize,
    );
    let node = world.library.get(id)?;
    let children = node.children.as_ref()?;
    id = children[slot_b];
    if id == EMPTY_NODE {
        return None;
    }

    Some((id, lp.layer + 2))
}

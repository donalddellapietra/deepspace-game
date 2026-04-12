//! Save mode: hover a cell at a non-leaf view layer and stash the
//! underlying subtree as a reusable mesh.
//!
//! While `SaveMode.active`, the world block under the crosshair is
//! recoloured blue in place (by swapping its children's
//! `MeshMaterial3d` to [`SaveTintMaterial`]) and the normal outline
//! gizmo is drawn blue instead of white. There is no overlay mesh —
//! the existing rendered entity is what gets tinted, so the blue
//! exactly covers what the edit-mode outline would have covered.
//!
//! Left-clicking records the hovered `NodeId` into `SavedMeshes`,
//! fires a toast, and drops out of save mode. Only view layers
//! `L <= MAX_LAYER - 2` are eligible: any higher and one view cell
//! is smaller than a tree leaf, so there's no clean node to save.

use bevy::prelude::*;

use crate::block::{BslMaterial, Palette};
use crate::camera::CursorLocked;
use crate::interaction::TargetedBlock;
use crate::inventory::InventoryState;
use crate::world::edit::subtree_path_for_layer_pos;
use crate::world::position::LayerPos;
use crate::world::render::{SubMeshBlock, WorldRenderedNode};
use crate::world::tree::{NodeId, EMPTY_NODE, MAX_LAYER};
use crate::world::{CameraZoom, WorldState};

// -------------------------------------------------------------- resources

#[derive(Resource, Default)]
pub struct SaveMode {
    pub active: bool,
}

/// A saved subtree, referenced by its content-addressed `NodeId`.
///
/// Capturing a mesh bumps the library refcount on `node_id` (see
/// `save_on_click`), so the subtree survives in the library even
/// after the live world edits its original location away. Without
/// that pin, placing a captured mesh after even small unrelated
/// edits would panic in `install_subtree` — the captured id would
/// dangle.
#[derive(Clone, Debug)]
pub struct SavedMesh {
    pub node_id: NodeId,
    /// Tree layer of the saved node. Used by the inventory filter
    /// and by `place_block` to match saved meshes against the
    /// current placement target layer.
    pub layer: u8,
}

#[derive(Resource, Default)]
pub struct SavedMeshes {
    pub items: Vec<SavedMesh>,
}

/// A single bright-blue material reused for every tinted entity.
#[derive(Resource)]
pub struct SaveTintMaterial {
    pub handle: Handle<BslMaterial>,
}

/// Tracks the currently tinted rendered entity so we can restore
/// its children's original materials when the hover moves or save
/// mode exits.
#[derive(Resource, Default)]
pub struct SaveTintState {
    current_node: Option<NodeId>,
    current_entity: Option<Entity>,
}

// ------------------------------------------------------ material init

pub fn init_save_tint_material(
    mut commands: Commands,
    mut materials: ResMut<Assets<BslMaterial>>,
) {
    let handle = materials.add(BslMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.25, 0.5, 1.0),
            emissive: LinearRgba::new(0.15, 0.4, 1.1, 1.0),
            perceptual_roughness: 0.35,
            metallic: 0.1,
            ..default()
        },
        extension: Default::default(),
    });
    commands.insert_resource(SaveTintMaterial { handle });
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
    mut toast_writer: MessageWriter<crate::overlay::ToastEvent>,
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
    super::toast::show_toast(&mut toast_writer, message);

    save_mode.active = false;
}

// ----------------------------------------------------- tint system

/// Recolour the hovered world block by swapping each of its child
/// sub-meshes' `MeshMaterial3d` to the shared [`SaveTintMaterial`].
/// When the hover moves, restore the previously tinted entity's
/// children to their canonical block materials (looked up via the
/// `SubMeshBlock` marker the renderer attaches to each child).
///
/// This runs after `render::render_world` so we see the
/// freshly-spawned `WorldRenderedNode` entities on frames where the
/// hovered node was just brought into view.
pub fn update_save_tint(
    save_mode: Res<SaveMode>,
    targeted: Res<TargetedBlock>,
    zoom: Res<CameraZoom>,
    world: Res<WorldState>,
    palette: Option<Res<Palette>>,
    tint: Option<Res<SaveTintMaterial>>,
    mut state: ResMut<SaveTintState>,
    mut commands: Commands,
    rendered_q: Query<(Entity, &WorldRenderedNode)>,
    children_q: Query<&Children>,
    sub_q: Query<&SubMeshBlock>,
) {
    let (Some(palette), Some(tint)) = (palette, tint) else {
        return;
    };

    // What node (if any) should be tinted this frame? Rendered
    // entities live at emit_layer (target - 1), so resolve one
    // level above target to find the matching WorldRenderedNode.
    let target_node: Option<NodeId> =
        if save_mode.active && save_mode_eligible(zoom.layer) {
            targeted
                .hit_layer_pos
                .as_ref()
                .and_then(|lp| resolve_emit_node_at_lp(&world, lp))
        } else {
            None
        };

    // No change → no work. This is the common case: the hover
    // sits on the same node across many frames.
    if state.current_node == target_node {
        return;
    }

    // Restore whatever we tinted last frame. If the entity was
    // despawned by the renderer in the meantime, `get_entity`
    // returns Err and we silently drop it — the entity is gone, so
    // there's nothing to restore.
    if let Some(prev_entity) = state.current_entity.take() {
        if let Ok(children) = children_q.get(prev_entity) {
            for child in children.iter() {
                if let Ok(sub) = sub_q.get(child) {
                    if let Some(mat) = palette.material(sub.0) {
                        if let Ok(mut ec) = commands.get_entity(child) {
                            ec.insert(MeshMaterial3d(mat));
                        }
                    }
                }
            }
        }
    }

    state.current_node = target_node;

    let Some(target_node) = target_node else { return };

    // Find the rendered parent that matches our target. The number
    // of `WorldRenderedNode` entities is bounded by the render
    // radius (hundreds to low thousands), and this only runs on
    // hover-change, so a linear scan is fine.
    for (entity, rendered) in &rendered_q {
        if rendered.0 == target_node {
            if let Ok(children) = children_q.get(entity) {
                for child in children.iter() {
                    if sub_q.contains(child) {
                        if let Ok(mut ec) = commands.get_entity(child) {
                            ec.insert(MeshMaterial3d(tint.handle.clone()));
                        }
                    }
                }
            }
            state.current_entity = Some(entity);
            break;
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

/// Like [`resolve_node_at_lp`] but walks one fewer level, returning
/// the emit-layer node that *contains* the target-layer subtree.
/// Used for tinting: rendered entities live at emit_layer, so the
/// tint must match the parent of the target node.
fn resolve_emit_node_at_lp(
    world: &WorldState,
    lp: &LayerPos,
) -> Option<NodeId> {
    if !save_mode_eligible(lp.layer) {
        return None;
    }
    let path = subtree_path_for_layer_pos(lp);
    if path.len() < 2 {
        return None;
    }
    let emit_path = &path[..path.len() - 1];
    let mut id = world.root;
    for &slot in emit_path {
        let node = world.library.get(id)?;
        let children = node.children.as_ref()?;
        id = children[slot as usize];
        if id == EMPTY_NODE {
            return None;
        }
    }
    Some(id)
}

/// Resolve a hovered `LayerPos` to the `NodeId` of the subtree that
/// one view cell covers, plus the tree layer that subtree lives at.
/// Shares the path construction with `place_block` via the helper
/// in `world::edit` so there's one canonical slot-decomposition.
fn resolve_node_at_lp(
    world: &WorldState,
    lp: &LayerPos,
) -> Option<(NodeId, u8)> {
    if !save_mode_eligible(lp.layer) {
        return None;
    }
    let path = subtree_path_for_layer_pos(lp);
    let mut id = world.root;
    for &slot in &path {
        let node = world.library.get(id)?;
        let children = node.children.as_ref()?;
        id = children[slot as usize];
        if id == EMPTY_NODE {
            return None;
        }
    }
    // `save_mode_eligible` guarantees `path.len() == lp.layer + 2`.
    Some((id, path.len() as u8))
}

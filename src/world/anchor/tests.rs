use super::*;
use crate::world::tree::{slot_index, NodeLibrary};

fn lib() -> NodeLibrary {
    NodeLibrary::default()
}

/// Sentinel "no real world" root for the kind-agnostic path.
/// `node_kind_at_depth` returns `None` for missing nodes, so
/// `add_local` falls through to the Cartesian bubble — equivalent
/// to the pre-Phase-2 behavior. Used by the existing tests below.
const NO_ROOT: crate::world::tree::NodeId = 0;

#[test]
fn path_root_and_push_pop() {
    let mut p = Path::root();
    assert_eq!(p.depth(), 0);
    assert!(p.is_root());
    p.push(13);
    p.push(5);
    assert_eq!(p.depth(), 2);
    assert_eq!(p.as_slice(), &[13, 5]);
    assert_eq!(p.pop(), Some(5));
    assert_eq!(p.pop(), Some(13));
    assert_eq!(p.pop(), None);
}

#[test]
fn path_eq_and_hash() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut a = Path::root();
    a.push(1);
    a.push(2);
    let mut b = Path::root();
    b.push(1);
    b.push(2);
    // c shares prefix but differs in depth.
    let mut c = Path::root();
    c.push(1);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let hash = |p: &Path| -> u64 {
        let mut h = DefaultHasher::new();
        p.hash(&mut h);
        h.finish()
    };
    assert_eq!(hash(&a), hash(&b));
}

#[test]
fn common_prefix() {
    let mut a = Path::root();
    let mut b = Path::root();
    for s in [1u8, 2, 3, 4] { a.push(s); }
    for s in [1u8, 2, 9, 0] { b.push(s); }
    assert_eq!(a.common_prefix_len(&b), 2);
}

#[test]
fn step_neighbor_within_cell() {
    // At depth 2 starting at slot (1,1,1), step +x -> (2,1,1).
    let mut p = Path::root();
    p.push(0);
    p.push(slot_index(1, 1, 1) as u8);
    p.step_neighbor_cartesian(0, 1);
    assert_eq!(p.slot(1), slot_index(2, 1, 1) as u8);
    p.step_neighbor_cartesian(0, -1);
    assert_eq!(p.slot(1), slot_index(1, 1, 1) as u8);
}

#[test]
fn step_neighbor_bubbles_up() {
    // Depth 2 at (0, 0, 0) within parent (0, 0, 0). Step -x
    // should bubble up; parent is already at x=0 of root, so
    // root step is clamped (no-op), and the child slot should
    // be rewritten as if we crossed the boundary (to x=2).
    let mut p = Path::root();
    p.push(slot_index(1, 1, 1) as u8);
    p.push(slot_index(0, 1, 1) as u8);
    p.step_neighbor_cartesian(0, -1);
    // Parent stepped from (1,1,1) to (0,1,1); child wrapped to (2,1,1).
    assert_eq!(p.slot(0), slot_index(0, 1, 1) as u8);
    assert_eq!(p.slot(1), slot_index(2, 1, 1) as u8);
}

#[test]
fn zoom_round_trip() {
    let anchor = {
        let mut p = Path::root();
        p.push(5);
        p
    };
    let mut pos = WorldPos::new(anchor, [0.25, 0.5, 0.75]);
    let before = pos;
    pos.zoom_in();
    assert_eq!(pos.anchor.depth(), 2);
    pos.zoom_out();
    assert_eq!(pos.anchor, before.anchor);
    for i in 0..3 {
        assert!((pos.offset[i] - before.offset[i]).abs() < 1e-5);
    }
}

#[test]
fn zoom_in_preserves_invariant() {
    // Offset at 1-eps corners still ends up in [0, 1).
    let anchor = Path::root();
    let mut pos = WorldPos::new(anchor, [1.0 - f32::EPSILON; 3]);
    pos.zoom_in();
    for v in pos.offset.iter() {
        assert!((0.0..1.0).contains(v), "offset {} out of range", v);
    }
}

#[test]
fn zoom_out_at_root_is_noop() {
    let mut pos = WorldPos::new(Path::root(), [0.1, 0.2, 0.3]);
    let before = pos;
    pos.zoom_out();
    assert_eq!(pos, before);
}

#[test]
fn add_local_small_delta() {
    let l = lib();
    let mut pos = WorldPos::new(Path::root(), [0.5, 0.5, 0.5]);
    let t = pos.add_local([0.1, 0.0, 0.0], &l, NO_ROOT);
    assert_eq!(t, Transition::None);
    assert!((pos.offset[0] - 0.6).abs() < 1e-5);
    assert_eq!(pos.anchor, Path::root());
}

#[test]
fn add_local_crosses_cell_boundary() {
    let l = lib();
    // At depth 1, slot = (1,1,1), offset near x=1. Step +x.
    let mut anchor = Path::root();
    anchor.push(slot_index(1, 1, 1) as u8);
    let mut pos = WorldPos::new(anchor, [0.9, 0.5, 0.5]);
    pos.add_local([0.2, 0.0, 0.0], &l, NO_ROOT);
    assert_eq!(pos.anchor.slot(0), slot_index(2, 1, 1) as u8);
    assert!((pos.offset[0] - 0.1).abs() < 1e-4);
}

#[test]
fn add_local_bubbles_up_parent() {
    let l = lib();
    // Depth 2; child at (2,1,1) of parent (1,1,1). Step +x
    // overflows child; parent becomes (2,1,1); child becomes (0,1,1).
    let mut anchor = Path::root();
    anchor.push(slot_index(1, 1, 1) as u8);
    anchor.push(slot_index(2, 1, 1) as u8);
    let mut pos = WorldPos::new(anchor, [0.9, 0.5, 0.5]);
    pos.add_local([0.2, 0.0, 0.0], &l, NO_ROOT);
    assert_eq!(pos.anchor.slot(0), slot_index(2, 1, 1) as u8);
    assert_eq!(pos.anchor.slot(1), slot_index(0, 1, 1) as u8);
    assert!((pos.offset[0] - 0.1).abs() < 1e-4);
}

#[test]
fn add_local_large_negative_delta() {
    let l = lib();
    // Step back across two cells.
    let mut anchor = Path::root();
    anchor.push(slot_index(2, 1, 1) as u8);
    let mut pos = WorldPos::new(anchor, [0.1, 0.5, 0.5]);
    pos.add_local([-1.2, 0.0, 0.0], &l, NO_ROOT);
    // From slot (2,1,1) step back 2 cells -> (0,1,1); offset
    // becomes 0.1 - 1.2 + 2 = 0.9.
    assert_eq!(pos.anchor.slot(0), slot_index(0, 1, 1) as u8);
    assert!((pos.offset[0] - 0.9).abs() < 1e-4);
}

// ---- zoom / position preservation tests ----
// These use in_frame(&Path::root()) to verify position is
// unchanged — equivalent to the old to_world_xyz() at shallow
// depths where f32 is precise.

#[test]
fn zoom_preserves_position() {
    let mut p = WorldPos::from_frame_local(&Path::root(), [1.23, 2.34, 0.56], 5);
    let before = p.in_frame(&Path::root());
    p.zoom_in();
    let after_in = p.in_frame(&Path::root());
    for i in 0..3 {
        assert!((before[i] - after_in[i]).abs() < 1e-4);
    }
    p.zoom_out();
    let after_out = p.in_frame(&Path::root());
    for i in 0..3 {
        assert!((before[i] - after_out[i]).abs() < 1e-4);
    }
}

#[test]
fn zoom_in_then_zoom_out_preserves_position() {
    let mut p = WorldPos::from_frame_local(&Path::root(), [1.234, 2.345, 0.567], 4);
    let before = p.in_frame(&Path::root());
    for _ in 0..16 { p.zoom_in(); }
    for _ in 0..16 { p.zoom_out(); }
    let after = p.in_frame(&Path::root());
    for i in 0..3 {
        assert!((after[i] - before[i]).abs() < 1e-4,
            "axis {}: {} -> {}", i, before[i], after[i]);
    }
}

#[test]
fn many_zoom_ins_preserve_position() {
    let mut p = WorldPos::from_frame_local(&Path::root(), [1.234, 2.345, 0.567], 4);
    let before = p.in_frame(&Path::root());
    for k in 0..15 {
        p.zoom_in();
        let after = p.in_frame(&Path::root());
        for i in 0..3 {
            assert!((after[i] - before[i]).abs() < 1e-4,
                "after {} zoom_ins, axis {}: {} -> {}",
                k + 1, i, before[i], after[i]);
        }
    }
}

#[test]
fn deepened_to_preserves_position() {
    let p = WorldPos::from_frame_local(&Path::root(), [1.234, 2.345, 0.567], 4);
    let before = p.in_frame(&Path::root());
    for d in [4u8, 6, 8, 12] {
        let q = p.deepened_to(d);
        let after = q.in_frame(&Path::root());
        for i in 0..3 {
            assert!((before[i] - after[i]).abs() < 1e-4,
                "depth {}: axis {}: {} vs {}", d, i, before[i], after[i]);
        }
    }
}

// ---- in_frame tests ----

#[test]
fn in_frame_at_root_gives_expected_coords() {
    // At shallow depth, root-frame-local coords match the input.
    let p = WorldPos::from_frame_local(&Path::root(), [1.5, 2.25, 0.75], 7);
    let local = p.in_frame(&Path::root());
    assert!((local[0] - 1.5).abs() < 1e-4);
    assert!((local[1] - 2.25).abs() < 1e-4);
    assert!((local[2] - 0.75).abs() < 1e-4);
}

#[test]
fn in_frame_round_trip_via_from_frame_local() {
    let p = WorldPos::from_frame_local(&Path::root(), [1.5, 2.1, 0.9], 12);
    let mut frame = p.anchor;
    frame.truncate(frame.depth() - 3);
    let local = p.in_frame(&frame);
    let q = WorldPos::from_frame_local(&frame, local, p.anchor.depth());
    // Both should project to the same root-frame coords.
    let back = q.in_frame(&Path::root());
    let orig = p.in_frame(&Path::root());
    for i in 0..3 {
        assert!((back[i] - orig[i]).abs() < 1e-4);
    }
}

#[test]
fn in_frame_cross_branch() {
    // Point and frame in different depth-1 branches: the returned
    // local coords fall outside [0, WORLD_SIZE) because the point
    // is outside the frame's cell.
    let point = WorldPos::from_frame_local(&Path::root(), [2.5, 0.25, 1.5], 4);
    let mut frame = Path::root();
    frame.push(slot_index(0, 2, 0) as u8); // depth-1 cell (0, 2, 0)
    frame.push(slot_index(1, 1, 1) as u8); // depth-2 center within it
    let actual = point.in_frame(&frame);
    // Point is at (2.5, 0.25, 1.5) in root frame.
    // Frame cell origin is (0+1/3, 2+1/3, 0+1/3) = (1/3, 7/3, 1/3) in root frame.
    // Frame cell size is 1/9 in root frame.
    // Frame local = (root_pos - frame_origin) / frame_cell_size * WORLD_SIZE
    // The exact values depend on slot arithmetic, but the point should
    // be well outside [0, WORLD_SIZE) on at least one axis.
    assert!(actual[0] > WORLD_SIZE || actual[1] < 0.0,
        "cross-branch point should be outside frame bounds: {:?}", actual);
}

#[test]
fn in_frame_precision_at_deep_anchor() {
    // Construct at shallow depth then deepen — the frame-local
    // coord should stay within [0, WORLD_SIZE) since the anchor
    // shares a deep prefix with the frame.
    let p = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4)
        .deepened_to(18);
    let mut frame = p.anchor;
    frame.truncate(frame.depth() - 3);
    let local = p.in_frame(&frame);
    for &v in &local {
        assert!((0.0..super::WORLD_SIZE).contains(&v), "local {v} out of frame");
    }
}

// ---- offset_from tests ----

#[test]
fn offset_from_consistent_across_depths() {
    // Construct at shallow depth then deepen via zoom_in (always
    // precise). offset_from should give the same result regardless
    // of anchor depth.
    let planet = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
    let cam_shallow = WorldPos::from_frame_local(&Path::root(), [1.5, 2.32, 1.5], 4);
    let baseline = cam_shallow.offset_from(&planet);
    assert!((baseline[1] - 0.82).abs() < 1e-4);
    for d in [4u8, 8, 12, 16, 20] {
        let cam = cam_shallow.deepened_to(d);
        let oc = cam.offset_from(&planet);
        for i in 0..3 {
            assert!((oc[i] - baseline[i]).abs() < 1e-4,
                "depth {d}: axis {i}: {} vs baseline {}", oc[i], baseline[i]);
        }
    }
}

#[test]
fn offset_from_after_zoom_chain_matches_baseline() {
    let planet = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
    let cam = WorldPos::from_frame_local(&Path::root(), [1.5, 2.32, 1.5], 4)
        .deepened_to(16);
    let mut zoomed = cam;
    for _ in 0..7 { zoomed.zoom_out(); }
    assert_eq!(zoomed.anchor.depth(), 9);
    let oc_chained = zoomed.offset_from(&planet);
    let oc_deep = cam.offset_from(&planet);
    for i in 0..3 {
        assert!(
            (oc_chained[i] - oc_deep[i]).abs() < 1e-4,
            "axis {}: chained {} vs deep {}",
            i, oc_chained[i], oc_deep[i],
        );
    }
    assert!(oc_chained[1].abs() > 0.5,
        "oc.y collapsed to 0 after zoom chain — sphere would be invisible");
}

#[test]
fn offset_from_self_is_zero() {
    let base = WorldPos::from_frame_local(&Path::root(), [1.5, 2.0, 0.7], 4);
    for d in [4u8, 8, 12, 16] {
        let p = base.deepened_to(d);
        let o = p.offset_from(&p);
        for v in o {
            assert!(v.abs() < 1e-6, "depth {}: o = {:?}", d, o);
        }
    }
}

#[test]
fn offset_from_is_antisymmetric() {
    let a = WorldPos::from_frame_local(&Path::root(), [1.5, 2.0, 0.7], 8);
    let b = WorldPos::from_frame_local(&Path::root(), [0.5, 1.5, 1.5], 8);
    let ab = a.offset_from(&b);
    let ba = b.offset_from(&a);
    for i in 0..3 {
        assert!((ab[i] + ba[i]).abs() < 1e-5,
            "axis {}: ab={} ba={}", i, ab[i], ba[i]);
    }
}

#[test]
fn offset_from_satisfies_triangle_equality() {
    let a = WorldPos::from_frame_local(&Path::root(), [0.5, 1.5, 1.5], 6);
    let b = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 6);
    let c = WorldPos::from_frame_local(&Path::root(), [2.0, 1.5, 1.5], 6);
    let ac = a.offset_from(&c);
    let ab = a.offset_from(&b);
    let bc = b.offset_from(&c);
    for i in 0..3 {
        let sum = ab[i] + bc[i];
        assert!((ac[i] - sum).abs() < 1e-5,
            "axis {}: ac={} ab+bc={}", i, ac[i], sum);
    }
}

#[test]
fn offset_from_invariant_under_anchor_depth() {
    // Construct at depth 4 then deepen. offset_from should be
    // consistent because deepened_to is pure slot arithmetic.
    let target = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
    let base = WorldPos::from_frame_local(&Path::root(), [1.5, 2.0, 0.7], 4);
    let baseline = base.offset_from(&target);
    for depth in [4u8, 6, 8, 12, 16, 20] {
        let p = base.deepened_to(depth);
        let o = p.offset_from(&target);
        for i in 0..3 {
            assert!(
                (o[i] - baseline[i]).abs() < 1e-5,
                "depth {}: axis {}: {} vs baseline {}",
                depth, i, o[i], baseline[i],
            );
        }
    }
}

#[test]
fn deepened_offset_from_matches_base() {
    let target = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
    let base = WorldPos::from_frame_local(&Path::root(), [1.5, 2.0, 0.7], 4);
    let base_o = base.offset_from(&target);
    for d in [4u8, 6, 8, 12, 16, 20] {
        let deeper = base.deepened_to(d);
        let o = deeper.offset_from(&target);
        for i in 0..3 {
            assert!((o[i] - base_o[i]).abs() < 1e-5,
                "depth {}: axis {}: {} vs base {}",
                d, i, o[i], base_o[i]);
        }
    }
}

#[test]
fn offset_from_matches_in_frame_diff_at_shallow_anchors() {
    // At shallow depth, offset_from(b) should equal the
    // root-frame coordinate difference.
    let a = WorldPos::from_frame_local(&Path::root(), [2.5, 0.25, 1.5], 4);
    let b = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 1.5], 4);
    let o = a.offset_from(&b);
    let aw = a.in_frame(&Path::root());
    let bw = b.in_frame(&Path::root());
    for i in 0..3 {
        assert!((o[i] - (aw[i] - bw[i])).abs() < 1e-5);
    }
}

#[test]
fn offset_from_precision_at_deep_common_prefix() {
    // Two positions inside the same depth-12 cell — common
    // prefix is 12, so the offset resolves at sub-cell precision
    // even though root-frame coords would lose precision.
    let mut anchor = Path::root();
    for _ in 0..12 { anchor.push(slot_index(1, 1, 1) as u8); }
    let a = WorldPos::new(anchor, [0.30, 0.50, 0.70]);
    let b = WorldPos::new(anchor, [0.20, 0.50, 0.70]);
    let o = a.offset_from(&b);
    let cell = WORLD_SIZE / 3.0f32.powi(12);
    let expected_x = 0.10 * cell;
    assert!((o[0] - expected_x).abs() < cell * 1e-5,
        "diff {} expected {}", o[0], expected_x);
    assert!(o[1].abs() < cell * 1e-5);
    assert!(o[2].abs() < cell * 1e-5);
}

#[test]
fn add_local_offset_is_normalized() {
    let l = lib();
    let mut pos = WorldPos::new(Path::root(), [0.0, 0.0, 0.0]);
    pos.add_local([0.3, 0.7, 0.999], &l, NO_ROOT);
    for &v in &pos.offset {
        assert!((0.0..1.0).contains(&v));
    }
}

/// Hitting the world boundary clamps the position but must NOT
/// collapse the anchor to depth 0. The renormalize loop has to
/// redescend after the clamp so the camera ends up in the boundary
/// cell at the original target depth, not floating at empty-root.
#[test]
fn add_local_at_world_boundary_preserves_anchor_depth() {
    let l = lib();
    let mut anchor = Path::root();
    anchor.push(crate::world::tree::slot_index(1, 2, 1) as u8);
    anchor.push(crate::world::tree::slot_index(1, 2, 1) as u8);
    let mut pos = WorldPos::new(anchor, [0.5, 0.5, 0.5]);
    let target_depth = pos.anchor.depth();
    // Push offset[1] way past 1.0 — equivalent to the camera trying
    // to leave the world via the +Y face. The renormalize must
    // clamp at the world boundary AND re-walk back down to the
    // boundary cell at target_depth.
    pos.add_local([0.0, 5.0, 0.0], &l, NO_ROOT);
    assert_eq!(pos.anchor.depth(), target_depth,
        "world-boundary clamp collapsed anchor: {:?}", pos.anchor.as_slice());
    for &v in &pos.offset {
        assert!((0.0..1.0).contains(&v),
            "offset out of range after boundary clamp: {:?}", pos.offset);
    }
}

/// Crossing into a `TangentBlock` ancestor must preserve world
/// position. Without the kind-boundary fixup in `renormalize_world`,
/// `step_neighbor` inherits source-cell slot indices in the wrong
/// frame and the camera teleports off-axis the moment its anchor
/// crosses the rotated cube's face.
#[test]
fn add_local_preserves_world_pos_across_tangent_block_boundary() {
    use crate::world::bootstrap::rotated_cube_test_world;
    let world = rotated_cube_test_world();
    // Spawn just above the +Y face of the rotated middle cube
    // (slot 13 of root = world cell [1,2]³).
    // Anchor depth 3 puts the camera in a depth-3 cell of root,
    // matching the user's reported repro.
    let mut pos = WorldPos::from_world_xyz(
        [1.26083, 2.03101, 1.26498], 3, &world.library, world.root,
    );
    let world_before = pos.in_frame_rot(
        &world.library, world.root, &Path::root(),
    );
    // Sanity: the constructor lands at the requested world pos.
    for i in 0..3 {
        assert!((world_before[i] - [1.26083, 2.03101, 1.26498][i]).abs() < 1e-4,
            "from_world_xyz lost precision at axis {}: got {:?} want {:?}",
            i, world_before, [1.26083, 2.03101, 1.26498]);
    }
    // Move just enough to cross the y=2 boundary into the TB.
    // `add_local` takes delta in deepest-cell offset units (`[0, 1)`
    // = one cell), so to move world Y by -0.06 at depth 3 (cell
    // size = 3 / 3^3 = 1/9 world per offset unit) we pass
    // `dy_offset = -0.06 * 9 = -0.54`.
    let cell_world = WORLD_SIZE / 3.0_f32.powi(pos.anchor.depth() as i32);
    let dy_world = -0.06; // crosses y=2 boundary into TB
    let dy_offset = dy_world / cell_world;
    pos.add_local([0.0, dy_offset, 0.0], &world.library, world.root);
    let world_after = pos.in_frame_rot(
        &world.library, world.root, &Path::root(),
    );
    // World Y should drop by ~|dy_world|. X and Z must stay
    // continuous — the user's reported teleport jumped X by -0.10
    // and Z by +0.24 at the boundary.
    let tol = 1e-3;
    assert!((world_after[0] - world_before[0]).abs() < tol,
        "X teleported across TB boundary: before {} after {}",
        world_before[0], world_after[0]);
    assert!((world_after[2] - world_before[2]).abs() < tol,
        "Z teleported across TB boundary: before {} after {}",
        world_before[2], world_after[2]);
    assert!((world_after[1] - (world_before[1] + dy_world)).abs() < tol,
        "Y didn't track the requested delta: before {} after {} dy_world {}",
        world_before[1], world_after[1], dy_world);
    // Sanity: anchor should now have the TB on its path (slot 13).
    assert_eq!(pos.anchor.slot(0), 13,
        "after crossing into the TB, anchor[0] should be 13 (TB slot)");
}

/// At deep anchors (where each cell is well below f32 epsilon in
/// world units), `add_local` must remain stable — the precision of
/// the camera position is bounded by the cell-fraction, not by the
/// absolute world coordinate. The cell-local pop+redescend in
/// `renormalize_world` should keep arithmetic magnitudes ≤ 0.5
/// regardless of depth. World-absolute snap algorithms (e.g. lifting
/// through `in_frame_rot` to root and re-walking from XYZ) lose all
/// information about the offset by ~depth 25.
#[test]
fn add_local_is_precision_stable_at_deep_anchor() {
    let l = lib();
    // Plain Cartesian world (no TB needed for this precision check).
    let mut pos = WorldPos::uniform_column(13, 28, [0.5, 0.5, 0.5]);
    // Move by a fraction of the deepest cell. The same offset delta
    // moves the same fraction of the cell at any depth — that's the
    // whole point of cell-local arithmetic.
    let dx = 0.1_f32;
    let off_before = pos.offset;
    pos.add_local([dx, 0.0, 0.0], &l, NO_ROOT);
    let off_after = pos.offset;
    // No cell-boundary crossing: offset should advance by exactly dx
    // (within f32 epsilon).
    assert!((off_after[0] - (off_before[0] + dx)).abs() < 1e-6,
        "deep-anchor add_local lost precision: before {} after {} dx {}",
        off_before[0], off_after[0], dx);
}

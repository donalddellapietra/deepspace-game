use super::*;
use crate::world::tree::{slot_index, NodeLibrary};

fn lib() -> NodeLibrary {
    NodeLibrary::default()
}

const NO_ROOT: crate::world::tree::NodeId = 0;

#[test]
fn path_root_and_push_pop() {
    let mut p = Path::root();
    assert_eq!(p.depth(), 0);
    assert!(p.is_root());
    p.push(5);
    p.push(3);
    assert_eq!(p.depth(), 2);
    assert_eq!(p.as_slice(), &[5, 3]);
    assert_eq!(p.pop(), Some(3));
    assert_eq!(p.pop(), Some(5));
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
    for s in [1u8, 2, 7, 0] { b.push(s); }
    assert_eq!(a.common_prefix_len(&b), 2);
}

#[test]
fn step_neighbor_within_cell() {
    // In base-2, slot (0,0,0) step +x → (1,0,0).
    let mut p = Path::root();
    p.push(0);
    p.push(slot_index(0, 0, 0) as u8);
    p.step_neighbor_cartesian(0, 1);
    assert_eq!(p.slot(1), slot_index(1, 0, 0) as u8);
    p.step_neighbor_cartesian(0, -1);
    assert_eq!(p.slot(1), slot_index(0, 0, 0) as u8);
}

#[test]
fn step_neighbor_bubbles_up() {
    // Depth 2, child at (0,1,1) of parent (1,1,1). Step -x
    // overflows; parent steps from (1,1,1) to (0,1,1), child wraps to (1,1,1).
    let mut p = Path::root();
    p.push(slot_index(1, 1, 1) as u8);
    p.push(slot_index(0, 1, 1) as u8);
    p.step_neighbor_cartesian(0, -1);
    assert_eq!(p.slot(0), slot_index(0, 1, 1) as u8);
    assert_eq!(p.slot(1), slot_index(1, 1, 1) as u8);
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
    let mut anchor = Path::root();
    anchor.push(slot_index(0, 1, 1) as u8);
    let mut pos = WorldPos::new(anchor, [0.9, 0.5, 0.5]);
    pos.add_local([0.2, 0.0, 0.0], &l, NO_ROOT);
    assert_eq!(pos.anchor.slot(0), slot_index(1, 1, 1) as u8);
    assert!((pos.offset[0] - 0.1).abs() < 1e-4);
}

#[test]
fn add_local_bubbles_up_parent() {
    let l = lib();
    // Depth 2; child at (1,1,1) of parent (0,1,1). Step +x
    // overflows child; parent becomes (1,1,1); child becomes (0,1,1).
    let mut anchor = Path::root();
    anchor.push(slot_index(0, 1, 1) as u8);
    anchor.push(slot_index(1, 1, 1) as u8);
    let mut pos = WorldPos::new(anchor, [0.9, 0.5, 0.5]);
    pos.add_local([0.2, 0.0, 0.0], &l, NO_ROOT);
    assert_eq!(pos.anchor.slot(0), slot_index(1, 1, 1) as u8);
    assert_eq!(pos.anchor.slot(1), slot_index(0, 1, 1) as u8);
    assert!((pos.offset[0] - 0.1).abs() < 1e-4);
}

#[test]
fn add_local_large_negative_delta() {
    let l = lib();
    let mut anchor = Path::root();
    anchor.push(slot_index(1, 1, 1) as u8);
    let mut pos = WorldPos::new(anchor, [0.1, 0.5, 0.5]);
    pos.add_local([-1.2, 0.0, 0.0], &l, NO_ROOT);
    // From slot (1,1,1) step back → wraps. With base-2 there's only
    // slots 0 and 1 on x. step -1 from x=1 lands on x=0 (no bubble).
    // Then another -0.2 overflows x=0, bubble to root (fails), clamp.
    // The exact landing depends on boundary clamping. Just verify
    // offset is normalized and reasonable.
    assert!((0.0..1.0).contains(&pos.offset[0]));
}

// ---- zoom / position preservation tests ----

#[test]
fn zoom_preserves_position() {
    let mut p = WorldPos::from_frame_local(&Path::root(), [0.73, 1.34, 0.56], 5);
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
    let mut p = WorldPos::from_frame_local(&Path::root(), [0.734, 1.345, 0.567], 4);
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
    let mut p = WorldPos::from_frame_local(&Path::root(), [0.734, 1.345, 0.567], 4);
    let before = p.in_frame(&Path::root());
    for k in 0..20 {
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
    let p = WorldPos::from_frame_local(&Path::root(), [0.734, 1.345, 0.567], 4);
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
    let p = WorldPos::from_frame_local(&Path::root(), [0.5, 1.25, 0.75], 7);
    let local = p.in_frame(&Path::root());
    assert!((local[0] - 0.5).abs() < 1e-4);
    assert!((local[1] - 1.25).abs() < 1e-4);
    assert!((local[2] - 0.75).abs() < 1e-4);
}

#[test]
fn in_frame_round_trip_via_from_frame_local() {
    let p = WorldPos::from_frame_local(&Path::root(), [0.5, 1.1, 0.9], 12);
    let mut frame = p.anchor;
    frame.truncate(frame.depth() - 3);
    let local = p.in_frame(&frame);
    let q = WorldPos::from_frame_local(&frame, local, p.anchor.depth());
    let back = q.in_frame(&Path::root());
    let orig = p.in_frame(&Path::root());
    for i in 0..3 {
        assert!((back[i] - orig[i]).abs() < 1e-4);
    }
}

#[test]
fn in_frame_cross_branch() {
    let point = WorldPos::from_frame_local(&Path::root(), [1.5, 0.25, 0.5], 4);
    let mut frame = Path::root();
    frame.push(slot_index(0, 1, 0) as u8);
    frame.push(slot_index(1, 1, 1) as u8);
    let actual = point.in_frame(&frame);
    assert!(actual[0] > WORLD_SIZE || actual[1] < 0.0,
        "cross-branch point should be outside frame bounds: {:?}", actual);
}

#[test]
fn in_frame_precision_at_deep_anchor() {
    let p = WorldPos::from_frame_local(&Path::root(), [1.0, 1.0, 1.0], 4)
        .deepened_to(30);
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
    let planet = WorldPos::from_frame_local(&Path::root(), [1.0, 1.0, 1.0], 4);
    let cam_shallow = WorldPos::from_frame_local(&Path::root(), [1.0, 1.82, 1.0], 4);
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
    let planet = WorldPos::from_frame_local(&Path::root(), [1.0, 1.0, 1.0], 4);
    let cam = WorldPos::from_frame_local(&Path::root(), [1.0, 1.82, 1.0], 4)
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
    let base = WorldPos::from_frame_local(&Path::root(), [1.0, 1.5, 0.7], 4);
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
    let a = WorldPos::from_frame_local(&Path::root(), [1.5, 1.5, 0.7], 8);
    let b = WorldPos::from_frame_local(&Path::root(), [0.5, 1.0, 1.0], 8);
    let ab = a.offset_from(&b);
    let ba = b.offset_from(&a);
    for i in 0..3 {
        assert!((ab[i] + ba[i]).abs() < 1e-5,
            "axis {}: ab={} ba={}", i, ab[i], ba[i]);
    }
}

#[test]
fn offset_from_satisfies_triangle_equality() {
    let a = WorldPos::from_frame_local(&Path::root(), [0.5, 1.0, 1.0], 6);
    let b = WorldPos::from_frame_local(&Path::root(), [1.0, 1.0, 1.0], 6);
    let c = WorldPos::from_frame_local(&Path::root(), [1.5, 1.0, 1.0], 6);
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
    let target = WorldPos::from_frame_local(&Path::root(), [1.0, 1.0, 1.0], 4);
    let base = WorldPos::from_frame_local(&Path::root(), [1.0, 1.5, 0.7], 4);
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
    let target = WorldPos::from_frame_local(&Path::root(), [1.0, 1.0, 1.0], 4);
    let base = WorldPos::from_frame_local(&Path::root(), [1.0, 1.5, 0.7], 4);
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
    let a = WorldPos::from_frame_local(&Path::root(), [1.5, 0.25, 0.5], 4);
    let b = WorldPos::from_frame_local(&Path::root(), [1.0, 1.0, 0.5], 4);
    let o = a.offset_from(&b);
    let aw = a.in_frame(&Path::root());
    let bw = b.in_frame(&Path::root());
    for i in 0..3 {
        assert!((o[i] - (aw[i] - bw[i])).abs() < 1e-5);
    }
}

#[test]
fn offset_from_precision_at_deep_common_prefix() {
    let mut anchor = Path::root();
    for _ in 0..12 { anchor.push(slot_index(1, 1, 1) as u8); }
    let a = WorldPos::new(anchor, [0.30, 0.50, 0.70]);
    let b = WorldPos::new(anchor, [0.20, 0.50, 0.70]);
    let o = a.offset_from(&b);
    let cell = WORLD_SIZE / 2.0f32.powi(12);
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

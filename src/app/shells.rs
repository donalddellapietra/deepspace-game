use crate::world::anchor::Path;
use crate::world::tree::MAX_DEPTH;

pub const MAX_CARTESIAN_SHELLS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CartesianShellSpec {
    pub ancestor_steps: u8,
    pub min_depth_limit: u8,
}

// Tunable first-pass local shell schedule:
// shell 0 = deepest local shell
// shell 1 = parent context shell
// shell 2 = farther context shell
pub const DEFAULT_CARTESIAN_SHELL_SPECS: [CartesianShellSpec; 3] = [
    CartesianShellSpec {
        ancestor_steps: 0,
        min_depth_limit: 10,
    },
    CartesianShellSpec {
        ancestor_steps: 3,
        min_depth_limit: 4,
    },
    CartesianShellSpec {
        ancestor_steps: 6,
        min_depth_limit: 3,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CartesianShell {
    pub path: Path,
    pub ribbon_level: u32,
    pub depth_limit: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CartesianShellStack {
    shells: Vec<CartesianShell>,
}

impl CartesianShellStack {
    pub fn build(deepest_path: Path, base_visual_depth: u32) -> Self {
        Self::build_with_specs(
            deepest_path,
            base_visual_depth,
            &DEFAULT_CARTESIAN_SHELL_SPECS,
        )
    }

    pub fn build_with_specs(
        deepest_path: Path,
        base_visual_depth: u32,
        specs: &[CartesianShellSpec],
    ) -> Self {
        let deepest_depth = deepest_path.depth() as u32;
        let target_depth = (deepest_depth + base_visual_depth.max(1)).min(MAX_DEPTH as u32);
        let mut shells: Vec<CartesianShell> =
            Vec::with_capacity(specs.len().min(MAX_CARTESIAN_SHELLS));

        for spec in specs.iter().copied().take(MAX_CARTESIAN_SHELLS) {
            let shell_depth = deepest_path.depth().saturating_sub(spec.ancestor_steps);
            let ribbon_level = deepest_depth.saturating_sub(shell_depth as u32);
            let required_depth_limit = target_depth.saturating_sub(shell_depth as u32).max(1);
            let depth_limit = required_depth_limit
                .max(spec.min_depth_limit as u32)
                .min(MAX_DEPTH as u32);
            let path = ancestor_path(deepest_path, shell_depth);
            if let Some(last) = shells.last_mut() {
                if last.ribbon_level == ribbon_level {
                    last.depth_limit = last.depth_limit.max(depth_limit);
                    continue;
                }
            }
            shells.push(CartesianShell {
                path,
                ribbon_level,
                depth_limit,
            });
        }

        Self { shells }
    }

    pub fn shells(&self) -> &[CartesianShell] {
        &self.shells
    }

    pub fn deepest_path(&self) -> Path {
        self.shells
            .first()
            .map(|shell| shell.path)
            .unwrap_or(Path::root())
    }

    pub fn deepest_depth_limit(&self) -> u32 {
        self.shells
            .first()
            .map(|shell| shell.depth_limit)
            .unwrap_or(1)
    }

    pub fn preserve_regions(&self) -> Vec<(Path, u8)> {
        self.shells
            .iter()
            .map(|shell| (shell.path, shell.depth_limit.min(u8::MAX as u32) as u8))
            .collect()
    }

    pub fn shader_pairs(&self) -> ([[u32; 4]; MAX_CARTESIAN_SHELLS / 2], u32) {
        let mut pairs = [[0u32; 4]; MAX_CARTESIAN_SHELLS / 2];
        for (idx, shell) in self.shells.iter().take(MAX_CARTESIAN_SHELLS).enumerate() {
            let pair = &mut pairs[idx / 2];
            if idx % 2 == 0 {
                pair[0] = shell.ribbon_level;
                pair[1] = shell.depth_limit;
            } else {
                pair[2] = shell.ribbon_level;
                pair[3] = shell.depth_limit;
            }
        }
        (pairs, self.shells.len().min(MAX_CARTESIAN_SHELLS) as u32)
    }
}

fn ancestor_path(mut path: Path, depth: u8) -> Path {
    path.truncate(depth);
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated(slot: u8, depth: u8) -> Path {
        let mut path = Path::root();
        for _ in 0..depth {
            path.push(slot);
        }
        path
    }

    #[test]
    fn shell_stack_uses_spaced_local_shells_with_minimum_budgets() {
        let stack = CartesianShellStack::build(repeated(13, 30), 2);
        let (pairs, count) = stack.shader_pairs();
        assert_eq!(count, 3);
        assert_eq!(stack.deepest_path().depth(), 30);
        assert_eq!(stack.preserve_regions()[0], (repeated(13, 30), 10));
        assert_eq!(stack.preserve_regions()[1], (repeated(13, 27), 5));
        assert_eq!(stack.preserve_regions()[2], (repeated(13, 24), 8));
        assert_eq!(pairs[0], [0, 10, 3, 5]);
        assert_eq!(pairs[1][0], 6);
        assert_eq!(pairs[1][1], 8);
    }

    #[test]
    fn shell_stack_respects_required_visible_depth_when_it_exceeds_minimums() {
        let stack = CartesianShellStack::build(repeated(13, 30), 12);
        let preserve = stack.preserve_regions();
        assert_eq!(preserve[0], (repeated(13, 30), 12));
        assert_eq!(preserve[1], (repeated(13, 27), 15));
        assert_eq!(preserve[2], (repeated(13, 24), 18));
    }
}

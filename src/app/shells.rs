use crate::world::anchor::Path;
use crate::world::tree::MAX_DEPTH;

pub const MAX_CARTESIAN_SHELLS: usize = 8;
const DEFAULT_SHELL_MIN_BUDGETS: [u8; 3] = [10, 4, 3];

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
        let _ = base_visual_depth;
        let deepest_depth = deepest_path.depth() as u32;
        let shell_depths = [
            deepest_path.depth(),
            deepest_path.depth().saturating_sub(3),
            0,
        ];
        let mut shells: Vec<CartesianShell> = Vec::with_capacity(shell_depths.len());
        for (shell_depth, min_budget) in shell_depths
            .iter()
            .copied()
            .zip(DEFAULT_SHELL_MIN_BUDGETS.iter().copied())
        {
            let ribbon_level = deepest_depth.saturating_sub(shell_depth as u32);
            let depth_limit = (min_budget as u32).min(MAX_DEPTH as u32);
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
    fn shell_stack_uses_three_local_shells() {
        let stack = CartesianShellStack::build(repeated(13, 30), 2);
        let (pairs, count) = stack.shader_pairs();
        assert_eq!(count, 3);
        assert_eq!(stack.deepest_path().depth(), 30);
        assert_eq!(stack.preserve_regions()[0].0.depth(), 30);
        assert_eq!(stack.preserve_regions()[1].0.depth(), 27);
        assert_eq!(stack.preserve_regions()[2].0.depth(), 0);
        assert_eq!(pairs[0], [0, 10, 3, 4]);
        assert_eq!(pairs[1][0], 30);
        assert_eq!(pairs[1][1], 3);
    }
}

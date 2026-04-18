import { useCrosshair } from "../hooks/useGameState";
import "./Crosshair.css";

/**
 * Center-screen crosshair reticle.
 *
 * The `visible` and `onTarget` bits come from Rust (pushed via
 * `GameStateUpdate::Crosshair` after each `update_highlight` tick).
 * We translate them into CSS classes; the stroke geometry is entirely
 * CSS. That keeps this component essentially a dumb rendered state
 * flag — all the styling decisions live in `Crosshair.css`.
 *
 * Why DOM instead of canvas / WebGL overlay:
 * - Native (physical) resolution regardless of the 3D render scale.
 * - No participation in TAAU history, tonemapping, or any future
 *   post effects — the crosshair stays pixel-crisp and color-stable.
 * - CSS custom properties make the on-target color flip trivial to
 *   swap or extend (e.g., distinct colors for mineable vs. NPC).
 */
export function Crosshair() {
  const { visible, onTarget } = useCrosshair();
  const classes = ["crosshair"];
  if (visible) classes.push("visible");
  if (onTarget) classes.push("on-target");
  return <div className={classes.join(" ")} aria-hidden="true" />;
}

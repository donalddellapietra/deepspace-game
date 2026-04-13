import { useModeIndicator } from "../hooks/useGameState";
import "./ModeIndicator.css";

export function ModeIndicator() {
  const { layer, saveMode, saveEligible, entityEditMode } = useModeIndicator();

  const mode = entityEditMode
    ? "Entity Edit"
    : saveMode
      ? saveEligible
        ? "Save"
        : "Save \u2014 zoom out (Q)"
      : "Terrain";

  const modeClass = entityEditMode
    ? "entity-edit"
    : saveMode
      ? saveEligible
        ? "save"
        : "save-warning"
      : "terrain";

  return (
    <div className="mode-indicator">
      <span className="mode-layer">Layer {layer}</span>
      <span className="mode-divider">/</span>
      <span className={`mode-current ${modeClass}`}>{mode}</span>
    </div>
  );
}

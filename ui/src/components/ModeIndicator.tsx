import { useModeIndicator } from "../hooks/useGameState";
import "./ModeIndicator.css";

export function ModeIndicator() {
  const { layer, saveMode, saveEligible, entityEditMode } = useModeIndicator();

  return (
    <div className="mode-indicator">
      <span className="mode-layer">Layer {layer}</span>
      {saveMode && (
        <span
          className={`mode-badge ${saveEligible ? "save" : "save-warning"}`}
        >
          {saveEligible ? "SAVE" : "SAVE \u2014 zoom out (Q)"}
        </span>
      )}
      {entityEditMode && (
        <span className="mode-badge entity-edit">ENTITY EDIT (G)</span>
      )}
    </div>
  );
}

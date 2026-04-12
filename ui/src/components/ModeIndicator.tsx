import { useModeIndicator } from "../hooks/useGameState";
import "./ModeIndicator.css";

export function ModeIndicator() {
  const { layer, saveMode, saveEligible } = useModeIndicator();

  return (
    <div className="mode-indicator">
      <div className="mode-layer">Layer {layer}</div>
      {saveMode && (
        <div
          className={`mode-save ${saveEligible ? "eligible" : "warning"}`}
        >
          {saveEligible ? "SAVE MODE" : "SAVE \u2014 zoom out (Q)"}
        </div>
      )}
    </div>
  );
}

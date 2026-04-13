import { usePauseMenu } from "../hooks/useGameState";
import { saveGame, loadGame, closePauseMenu, setUiFocused } from "../hooks/useIpc";
import "./PauseMenu.css";

export function PauseMenu() {
  const { open, saveStatus } = usePauseMenu();

  if (!open) return null;

  return (
    <div
      className="pause-overlay"
      onMouseEnter={() => setUiFocused(true)}
      onMouseLeave={() => setUiFocused(false)}
    >
      <div className="pause-panel">
        <h2 className="pause-title">PAUSED</h2>

        <div className="pause-buttons">
          <button className="pause-btn" onClick={() => closePauseMenu()}>
            Resume
          </button>
          <button className="pause-btn" onClick={() => saveGame()}>
            Save Game
          </button>
          <button className="pause-btn" onClick={() => loadGame()}>
            Load Game
          </button>
        </div>

        {saveStatus && (
          <div className="pause-status">{saveStatus}</div>
        )}

        <div className="pause-footer">
          ESC: resume
        </div>
      </div>
    </div>
  );
}

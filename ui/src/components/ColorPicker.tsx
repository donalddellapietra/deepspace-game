import { useColorPicker } from "../hooks/useGameState";
import {
  setColorPickerRgb,
  createBlock,
  setUiFocused,
} from "../hooks/useIpc";
import "./ColorPicker.css";

export function ColorPicker() {
  const { open, r, g, b } = useColorPicker();

  if (!open) return null;

  const preview = `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
  const hex = `#${Math.round(r * 255)
    .toString(16)
    .padStart(2, "0")}${Math.round(g * 255)
    .toString(16)
    .padStart(2, "0")}${Math.round(b * 255)
    .toString(16)
    .padStart(2, "0")}`.toUpperCase();

  return (
    <div
      className="color-picker-panel"
      onMouseEnter={() => setUiFocused(true)}
      onMouseLeave={() => setUiFocused(false)}
    >
      <h2 className="cp-title">CREATE BLOCK</h2>
      <p className="cp-subtitle">Drag sliders to pick a color</p>

      <div className="cp-preview" style={{ backgroundColor: preview }} />
      <span className="cp-hex">{hex}</span>

      <div className="cp-sliders">
        <div className="cp-slider-row">
          <span className="cp-label cp-label-r">R</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.004"
            value={r}
            onChange={(e) =>
              setColorPickerRgb(parseFloat(e.target.value), g, b)
            }
          />
          <span className="cp-value">{Math.round(r * 255)}</span>
        </div>
        <div className="cp-slider-row">
          <span className="cp-label cp-label-g">G</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.004"
            value={g}
            onChange={(e) =>
              setColorPickerRgb(r, parseFloat(e.target.value), b)
            }
          />
          <span className="cp-value">{Math.round(g * 255)}</span>
        </div>
        <div className="cp-slider-row">
          <span className="cp-label cp-label-b">B</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.004"
            value={b}
            onChange={(e) =>
              setColorPickerRgb(r, g, parseFloat(e.target.value))
            }
          />
          <span className="cp-value">{Math.round(b * 255)}</span>
        </div>
      </div>

      <button className="cp-create-btn" onClick={createBlock}>
        Create Block
      </button>

      <span className="cp-hint">C: close</span>
    </div>
  );
}

import { useHotbar } from "../hooks/useGameState";
import { setUiFocused } from "../hooks/useIpc";
import "./Hotbar.css";

export function Hotbar() {
  const { active, slots } = useHotbar();

  const activeSlot = slots[active];
  const activeName = activeSlot?.name ?? `Slot ${active + 1}`;

  return (
    <div
      className="hotbar"
      onMouseEnter={() => setUiFocused(true)}
      onMouseLeave={() => setUiFocused(false)}
    >
      <div className="hotbar-label">{activeName}</div>

      <div className="hotbar-tray">
        {Array.from({ length: 10 }, (_, i) => {
          const slot = slots[i];
          const isActive = i === active;
          const keyLabel = i === 9 ? "0" : `${i + 1}`;

          const bgColor = slot
            ? `rgba(${slot.color[0] * 255}, ${slot.color[1] * 255}, ${slot.color[2] * 255}, ${slot.color[3]})`
            : "rgba(77, 77, 77, 1)";

          return (
            <div key={i} className="hotbar-slot-col">
              <span className={`hotbar-key ${isActive ? "active" : ""}`}>
                {keyLabel}
              </span>
              <div
                className={`hotbar-swatch ${isActive ? "active" : ""}`}
                style={{ backgroundColor: bgColor }}
              />
            </div>
          );
        })}
      </div>

      <div className="hotbar-hint">
        1-0: select &nbsp;|&nbsp; E: inventory &nbsp;|&nbsp; C: color picker
        &nbsp;|&nbsp; Q/F: zoom &nbsp;|&nbsp; V: save
      </div>
    </div>
  );
}

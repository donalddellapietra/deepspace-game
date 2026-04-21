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

          const alpha = slot ? slot.color[3] : 1;
          const translucent = !!slot && alpha < 0.98;
          const bgColor = slot
            ? `rgba(${slot.color[0] * 255}, ${slot.color[1] * 255}, ${slot.color[2] * 255}, ${alpha})`
            : "rgba(77, 77, 77, 1)";

          return (
            <div key={i} className="hotbar-slot-col">
              <span className={`hotbar-key ${isActive ? "active" : ""}`}>
                {keyLabel}
              </span>
              <div
                className={`hotbar-swatch ${isActive ? "active" : ""} ${
                  translucent ? "translucent" : ""
                }`}
              >
                <div
                  className="hotbar-swatch-fill"
                  style={{ backgroundColor: bgColor }}
                />
                {translucent && (
                  <span className="hotbar-alpha-badge">
                    α{alpha.toFixed(2).replace(/^0/, "")}
                  </span>
                )}
              </div>
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

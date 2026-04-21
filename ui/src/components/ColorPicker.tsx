import { useCallback, useEffect, useRef, useState } from "react";
import { useColorPicker } from "../hooks/useGameState";
import { setColorPickerRgb, createBlock, setUiFocused } from "../hooks/useIpc";
import "./ColorPicker.css";

// ── Color conversion helpers ──────────────────────────────────────

function hsvToRgb(
  h: number,
  s: number,
  v: number
): [number, number, number] {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let rr = 0,
    gg = 0,
    bb = 0;
  if (h < 60) {
    rr = c;
    gg = x;
  } else if (h < 120) {
    rr = x;
    gg = c;
  } else if (h < 180) {
    gg = c;
    bb = x;
  } else if (h < 240) {
    gg = x;
    bb = c;
  } else if (h < 300) {
    rr = x;
    bb = c;
  } else {
    rr = c;
    bb = x;
  }
  return [rr + m, gg + m, bb + m];
}

function rgbToHsv(
  r: number,
  g: number,
  b: number
): [number, number, number] {
  const max = Math.max(r, g, b),
    min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = 60 * (((g - b) / d) % 6);
    else if (max === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
  }
  if (h < 0) h += 360;
  const s = max === 0 ? 0 : d / max;
  return [h, s, max];
}

function toHex(r: number, g: number, b: number): string {
  return (
    "#" +
    [r, g, b]
      .map((c) =>
        Math.round(c * 255)
          .toString(16)
          .padStart(2, "0")
      )
      .join("")
      .toUpperCase()
  );
}

function hexToRgb(hex: string): [number, number, number] | null {
  const m = hex
    .replace("#", "")
    .match(/^([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  if (!m) return null;
  return [
    parseInt(m[1], 16) / 255,
    parseInt(m[2], 16) / 255,
    parseInt(m[3], 16) / 255,
  ];
}

// ── Component ─────────────────────────────────────────────────────

export function ColorPicker() {
  const { open, r, g, b, a } = useColorPicker();

  const [hsv, setHsv] = useState<[number, number, number]>(() =>
    rgbToHsv(r, g, b)
  );
  const [hexInput, setHexInput] = useState(() => toHex(r, g, b));

  const svRef = useRef<HTMLDivElement>(null);
  const hueRef = useRef<HTMLDivElement>(null);
  const svDrag = useRef(false);
  const hueDrag = useRef(false);

  // Sync HSV/hex when Rust pushes new RGB values
  useEffect(() => {
    setHsv(rgbToHsv(r, g, b));
    setHexInput(toHex(r, g, b));
  }, [r, g, b]);

  const pushRgba = useCallback(
    (nr: number, ng: number, nb: number, na: number) => {
      setColorPickerRgb(nr, ng, nb, na);
      setHexInput(toHex(nr, ng, nb));
    },
    []
  );

  const updateFromHsv = useCallback(
    (h: number, s: number, v: number) => {
      setHsv([h, s, v]);
      const [nr, ng, nb] = hsvToRgb(h, s, v);
      pushRgba(nr, ng, nb, a);
    },
    [pushRgba, a]
  );

  // ── SV area drag ──
  const handleSv = useCallback(
    (e: MouseEvent | React.MouseEvent) => {
      const rect = svRef.current?.getBoundingClientRect();
      if (!rect) return;
      const s = Math.max(
        0,
        Math.min(1, (e.clientX - rect.left) / rect.width)
      );
      const v = Math.max(
        0,
        Math.min(1, 1 - (e.clientY - rect.top) / rect.height)
      );
      updateFromHsv(hsv[0], s, v);
    },
    [hsv, updateFromHsv]
  );

  // ── Hue bar drag ──
  const handleHue = useCallback(
    (e: MouseEvent | React.MouseEvent) => {
      const rect = hueRef.current?.getBoundingClientRect();
      if (!rect) return;
      const h = Math.max(
        0,
        Math.min(359.9, ((e.clientX - rect.left) / rect.width) * 360)
      );
      updateFromHsv(h, hsv[1], hsv[2]);
    },
    [hsv, updateFromHsv]
  );

  // Global mouse handlers for drag
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (svDrag.current) handleSv(e);
      if (hueDrag.current) handleHue(e);
    };
    const onUp = () => {
      svDrag.current = false;
      hueDrag.current = false;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [handleSv, handleHue]);

  if (!open) return null;

  const hex = toHex(r, g, b);
  const hueColor = `hsl(${hsv[0]}, 100%, 50%)`;
  // rgba(...) preview over a CSS checker background so the user can
  // see how translucent the chosen colour really is.
  const previewColor = `rgba(${Math.round(r * 255)}, ${Math.round(
    g * 255
  )}, ${Math.round(b * 255)}, ${a})`;

  return (
    <div
      className="color-picker-panel"
      onMouseEnter={() => setUiFocused(true)}
      onMouseLeave={() => setUiFocused(false)}
    >
      <h2 className="cp-title">CREATE BLOCK</h2>

      {/* Preview swatch over checker so the alpha channel is visible */}
      <div className="cp-preview-area">
        <div className="cp-preview cp-checker">
          <div
            className="cp-preview-fill"
            style={{ backgroundColor: previewColor }}
          />
        </div>
        <input
          className="cp-hex-input"
          value={hexInput}
          onChange={(e) => {
            setHexInput(e.target.value);
            const parsed = hexToRgb(e.target.value);
            if (parsed) pushRgba(parsed[0], parsed[1], parsed[2], a);
          }}
          onKeyDown={(e) => e.stopPropagation()}
        />
      </div>

      {/* Saturation-Value area */}
      <div
        ref={svRef}
        className="cp-sv-area"
        style={{
          background: `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, ${hueColor})`,
        }}
        onMouseDown={(e) => {
          svDrag.current = true;
          handleSv(e);
        }}
      >
        <div
          className="cp-sv-thumb"
          style={{
            left: `${hsv[1] * 100}%`,
            top: `${(1 - hsv[2]) * 100}%`,
          }}
        />
      </div>

      {/* Hue bar */}
      <div
        ref={hueRef}
        className="cp-hue-bar"
        onMouseDown={(e) => {
          hueDrag.current = true;
          handleHue(e);
        }}
      >
        <div
          className="cp-hue-thumb"
          style={{ left: `${(hsv[0] / 360) * 100}%` }}
        />
      </div>

      {/* RGB + A sliders */}
      <div className="cp-sliders">
        <div className="cp-slider-row">
          <span className="cp-label cp-label-r">R</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.004"
            value={r}
            onChange={(e) => {
              const nr = parseFloat(e.target.value);
              setColorPickerRgb(nr, g, b, a);
            }}
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
            onChange={(e) => {
              const ng = parseFloat(e.target.value);
              setColorPickerRgb(r, ng, b, a);
            }}
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
            onChange={(e) => {
              const nb = parseFloat(e.target.value);
              setColorPickerRgb(r, g, nb, a);
            }}
          />
          <span className="cp-value">{Math.round(b * 255)}</span>
        </div>
        <div className="cp-slider-row">
          <span className="cp-label cp-label-a">A</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={a}
            onChange={(e) => {
              const na = parseFloat(e.target.value);
              setColorPickerRgb(r, g, b, na);
            }}
          />
          <span className="cp-value">{a.toFixed(2)}</span>
        </div>
      </div>

      <button className="cp-create-btn" onClick={createBlock}>
        Create Block
      </button>

      <span className="cp-hint">C: close</span>
    </div>
  );
}

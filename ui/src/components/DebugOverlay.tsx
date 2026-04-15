import { useDebugOverlay } from "../hooks/useGameState";
import "./DebugOverlay.css";

function pad(label: string, value: string, width = 30): string {
  const gap = Math.max(1, width - label.length - value.length);
  return label + " ".repeat(gap) + value;
}

export function DebugOverlay() {
  const s = useDebugOverlay();
  if (!s.visible) return null;

  const lines = [
    "DEBUG [  ]",
    pad("fps", s.fps.toFixed(1)),
    pad("frame time", s.frameTimeMs.toFixed(2) + " ms"),
    pad("zoom level", String(s.zoomLevel)),
    pad("tree depth", String(s.treeDepth)),
    pad("edit depth", String(s.editDepth)),
    pad("visual depth", String(s.visualDepth)),
    pad("anchor depth", String(s.cameraAnchorDepth)),
    pad("camera lx", s.cameraLocal[0].toFixed(4)),
    pad("camera ly", s.cameraLocal[1].toFixed(4)),
    pad("camera lz", s.cameraLocal[2].toFixed(4)),
    pad("fov", s.fov.toFixed(3)),
    pad("nodes", String(s.nodeCount)),
  ];

  return <div className="debug-overlay">{lines.join("\n")}</div>;
}

import { useSyncExternalStore } from "react";
import type {
  GameStateUpdate,
  HotbarState,
  InventoryState,
  ColorPickerState,
  ModeIndicatorState,
  ToastMessage,
  PauseMenuState,
  DebugOverlayState,
  CrosshairState,
} from "../types";
import { getTransport } from "./useTransport";

// ── In-memory stores ─���────────────────────────────────────────────

type Listener = () => void;

function createStore<T>(initial: T) {
  let value = initial;
  const listeners = new Set<Listener>();

  return {
    get: () => value,
    set: (next: T) => {
      value = next;
      listeners.forEach((l) => l());
    },
    subscribe: (l: Listener) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
  };
}

const hotbarStore = createStore<HotbarState>({
  active: 0,
  slots: [],
  layer: 2,
});

const inventoryStore = createStore<InventoryState>({
  open: false,
  builtinBlocks: [],
  customBlocks: [],
  savedMeshes: [],
  layer: 2,
});

const colorPickerStore = createStore<ColorPickerState>({
  open: false,
  r: 0.5,
  g: 0.5,
  b: 0.5,
});

const modeIndicatorStore = createStore<ModeIndicatorState>({
  layer: 2,
  saveMode: false,
  saveEligible: false,
  entityEditMode: false,
});

const toastStore = createStore<ToastMessage | null>(null);

const pauseMenuStore = createStore<PauseMenuState>({
  open: false,
  saveStatus: null,
});

const debugOverlayStore = createStore<DebugOverlayState>({
  visible: false,
  fps: 0,
  frameTimeMs: 0,
  zoomLevel: 0,
  treeDepth: 0,
  editDepth: 0,
  visualDepth: 0,
  cameraAnchorDepth: 0,
  cameraLocal: [0, 0, 0],
  fov: 0,
  nodeCount: 0,
  cameraRootXyz: [0, 0, 0],
  anchorCellSizeRoot: 0,
  anchorSlotsCsv: "",
  activeFrameKind: "",
  renderPathCsv: "",
  tbOnAnchorPath: false,
  anchorCumulativeYawDeg: 0,
  copySeq: 0,
});

// Default `visible: true` so the crosshair is present from the very
// first frame — Rust pushes `visible: false` only when the cursor
// unlocks. If we defaulted to false, there'd be a one-frame flash
// of no-crosshair on startup before the first push arrives.
const crosshairStore = createStore<CrosshairState>({
  onTarget: false,
  visible: true,
});

// ── Dispatch from Rust ────────────────────────────────────────────

function handleGameState(update: GameStateUpdate) {
  switch (update.type) {
    case "hotbar":
      hotbarStore.set(update.data);
      break;
    case "inventory":
      inventoryStore.set(update.data);
      break;
    case "colorPicker":
      colorPickerStore.set(update.data);
      break;
    case "modeIndicator":
      modeIndicatorStore.set(update.data);
      break;
    case "toast":
      toastStore.set(update.data);
      break;
    case "pauseMenu":
      pauseMenuStore.set(update.data);
      break;
    case "debugOverlay":
      debugOverlayStore.set(update.data);
      break;
    case "crosshair":
      crosshairStore.set(update.data);
      break;
  }
}

// Register global handler for WASM mode (Rust calls window.__onGameState).
// In WebSocket mode, the transport calls handleGameState directly.
(window as any).__onGameState = (data: GameStateUpdate | string) => {
  const parsed = typeof data === "string" ? JSON.parse(data) : data;
  handleGameState(parsed);
};

// Wire up the transport — in WS mode this starts the WebSocket connection
// and routes incoming messages through handleGameState.
getTransport().onState(handleGameState);

// ── React hooks ────────���─────────────────────────────���────────────

export function useHotbar(): HotbarState {
  return useSyncExternalStore(hotbarStore.subscribe, hotbarStore.get);
}

export function useInventory(): InventoryState {
  return useSyncExternalStore(inventoryStore.subscribe, inventoryStore.get);
}

export function useColorPicker(): ColorPickerState {
  return useSyncExternalStore(
    colorPickerStore.subscribe,
    colorPickerStore.get
  );
}

export function useModeIndicator(): ModeIndicatorState {
  return useSyncExternalStore(
    modeIndicatorStore.subscribe,
    modeIndicatorStore.get
  );
}

export function useToast(): ToastMessage | null {
  return useSyncExternalStore(toastStore.subscribe, toastStore.get);
}

export function usePauseMenu(): PauseMenuState {
  return useSyncExternalStore(pauseMenuStore.subscribe, pauseMenuStore.get);
}

export function useDebugOverlay(): DebugOverlayState {
  return useSyncExternalStore(debugOverlayStore.subscribe, debugOverlayStore.get);
}

export function useCrosshair(): CrosshairState {
  return useSyncExternalStore(crosshairStore.subscribe, crosshairStore.get);
}

// For clearing toast after display
export function clearToast() {
  toastStore.set(null);
}

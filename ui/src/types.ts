// Types mirroring Bevy resources — shared between React UI and Rust bridge

export interface SlotInfo {
  kind: "block" | "model";
  /** 1-based palette voxel index (for blocks) or mesh index (for models) */
  index: number;
  name: string;
  color: [number, number, number, number]; // r,g,b,a in 0..1
}

export interface HotbarState {
  active: number; // 0-9
  slots: SlotInfo[];
  layer: number;
}

export interface BlockInfo {
  voxel: number; // 1-based palette index
  name: string;
  color: [number, number, number, number];
}

export interface MeshInfo {
  index: number;
  layer: number;
}

export interface InventoryState {
  open: boolean;
  builtinBlocks: BlockInfo[];
  customBlocks: BlockInfo[];
  savedMeshes: MeshInfo[];
  layer: number;
}

export interface ColorPickerState {
  open: boolean;
  r: number;
  g: number;
  b: number;
}

export interface ModeIndicatorState {
  layer: number;
  saveMode: boolean;
  saveEligible: boolean;
}

export interface ToastMessage {
  text: string;
  id: number;
}

// Union of all state updates pushed from Rust
export type GameStateUpdate =
  | { type: "hotbar"; data: HotbarState }
  | { type: "inventory"; data: InventoryState }
  | { type: "colorPicker"; data: ColorPickerState }
  | { type: "modeIndicator"; data: ModeIndicatorState }
  | { type: "toast"; data: ToastMessage };

// Commands sent from React to Rust
export type UiCommand =
  | { cmd: "selectHotbarSlot"; slot: number }
  | { cmd: "assignBlockToSlot"; voxel: number }
  | { cmd: "assignMeshToSlot"; meshIndex: number }
  | { cmd: "setColorPickerRgb"; r: number; g: number; b: number }
  | { cmd: "createBlock" }
  | { cmd: "toggleInventory" }
  | { cmd: "toggleColorPicker" }
  | { cmd: "closeAllPanels" }
  | { cmd: "uiFocused"; focused: boolean };

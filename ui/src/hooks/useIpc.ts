import type { UiCommand } from "../types";

// Command queue that Rust polls via wasm-bindgen
const queue: UiCommand[] = [];

/** Called by Rust to drain pending commands. Returns JSON array. */
(window as any).__pollUiCommands = (): string => {
  const cmds = JSON.stringify(queue);
  queue.length = 0;
  return cmds;
};

/** Send a command to the Rust game. */
export function sendCommand(cmd: UiCommand) {
  queue.push(cmd);
}

// Convenience wrappers

export function selectHotbarSlot(slot: number) {
  sendCommand({ cmd: "selectHotbarSlot", slot });
}

export function assignBlockToSlot(voxel: number) {
  sendCommand({ cmd: "assignBlockToSlot", voxel });
}

export function assignMeshToSlot(meshIndex: number) {
  sendCommand({ cmd: "assignMeshToSlot", meshIndex });
}

export function setColorPickerRgb(r: number, g: number, b: number) {
  sendCommand({ cmd: "setColorPickerRgb", r, g, b });
}

export function createBlock() {
  sendCommand({ cmd: "createBlock" });
}

export function toggleInventory() {
  sendCommand({ cmd: "toggleInventory" });
}

export function toggleColorPicker() {
  sendCommand({ cmd: "toggleColorPicker" });
}

export function setUiFocused(focused: boolean) {
  sendCommand({ cmd: "uiFocused", focused });
}

// Listen for browser exiting Pointer Lock (e.g., user presses Escape).
// The browser consumes the Escape keypress so Bevy never sees it —
// we detect the change here and tell Rust to disengage.
document.addEventListener("pointerlockchange", () => {
  if (!document.pointerLockElement) {
    sendCommand({ cmd: "pointerLockLost" });
  }
});

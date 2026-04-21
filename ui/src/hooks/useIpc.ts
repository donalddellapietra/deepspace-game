import type { UiCommand } from "../types";
import { getTransport } from "./useTransport";

// In WASM mode, Rust polls commands via __pollUiCommands.
// The transport handles this: in WASM mode it maintains the queue and
// sets up the global; in WS mode it sends over the WebSocket.

/** Send a command to the Rust game (works in both WASM and WS modes). */
export function sendCommand(cmd: UiCommand) {
  getTransport().sendCommand(cmd);
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

export function setColorPickerRgb(
  r: number,
  g: number,
  b: number,
  a: number = 1
) {
  sendCommand({ cmd: "setColorPickerRgb", r, g, b, a });
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

export function saveGame() {
  sendCommand({ cmd: "saveGame" });
}

export function loadGame() {
  sendCommand({ cmd: "loadGame" });
}

export function closePauseMenu() {
  sendCommand({ cmd: "closePauseMenu" });
}

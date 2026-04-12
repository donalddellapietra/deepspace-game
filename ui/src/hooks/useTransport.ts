/**
 * Transport abstraction for the Rust <-> React bridge.
 *
 * Three modes (auto-detected):
 * - **WASM mode**: Rust calls `window.__onGameState` via wasm-bindgen.
 *   Commands queued in-memory, polled by Rust via `window.__pollUiCommands`.
 * - **wry IPC mode**: Rust calls `window.__onGameState` via evaluate_script.
 *   Commands sent via `window.ipc.postMessage`.
 * - **WebSocket mode**: Fallback if neither wasm nor wry detected.
 *
 * The transport is a singleton — initialized once and shared across hooks.
 */

import type { GameStateUpdate, UiCommand } from "../types";

// ── Types ────────────────────────────────────────────────────────

type StateHandler = (update: GameStateUpdate) => void;

export interface Transport {
  /** Register the handler that receives game state updates from Rust. */
  onState(handler: StateHandler): void;
  /** Send a command to the Rust game. */
  sendCommand(cmd: UiCommand): void;
}

// ── Detection ────────────────────────────────────────────────────

function hasWryIpc(): boolean {
  return typeof (window as any).ipc?.postMessage === "function";
}

function isWasmMode(): boolean {
  return typeof (window as any).__onGameState === "function";
}

// ── WASM transport ───────────────────────────────────────────────

function createWasmTransport(): Transport {
  const queue: UiCommand[] = [];

  (window as any).__pollUiCommands = (): string => {
    const cmds = JSON.stringify(queue);
    queue.length = 0;
    return cmds;
  };

  return {
    onState(handler: StateHandler) {
      (window as any).__onGameState = (data: GameStateUpdate | string) => {
        const parsed = typeof data === "string" ? JSON.parse(data) : data;
        handler(parsed);
      };
    },
    sendCommand(cmd: UiCommand) {
      queue.push(cmd);
    },
  };
}

// ── wry IPC transport ───────────────────────────────────────────

function createWryTransport(): Transport {
  return {
    onState(handler: StateHandler) {
      // Replace the buffer handler with the real one
      (window as any).__onGameState = (data: GameStateUpdate | string) => {
        const parsed = typeof data === "string" ? JSON.parse(data) : data;
        handler(parsed);
      };
      // Replay any state updates that arrived before React mounted
      const buf = (window as any).__stateBuffer as GameStateUpdate[] | undefined;
      if (buf && buf.length > 0) {
        console.log(`[transport] Replaying ${buf.length} buffered state updates`);
        for (const update of buf) {
          handler(update);
        }
        buf.length = 0;
      }
    },
    sendCommand(cmd: UiCommand) {
      (window as any).ipc.postMessage(JSON.stringify([cmd]));
    },
  };
}

// ── WebSocket transport (fallback) ──────────────────────────────

const WS_URL = "ws://localhost:9000";
const RECONNECT_INTERVAL_MS = 2000;

function createWsTransport(): Transport {
  let ws: WebSocket | null = null;
  let handler: StateHandler | null = null;
  let pendingQueue: UiCommand[] = [];

  function connect() {
    try {
      ws = new WebSocket(WS_URL);
    } catch {
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      console.log("[transport] WebSocket connected to", WS_URL);
      if (pendingQueue.length > 0) {
        for (const cmd of pendingQueue) {
          ws!.send(JSON.stringify([cmd]));
        }
        pendingQueue = [];
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      if (handler && typeof event.data === "string") {
        try {
          const parsed: GameStateUpdate = JSON.parse(event.data);
          handler(parsed);
        } catch (e) {
          console.warn("[transport] Failed to parse state update:", e);
        }
      }
    };

    ws.onclose = () => {
      console.log("[transport] WebSocket disconnected, reconnecting...");
      ws = null;
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  function scheduleReconnect() {
    setTimeout(connect, RECONNECT_INTERVAL_MS);
  }

  connect();

  return {
    onState(h: StateHandler) {
      handler = h;
    },
    sendCommand(cmd: UiCommand) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify([cmd]));
      } else {
        pendingQueue.push(cmd);
      }
    },
  };
}

// ── Singleton ────────────────────────────────────────────────────

let _transport: Transport | null = null;

export function getTransport(): Transport {
  if (!_transport) {
    if (hasWryIpc()) {
      console.log("[transport] Using wry IPC");
      _transport = createWryTransport();
    } else if (isWasmMode()) {
      console.log("[transport] Using WASM bridge");
      _transport = createWasmTransport();
    } else {
      console.log("[transport] Using WebSocket fallback");
      _transport = createWsTransport();
    }
  }
  return _transport;
}

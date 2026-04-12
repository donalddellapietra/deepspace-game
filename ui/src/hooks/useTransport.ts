/**
 * Transport abstraction for the Rust <-> React bridge.
 *
 * Two modes:
 * - **WASM mode**: `window.__onGameState` exists (set by useGameState.ts).
 *   State arrives via that global callback; commands are queued in-memory
 *   and polled by Rust via `window.__pollUiCommands`.
 * - **WebSocket mode**: No globals exist, so we connect to `ws://localhost:9000`.
 *   State arrives as WebSocket messages; commands are sent as WebSocket messages.
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

function isWasmMode(): boolean {
  return typeof (window as any).__onGameState === "function";
}

// ── WASM transport ───────────────────────────────────────────────

function createWasmTransport(): Transport {
  // In WASM mode the globals are already set up by useGameState.ts and
  // useIpc.ts. We just wire into them.
  const queue: UiCommand[] = [];

  // Override the poll function to drain our queue.
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

// ── WebSocket transport ──────────────────────────────────────────

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
      // Flush any commands queued while disconnected.
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
      // onclose will fire after this — it handles reconnect.
      ws?.close();
    };
  }

  function scheduleReconnect() {
    setTimeout(connect, RECONNECT_INTERVAL_MS);
  }

  // Start connecting immediately.
  connect();

  return {
    onState(h: StateHandler) {
      handler = h;
    },
    sendCommand(cmd: UiCommand) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        // Send as a JSON array (same format Rust expects from __pollUiCommands).
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
    _transport = isWasmMode() ? createWasmTransport() : createWsTransport();
  }
  return _transport;
}

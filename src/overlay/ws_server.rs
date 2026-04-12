//! Native WebSocket server for the overlay bridge.
//!
//! Runs a TCP listener on a background thread, accepts multiple WebSocket
//! clients, broadcasts game-state JSON to all of them, and forwards any
//! command JSON they send back to the Bevy main thread via channels.
//!
//! Only compiled on non-wasm32 targets.

use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;

use std::time::Duration;

use tungstenite::protocol::Message;
use tungstenite::{accept, WebSocket};

/// Messages from the Bevy thread to the WS server thread.
pub enum ToServer {
    /// A serialized GameStateUpdate JSON string to broadcast.
    Broadcast(String),
    /// Shut down the server.
    Shutdown,
}

/// Handle held by the Bevy thread to communicate with the WS server.
pub struct WsServerHandle {
    /// Send state updates / shutdown signals to the server thread.
    pub tx: Sender<ToServer>,
    /// Receive command JSON strings from connected clients.
    pub rx: Receiver<String>,
}

impl Drop for WsServerHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(ToServer::Shutdown);
    }
}

/// Start the WebSocket server on `port`. Returns a handle for the main thread.
///
/// The server thread is detached — it will exit when it receives `Shutdown`
/// or when the sender side of `cmd_tx` is dropped.
pub fn start(port: u16, open_browser: bool) -> WsServerHandle {
    let (state_tx, state_rx) = std::sync::mpsc::channel::<ToServer>();
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<String>();

    thread::Builder::new()
        .name("ws-overlay-server".into())
        .spawn(move || {
            if let Err(e) = run_server(port, state_rx, cmd_tx) {
                eprintln!("overlay ws_server: {e}");
            }
        })
        .expect("failed to spawn WebSocket server thread");

    if open_browser {
        let _ = open_in_browser("http://localhost:5173");
    }

    WsServerHandle {
        tx: state_tx,
        rx: cmd_rx,
    }
}

fn run_server(
    port: u16,
    state_rx: Receiver<ToServer>,
    cmd_tx: Sender<String>,
) -> io::Result<()> {
    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))?;
    listener.set_nonblocking(true)?;
    println!("overlay: WebSocket server listening on ws://127.0.0.1:{port}");

    let clients: Arc<Mutex<Vec<WebSocket<TcpStream>>>> = Arc::new(Mutex::new(Vec::new()));

    loop {
        // Check for shutdown or state broadcasts.
        loop {
            match state_rx.try_recv() {
                    Ok(ToServer::Broadcast(json)) => {
                        broadcast(&clients, &json);
                    }
                    Ok(ToServer::Shutdown) => {
                        // Close all clients gracefully.
                        let mut guard = clients.lock().unwrap();
                        for ws in guard.iter_mut() {
                            let _ = ws.close(None);
                            let _ = ws.flush();
                        }
                        println!("overlay: WebSocket server shutting down");
                        return Ok(());
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        // Main thread dropped — shut down.
                        return Ok(());
                    }
                }
            }

        // Accept new connections (non-blocking listener, but handshake
        // needs a blocking stream — switch to non-blocking after).
        match listener.accept() {
            Ok((stream, addr)) => {
                println!("overlay: new WebSocket client from {addr}");
                // Accepted stream inherits listener's non-blocking mode.
                // Force blocking for the WebSocket handshake.
                stream.set_nonblocking(false).ok();
                match accept(stream) {
                    Ok(ws) => {
                        ws.get_ref().set_nonblocking(true).ok();
                        clients.lock().unwrap().push(ws);
                    }
                    Err(e) => {
                        eprintln!("overlay: WebSocket handshake failed: {e}");
                    }
                }
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                // No pending connections — that's fine.
            }
            Err(e) => {
                eprintln!("overlay: TCP accept error: {e}");
            }
        }

        // Poll all clients for incoming messages.
        poll_clients(&clients, &cmd_tx);

        // Don't spin — sleep briefly.
        thread::sleep(Duration::from_millis(1));
    }
}

/// Broadcast a JSON string to all connected clients, removing dead ones.
fn broadcast(clients: &Arc<Mutex<Vec<WebSocket<TcpStream>>>>, json: &str) {
    let mut guard = clients.lock().unwrap();
    let msg = Message::Text(json.into());
    guard.retain_mut(|ws| {
        if ws.write(msg.clone()).is_err() {
            return false;
        }
        ws.flush().is_ok()
    });
}

/// Read messages from all clients; forward command JSON to the channel.
fn poll_clients(
    clients: &Arc<Mutex<Vec<WebSocket<TcpStream>>>>,
    cmd_tx: &Sender<String>,
) {
    let mut guard = clients.lock().unwrap();
    guard.retain_mut(|ws| {
        loop {
            match ws.read() {
                Ok(Message::Text(text)) => {
                    // Client sent a command (or array of commands).
                    if cmd_tx.send(text.to_string()).is_err() {
                        return false; // Main thread gone.
                    }
                }
                Ok(Message::Close(_)) => {
                    let _ = ws.close(None);
                    let _ = ws.flush();
                    return false;
                }
                Ok(Message::Ping(data)) => {
                    let _ = ws.write(Message::Pong(data));
                    let _ = ws.flush();
                }
                Ok(_) => {
                    // Binary, Pong, etc. — ignore.
                }
                Err(tungstenite::Error::Io(ref e))
                    if e.kind() == io::ErrorKind::WouldBlock =>
                {
                    break; // No more data right now.
                }
                Err(_) => {
                    // Connection error — drop client.
                    return false;
                }
            }
        }
        true
    });
}

fn open_in_browser(url: &str) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn()?;
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(url).spawn()?;
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/c", "start", url])
            .spawn()?;
    }
    Ok(())
}

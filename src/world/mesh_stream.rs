//! Async mesh streaming: a dedicated I/O thread that reads prebaked
//! mesh entries from `meshes.bin` on demand.
//!
//! The renderer sends `MeshRequest`s via a channel. The I/O thread
//! reads, decompresses, and posts `MeshResponse`s back. The renderer
//! picks up responses each frame and converts them to GPU meshes.
//!
//! The renderer never blocks on I/O. If a mesh isn't ready yet, the
//! parent's coarser mesh stays visible (parent-fallback rendering).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{mpsc, Mutex};
use std::thread;

use super::serial::{MeshIndexMap, PrebakedEntry};
use super::tree::NodeId;

/// A request from the renderer to the I/O thread.
struct MeshRequest {
    node_id: NodeId,
    offset: u64,
    len: u32,
    priority: u32, // lower = higher priority (closer to camera)
}

/// A response from the I/O thread to the renderer.
pub struct MeshResponse {
    pub node_id: NodeId,
    pub entry: PrebakedEntry,
}

/// Handle to the mesh streaming system. Created once at startup,
/// stored as part of `MeshStore`.
pub struct MeshStreamer {
    /// Send requests to the I/O thread.
    request_tx: mpsc::Sender<MeshRequest>,
    /// Receive completed mesh data from the I/O thread.
    /// Wrapped in Mutex for Sync (Bevy Resource requirement).
    response_rx: Mutex<mpsc::Receiver<MeshResponse>>,
    /// The index: NodeId → (offset, len) in meshes.bin.
    index: MeshIndexMap,
    /// Track which nodes have been requested (avoid duplicate requests).
    in_flight: HashMap<NodeId, ()>,
}

impl MeshStreamer {
    /// Start the mesh streaming system. Loads the index from
    /// `meshes.idx` and opens `meshes.bin` on a dedicated I/O thread.
    pub fn start(idx_path: &Path, bin_path: &Path) -> std::io::Result<Self> {
        let index = MeshIndexMap::load(idx_path)?;
        eprintln!("mesh streamer: loaded index ({} entries)", index.len());

        let (request_tx, request_rx) = mpsc::channel::<MeshRequest>();
        let (response_tx, response_rx) = mpsc::channel::<MeshResponse>();

        let bin_path = bin_path.to_path_buf();
        thread::Builder::new()
            .name("mesh-io".into())
            .spawn(move || {
                io_thread(bin_path, request_rx, response_tx);
            })?;

        Ok(Self {
            request_tx,
            response_rx: Mutex::new(response_rx),
            index,
            in_flight: HashMap::new(),
        })
    }

    /// Request a mesh for a node. No-op if already in-flight or if the
    /// node isn't in the index. `priority`: lower = closer to camera.
    pub fn request(&mut self, node_id: NodeId, priority: u32) {
        if self.in_flight.contains_key(&node_id) {
            return;
        }
        let Some((offset, len)) = self.index.get(node_id) else {
            return;
        };
        self.in_flight.insert(node_id, ());
        // If the channel is full (thread busy), the request is dropped.
        // The renderer will re-request next frame.
        let _ = self.request_tx.send(MeshRequest {
            node_id,
            offset,
            len,
            priority,
        });
    }

    /// Check if a node is in the prebaked index.
    pub fn has_node(&self, node_id: NodeId) -> bool {
        self.index.get(node_id).is_some()
    }

    /// Synchronously load a mesh entry. Used for nodes that are
    /// visible RIGHT NOW and must render this frame. Opens a second
    /// file handle so it doesn't conflict with the I/O thread.
    pub fn load_sync(&self, node_id: NodeId) -> Option<PrebakedEntry> {
        let (offset, len) = self.index.get(node_id)?;
        // Open a fresh handle — the I/O thread owns the main one.
        let bin_path = std::path::Path::new("assets/meshes.bin");
        let mut file = std::fs::File::open(bin_path).ok()?;
        super::serial::read_mesh_entry(&mut file, offset, len).ok()
    }

    /// Drain all completed responses from the I/O thread.
    /// Call once per frame.
    pub fn drain_responses(&mut self) -> Vec<MeshResponse> {
        let rx = self.response_rx.get_mut().unwrap();
        let mut responses = Vec::new();
        while let Ok(resp) = rx.try_recv() {
            self.in_flight.remove(&resp.node_id);
            responses.push(resp);
        }
        responses
    }
}

/// The I/O thread's main loop. Reads requests, seeks into meshes.bin,
/// decompresses, and sends responses back.
fn io_thread(
    bin_path: std::path::PathBuf,
    request_rx: mpsc::Receiver<MeshRequest>,
    response_tx: mpsc::Sender<MeshResponse>,
) {
    let mut file = match std::fs::File::open(&bin_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("mesh-io thread: failed to open {}: {}", bin_path.display(), e);
            return;
        }
    };

    // Process requests. Sort pending requests by priority before
    // processing so closer nodes are loaded first.
    let mut batch: Vec<MeshRequest> = Vec::new();

    loop {
        // Block until at least one request arrives.
        match request_rx.recv() {
            Ok(req) => batch.push(req),
            Err(_) => return, // channel closed, renderer dropped
        }
        // Drain any additional pending requests.
        while let Ok(req) = request_rx.try_recv() {
            batch.push(req);
        }

        // Sort by priority (lower = more urgent).
        batch.sort_by_key(|r| r.priority);

        for req in batch.drain(..) {
            match super::serial::read_mesh_entry(&mut file, req.offset, req.len) {
                Ok(entry) => {
                    let _ = response_tx.send(MeshResponse {
                        node_id: req.node_id,
                        entry,
                    });
                }
                Err(e) => {
                    eprintln!(
                        "mesh-io: failed to read node {}: {}",
                        req.node_id, e,
                    );
                }
            }
        }
    }
}

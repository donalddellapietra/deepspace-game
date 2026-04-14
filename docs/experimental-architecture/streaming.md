# Streaming and Multiplayer

## Why Streaming Works Better with Ray Marching

With mesh-based rendering, streaming serves pre-baked triangle meshes. A missing mesh is a hole in the world — the renderer has nothing to show. Meshes are large (10-100KB each), change on edit (re-bake required), and have complex cache invalidation.

With ray marching, streaming serves tree nodes. A missing node is not a hole — the ray stops descending and renders at the parent's coarser resolution. The world is always complete, just at lower detail until data arrives. Nodes are tiny (~220 bytes), immutable (content-addressed), and never need cache invalidation.

## Content Addressing = Perfect Caching

Every node has a NodeId derived from its content (hash of its 27 children). This means:

- **A NodeId never changes.** If two nodes have the same children, they share the same NodeId. If a child changes, the parent gets a new NodeId.
- **No cache invalidation, ever.** A cached node is valid forever. New edits create new NodeIds — they don't modify existing ones.
- **CDN-native.** NodeId is the cache key. Nodes are small, immutable, static assets. Any CDN (CloudFront, Cloudflare, etc.) handles this natively.
- **Dedup at the network level.** Two players in the same forest fetch the same nodes. The CDN serves from cache.

## Data Sizes

| What | Size | Notes |
|------|------|-------|
| One node | ~220 bytes | 27 children + ref_count |
| Path from root to player | 63 nodes = ~14KB | Minimum to locate the player |
| Visible scene (near detail) | ~10,000 nodes = ~2.2MB | Full detail around the player |
| Visible scene (coarse far) | ~1,000 nodes = ~220KB | Low-detail horizon |
| One block edit | 63 new nodes = ~14KB | One new node per ancestor to root |
| 1 million edits (raw) | ~14GB | Before dedup |
| 1 million edits (deduped) | ~1-5GB | Many ancestors are shared or identical |

## Streaming Protocol

### Initial Load

1. Client connects to the game server. Server sends the current root NodeId.
2. Client requests the 63 nodes along the path from root to the player's position. (~14KB, instant.)
3. Client requests neighbor nodes within view radius, prioritized by:
   - Distance from player (near first)
   - Depth (coarse ancestors first, detail later)
4. GPU ray marches whatever is loaded. Missing nodes render at parent resolution.
5. Detail pops in smoothly over the first 1-2 seconds as nodes arrive.

### Steady State (Player Moving)

As the player moves, new nodes enter the view radius:

1. CPU prefetches nodes along the movement direction.
2. Nodes are requested in priority order: the node the player is about to enter, then its children, then its children's children.
3. Cache hits (CDN or local) are instant. Cache misses go to the origin server.
4. Evict nodes that are far behind the player from GPU memory.

### Edits

When the player breaks or places a block:

1. CPU creates 63 new nodes (one per ancestor to root). The old nodes remain valid.
2. The 63 new nodes are sent to the server. (~14KB upload.)
3. Server stores the new nodes, updates the world's root NodeId.
4. Server broadcasts the new root NodeId to other connected clients.
5. Other clients fetch the changed nodes on demand — only the nodes along the path to the edit, and only if they're within view radius.

### Multiplayer Edit Conflicts

Two players edit different blocks simultaneously:

1. Player A breaks block X → creates new ancestors A1..A63, sends to server.
2. Player B breaks block Y → creates new ancestors B1..B63, sends to server.
3. If X and Y are in different subtrees, the ancestors diverge at some layer. The server merges by creating a new root that includes both edits. This is straightforward because content addressing is composable — each edit's ancestors are independent above the divergence point.
4. If X and Y are in the SAME node (both editing the same 27-child block), the server resolves by ordering the edits. One applies first, the other applies on top.

Content addressing makes conflict resolution simple: edits are small (63 nodes), independent above their divergence point, and composable.

## Edge Cache Architecture

```
[Player Client] ←→ [Edge CDN (CloudFront/Cloudflare)] ←→ [Origin Server]
                          ↑
                    NodeId = cache key
                    Node data = cache value
                    TTL = forever (immutable)
```

The CDN layer is stateless. It caches nodes by NodeId. Because nodes are immutable, the TTL is infinite — a cached node is valid forever. The only "invalidation" is that new edits create new NodeIds that aren't in the cache yet. The CDN fetches them from origin on first request, then caches them for all subsequent clients.

### Cache Warming

For a new game world, the CDN is cold. The first player to visit a region fetches nodes from origin and warms the cache. Subsequent players in the same region get instant cache hits.

For popular regions (spawn points, cities, shared builds), the cache is naturally warm. For remote regions (deep space, unexplored areas), the first visitor pays the origin latency once.

### Storage Budget

The origin server stores all nodes ever created. With content-addressed dedup, identical subtrees are stored once. A world with:

- 1,000 unique tree types × ~1,000 nodes each = ~1M nodes = ~220MB
- 100 unique forest arrangements × ~10,000 nodes each = ~1M nodes = ~220MB
- Player edits: 1M edits × 63 nodes/edit × dedup = ~10M unique nodes = ~2.2GB

Total origin storage for a richly authored world with a million player edits: ~3-5GB. Well within a single server.

## What the Client Stores

The client maintains a local node cache (in-memory HashMap<NodeId, Node>):

- **Capacity:** ~100,000 nodes = ~22MB. Enough for the visible scene plus a generous prefetch buffer.
- **Eviction:** LRU by last access time. Nodes far from the player are evicted first.
- **Persistence:** Optionally write the cache to disk (IndexedDB on WASM, filesystem on native) so the next session starts warm.

The client never stores the full world. It stores a sliding window around the player, fetching new nodes and evicting old ones as the player moves.

## Offline / Single-Player Mode

For offline play, the full world (or a region of it) is bundled as a file:

- A flat list of `(NodeId, Node)` pairs, sorted by NodeId.
- The root NodeId.
- Player edits are stored as additional nodes appended to the file (or a separate save file).

The `gen_world` binary produces this file. The client loads it into the local node cache at startup. No network needed.

This is the same file format as the current `world.bin`, just with the new node format (27 children instead of voxel grid + optional children).

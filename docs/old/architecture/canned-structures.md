# Canned Structures and Canned Runs

This is the fundamental content model for the game. Everything in the
world — terrain, biomes, cities, NPCs, simulations — is built from
precomputed library entries called **canned structures**.

## The problem with procedural generation

Procedural noise (Perlin, Simplex, etc.) evaluated at runtime has three
fatal properties in a content-addressed tree:

1. **Destroys dedup.** Every surface node becomes unique. The library
   balloons from ~25 entries (grassland) to thousands.
2. **Kills startup.** Evaluating noise for every surface leaf is O(surface
   area). A 500-radius sphere has ~5000 surface leaves × 15,625 noise
   calls each. Debug builds take 10–55 seconds. Unacceptable.
3. **Bursts at zoom transitions.** Unique nodes all need cold bakes
   (mesh generation). The renderer can't budget hundreds of 2MB
   allocations per frame without pop-in or frame drops.

Procedural generation is the wrong tool. The tree's power is dedup, and
procedural noise defeats dedup.

## Canned structures

A **canned structure** is a precomputed subtree stored in the node
library. At each layer, you generate a fixed set of unique patterns
(say, 10,000) and store them. A planet at layer 6 is composed of 125
children, each selected from the library of layer-7 patterns. Those
layer-7 patterns each have 125 children selected from layer-8 patterns,
and so on down to leaves.

The key insight: **combinatorial variety from bounded storage.** With
10,000 patterns per layer across 6 layers, the number of possible
arrangements is 10,000^6 — more than enough that no player would ever
see repetition. But the storage is only 10,000 × 7 layers of patterns.
Dedup works perfectly because patterns are reused by design.

### How canned structures work with the tree

```
Layer 6:  1 planet node
            └── picks 125 children from library of layer-7 patterns

Layer 7:  ~200 unique continent patterns
            └── each picks 125 children from library of layer-8 patterns

Layer 8:  ~2,000 unique region patterns
            └── each picks 125 children from layer-9 patterns

...

Layer 12: ~10,000 unique leaf patterns
            └── 25³ voxel grids: terrain surfaces, interiors, air
```

Each non-leaf pattern is a `(VoxelGrid, Children)` pair — the same
`Node` structure the library already stores. The downsampled voxels are
precomputed. The children array references other patterns in the library.
Content-addressed dedup means identical patterns share storage
automatically.

### Generation is a build step

Canned structures are generated **offline**, not at runtime:

1. A build tool (`cargo run --bin gen-world`) generates all patterns
   for a given world configuration (planet shape, biome distribution,
   terrain noise parameters).
2. The result is serialized to disk (e.g., `assets/world.bin`).
3. At app startup, the game deserializes the file into a `WorldState`.
   No noise evaluation, no tree building — just load bytes.
4. When terrain parameters change, the developer reruns the build tool.

In production (MMO), the server holds the canonical library. Clients
download only the patterns they need for the layers they're viewing —
exactly like texture streaming but for the entire world state.

### Startup is instant

Loading a serialized `WorldState` is a single read + deserialize.
The cost is paid once by the build tool, never by the player. Debug
vs release build speed becomes irrelevant — the player never runs
noise code.

## Canned runs

A canned structure is not just geometry — it's a **snapshot of a
simulation at a point in time**. A forest biome at T=200 has trees at
specific growth stages, creatures in specific positions, weather in a
specific state. This snapshot is itself a canned structure.

A **canned run** is a sequence of canned structures representing a
simulation's evolution over time:

```
T=0:    young forest, saplings, few animals
T=100:  mature forest, full canopy, wolf pack
T=200:  old growth, deadfall, bear territory
T=300:  lightning fire, charred stumps, regrowth beginning
```

Each timestep is a complete subtree in the library. When a player
enters a biome, the game selects the snapshot matching the current
simulation time. The world feels alive because it IS alive — it was
simulated — just pre-recorded.

### Canned runs ARE canned structures

There is no distinction between a canned structure and a canned run.
A canned structure is the state of a simulation at one point in time.
A canned run is a collection of canned structures indexed by time.
The library stores both identically — they're all just `Node` entries
with `(VoxelGrid, Children)`.

This means the same serialization, the same dedup, the same streaming,
and the same rendering pipeline handle both spatial content and temporal
evolution. No special systems needed.

### Simulation fidelity from precomputation

The simulation that produces canned runs can be arbitrarily expensive
because it runs offline. You can simulate:

- Weather systems evolving over seasons
- Ecosystems with predator-prey dynamics
- Cities growing and decaying
- Wars advancing and retreating
- Geological processes (erosion, volcanic activity)

None of this cost hits the player. The build tool runs the simulation,
snapshots the state at intervals, and stores the snapshots as canned
structures. The game just loads the right snapshot.

## How this relates to the tree architecture

### Layer uniformity

Canned structures respect the layer-uniform principle perfectly. At
every layer, the operation is the same: look up a pattern from the
library, render its 25³ voxel grid. There are no layer-specific hacks,
no noise cutoffs, no special cases. A layer-6 planet pattern and a
layer-12 leaf pattern are both just library entries.

### Editing

When a player edits a voxel, `install_subtree` replaces one leaf and
propagates the downsample upward. The edited node is no longer a
canned pattern — it's now a unique node. This is fine: the library
stores it as a new entry, dedup handles the rest of the subtree, and
the edit persists as a delta against the canned base.

### Scale

The entire system scales by adding more patterns to the library, not
by making the runtime more complex. A richer world means a larger
`world.bin` file, not slower frame times.

## Terrain as canned structures

The immediate application: planet terrain.

Instead of evaluating noise at runtime, the build tool:

1. Defines the planet's density field (sphere SDF + noise octaves)
2. Generates leaf patterns for every surface configuration
3. Builds parent patterns by selecting and arranging child patterns
4. Serializes the complete tree

The player loads the result and sees a fully-detailed planet at every
zoom layer, with zero generation cost at runtime.

The noise parameters, octave counts, and terrain complexity are
decisions made by the build tool — they affect build time, not play
time. You can use 20 octaves of noise with fractal detail if you want.
The player never pays for it.

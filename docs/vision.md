# Vision: Seamless Scale from Galaxy to Street

## The aspiration

A single world where you can zoom from a galaxy map down to a village
street and talk to a specific peasant whose life is shaped by your
empire-level decisions — with no loading screens, no mode switches, and
no separate "strategy game" vs "RPG" executables.

Think EU4's strategic depth at the top, Dwarf Fortress's simulation in
the middle, and a first-person voxel RPG at the bottom — all running
in one tree, all visible through the same camera by scrolling the zoom
wheel.

This has never shipped. Spore promised it but faked it — five separate
games with loading screens between them. No Man's Sky does seamless
planet-to-space transitions but has no systemic depth on the ground.
Dwarf Fortress simulates entire civilizations but locks you to one
fortress. EU4 simulates millions of people as ledger entries you never
visit.

Nobody has built the full stack because it requires the same data
structure, the same rendering pipeline, and the same simulation
framework at every scale. That is exactly what the content-addressed
tree provides.

## Why the architecture supports this

### The tree is already a simulation hierarchy

The 12-layer tree maps naturally onto simulation tiers:

| Layers | Scale             | Simulation tier                         |
|--------|-------------------|-----------------------------------------|
| 1-3    | Galaxy / sector   | Civilizations as aggregate stats        |
| 4-6    | Continent / region| Provinces, armies, trade as flows       |
| 7-9    | City / district   | Buildings, crowds, local economy        |
| 10-12  | Street / room     | Individual NPCs, full AI, animation     |

Each layer isn't just a visual LOD — it's a **simulation LOD**. Systems
at layer 4 tick grand strategy (economy, diplomacy, war). Systems at
layer 8 tick city simulation (trade, crime, construction). Systems at
layer 12 tick individual AI (pathfinding, dialogue, combat). You only
run the expensive tiers near the player.

### Dedup handles the scale

A medieval kingdom with 500 identical peasant huts stores one hut node
referenced 500 times. A continent of grassland with scattered villages
is almost free. A crowd of 10,000 identical idle NPCs at a distance
compresses to one pattern node. The tree doesn't care whether a node
represents terrain or a character — dedup works the same way.

### Zoom transitions are already seamless

The renderer walks the tree to whatever depth matches the camera. A
city at layer 6 is a few colored voxels. At layer 8, you see buildings.
At layer 12, you see bricks. No streaming, no loading — just tree
traversal depth. Characters follow the same rule: zoomed out, they
collapse into the landscape via downsample, like everything else.

### Edits propagate across scale

If a player burns a village at leaf layer, the downsample walks up and
the village looks damaged from orbit. If an empire-level war razes a
province, an edit at a high layer cascades down. The tree handles both
directions with the same `install_subtree` walk.

## The simulation pyramid

"Billions of NPCs" is a layered illusion:

**Fully simulated** (dozens) — near the player. Full AI, skeletal
animation on voxel body parts, pathfinding, physics. Capped at ~50-100
entities.

**Simplified agents** (thousands) — further out. Reduced tick rate,
state-machine AI, no animation. Possibly just voxel figures with basic
movement patterns.

**Statistical simulation** (millions) — no individual entities. A city
node at layer 7 has metadata: `population: 50,000`, `mood: restless`,
`economy: declining`. When the player zooms in, individuals are spawned
from probability distributions seeded by the node's position. Zoom out,
they fold back into the stats.

**Pure data** (billions) — a civilization at layer 3 is numbers on a
ledger. Population, military strength, resources, diplomatic relations.
Grows and shrinks based on formulas. No one has a position or a face.

The key invariant: **transitions between tiers are seamless and
deterministic.** The same seed at the same node always generates the
same individuals. Player-caused changes are stored as deltas on the
node, so zooming back in reproduces the same village with the same
modifications.

## Voxel characters

Characters are voxels, not traditional meshes. At 3 layers below the
current view layer (125^3 ~ 2M voxels), per-voxel animation artifacts
are subpixel — invisible to the player. The approach:

- Load skeleton and animation clips from standard glTF files
- Voxelize the mesh at load time, partitioned by dominant bone
- Use Bevy's AnimationPlayer to compute bone transforms each frame
- Position voxel subtrees to match bone positions (rigid-group animation)
- At high resolution, joint gaps are invisible; at low resolution,
  characters are a single colored voxel — correct either way

The layer system gives character LOD for free:
- Close up: full voxel detail, smooth animation
- Medium distance: 25^3 silhouette, still recognizable
- Far away: single voxel dot — which is correct
- Zoomed out further: characters fold into population statistics

Characters start as Bevy entities with voxel meshes (outside the tree)
for performance. They enter the tree only for interactions that require
it: destruction, voxel-level physics, zoom-coherent rendering.

## What needs to be built (roughly, in order)

1. **Canned structure pipeline.** Offline build tool that generates
   world content (terrain, biomes, structures) as precomputed library
   patterns, serializes them to disk, and loads them at runtime. This
   replaces runtime procedural generation entirely. Player edits are
   stored as deltas against the canned base. See
   `docs/architecture/canned-structures.md`.

2. **Async meshing with priority queues.** Near-player chunks at high
   priority, distant chunks at low priority. Required before the world
   gets large enough to cause frame drops.

3. **Node metadata.** Attach simulation data (population, economy,
   faction) to tree nodes at coarse layers. This is the statistical
   simulation tier.

4. **Canned runs.** Offline simulation tool that evolves canned
   structures over time and snapshots the results. A forest at T=0 is
   saplings; at T=200 it's old growth; at T=300 it's post-fire
   regrowth. Each snapshot is a canned structure in the library. The
   game selects the snapshot matching the current simulation time.

5. **Entity spawning/despawning driven by zoom.** As the player zooms
   into a city, the population stat spawns representative NPCs. Zoom
   out, they fold back. Deterministic seeding ensures coherence.

6. **Voxel character pipeline.** glTF mesh voxelizer, bone-partitioned
   body parts, rigid-group animation driven by Bevy's AnimationPlayer.

7. **Event propagation across layers.** Empire declares war at layer 4;
   cities enter wartime production at layer 8; NPCs reference it in
   dialogue at layer 12. Message-passing that flows down the tree.

8. **Two adjacent layers working together.** The proof-of-concept
   milestone: city management + street-level exploration, with seamless
   zoom and coherent simulation handoff. Everything else expands from
   this.

## Design risk

The technical architecture supports this. The hard problem is design:
each zoom level must be a genuinely good game, not a shallow minigame.
A good strategy layer AND a good city builder AND a good RPG, all
coherent with each other. Spore failed here — the tech worked, but
each stage was thin.

The mitigation is to build one layer at a time, make it deep, then
connect it to its neighbors. The tree composes naturally because every
layer shares the same spatial hierarchy. Two good layers that hand off
cleanly are worth more than twelve shallow ones.

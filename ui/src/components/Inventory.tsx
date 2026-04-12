import { useInventory } from "../hooks/useGameState";
import {
  assignBlockToSlot,
  assignMeshToSlot,
  setUiFocused,
} from "../hooks/useIpc";
import type { BlockInfo, MeshInfo } from "../types";
import "./Inventory.css";

function BlockTile({ block }: { block: BlockInfo }) {
  const bgColor = `rgba(${block.color[0] * 255}, ${block.color[1] * 255}, ${block.color[2] * 255}, ${block.color[3]})`;

  return (
    <button
      className="inv-block-tile"
      onClick={() => assignBlockToSlot(block.voxel)}
    >
      <div className="inv-block-swatch" style={{ backgroundColor: bgColor }} />
      <span className="inv-block-name">{block.name}</span>
    </button>
  );
}

function MeshTile({ mesh }: { mesh: MeshInfo }) {
  return (
    <button
      className="inv-mesh-tile"
      onClick={() => assignMeshToSlot(mesh.index)}
    >
      <span className="inv-mesh-id">#{mesh.index}</span>
      <span className="inv-mesh-layer">L{mesh.layer}</span>
    </button>
  );
}

export function Inventory() {
  const { open, builtinBlocks, customBlocks, savedMeshes, layer } =
    useInventory();

  if (!open) return null;

  return (
    <div
      className="inventory-panel"
      onMouseEnter={() => setUiFocused(true)}
      onMouseLeave={() => setUiFocused(false)}
    >
      <h2 className="inv-title">INVENTORY</h2>
      <p className="inv-subtitle">
        Click a block to assign to active hotbar slot
      </p>

      <div className="inv-scroll">
        {/* Built-in Blocks */}
        <div className="inv-section-header">BUILT-IN BLOCKS</div>
        <div className="inv-grid">
          {builtinBlocks.map((b) => (
            <BlockTile key={b.voxel} block={b} />
          ))}
        </div>

        <div className="inv-divider" />

        {/* Custom Blocks */}
        <div className="inv-section-header">CUSTOM BLOCKS (C TO CREATE)</div>
        <div className="inv-grid">
          {customBlocks.length === 0 ? (
            <span className="inv-hint">Press C to create custom blocks</span>
          ) : (
            customBlocks.map((b) => <BlockTile key={b.voxel} block={b} />)
          )}
        </div>

        <div className="inv-divider" />

        {/* Saved Meshes */}
        <div className="inv-section-header inv-mesh-header">
          Saved Meshes &mdash; Layer {layer} ({savedMeshes.length})
        </div>
        <div className="inv-grid">
          {savedMeshes.map((m) => (
            <MeshTile key={m.index} mesh={m} />
          ))}
        </div>
      </div>

      <div className="inv-divider" />
      <div className="inv-footer">
        E: close &nbsp;|&nbsp; 1-0: select slot &nbsp;|&nbsp; Q/F: zoom
        &nbsp;|&nbsp; V: save mode
      </div>
    </div>
  );
}

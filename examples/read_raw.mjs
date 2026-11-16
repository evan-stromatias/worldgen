// Read a raw terrain bundle with no library: JSON for the manifest, typed arrays for the grids.
//
//     uv run worldgen --seed 7 --format raw --out out/raw
//     node examples/read_raw.mjs out/raw
//
// Every array is row-major [z][x] and little-endian: sample (x, z) is at index z * width + x.

import { readFileSync } from "node:fs";
import { join } from "node:path";

const TYPES = { uint8: Uint8Array, uint16: Uint16Array, float32: Float32Array, float64: Float64Array };

function readArray(dir, entry) {
  const bytes = readFileSync(join(dir, entry.file));
  const Type = TYPES[entry.dtype];
  const expected = entry.shape.reduce((a, b) => a * b, 1) * Type.BYTES_PER_ELEMENT;
  if (bytes.length !== expected) throw new Error(`${entry.file}: ${bytes.length} bytes, expected ${expected}`);
  // Copy into a fresh buffer: a typed array needs its offset aligned to its element size
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length);
  return new Type(buffer); // node runs on little-endian hosts, so no byte swapping
}

const dir = process.argv[2] ?? ".";
const manifest = JSON.parse(readFileSync(join(dir, "terrain.json"), "utf8"));
if (manifest.format !== "worldgen-raw" || manifest.format_version !== 1) {
  throw new Error(`not a raw bundle this reader knows: ${manifest.format} ${manifest.format_version}`);
}
const { world, arrays } = manifest;
const elevation = readArray(dir, arrays.elevation);
const water = readArray(dir, arrays.water);
const biome = readArray(dir, arrays.biome);
const [, vertices] = arrays.elevation.shape;
const [, cells] = arrays.biome.shape;
const meta = JSON.parse(readFileSync(join(dir, manifest.meta), "utf8"));

const [x, z] = [40, 70]; // a vertex, and the cell south-east of it
const surface = water[z * vertices + x];
console.log(`worldgen ${manifest.version}, seed ${manifest.seed}: ${world.size_cells}² cells of ${world.cell_size} m`);
console.log(`elevation at (${x}, ${z}): ${elevation[z * vertices + x].toFixed(3)} m`);
console.log(`water at (${x}, ${z}): ${Number.isNaN(surface) ? "dry" : `${surface.toFixed(3)} m`}`);
console.log(`biome at (${x}, ${z}): ${manifest.biomes[biome[z * cells + x]].id}`);
console.log(`${meta.lakes.length} lakes, ${meta.rivers.length} rivers`);

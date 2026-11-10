# Feature request: a `client` field on `TerrainConfig`

From: the `r` game repo (its `realmgen` package, plan 030), worldgen's first client.

## Problem

Clients embed a terrain recipe inside a richer recipe file. realmgen's recipes add game
sections (`towns`, `roads`, `bridges`, `flora`, `realms`, `names`) beside worldgen's
`world`/`elevation`/`erosion`/`hydrology`/`climate`/`biomes`, and `Section`'s
`extra="forbid"` rejects every one of them — so no existing client recipe can be handed to
`TerrainConfig` at all.

## Request

One top-level field:

```python
class TerrainConfig(Section):
    client: dict[str, Any] = Field(
        default_factory=dict,
        description="Client-owned data; stored and exported verbatim, never read by worldgen.",
    )
```

- worldgen never reads or validates it, beyond "is a mapping". Keeping `forbid` everywhere
  else keeps typo safety for worldgen's own keys — preferred over a blanket `extra="ignore"`.
- `export_terrain` writes it back into `terrain.yaml` verbatim (deterministic dump), so a
  terrain bundle carries the client recipe that made it: a client loading a bundle recovers
  its own config from the bundle alone.

```yaml
world: {...}
biomes: {...}
client:
  version: 3            # the client's own format counter for what follows
  biome_flora: {...}
  towns: {...}
```

## Not requested

No loosening of `BiomeRule`: the client is moving its per-biome `cover`/`grass` out of the
rules into `client.biome_flora`, keyed by biome id.

## Fallback

Without the field, the client parses the YAML itself, pops worldgen's six sections,
validates those as `TerrainConfig` and keeps the rest. That works, but breaks if worldgen
ever grows a colliding section, and cannot round-trip through `export_terrain`.

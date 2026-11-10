"""Offline terrain generation: terrain.yaml + seed -> elevation, water, biomes, lakes and rivers.

The recipe (`config`), the five stages that read it (`stages/`, run in order by `pipeline`), the noise and
grid helpers the stages share, the terrain in memory (`terrain`) and as a bundle on disk (`terrain_io`),
reading a terrain in patches (`patches`) and the hillshaded preview (`preview`). Runs once, offline.
"""

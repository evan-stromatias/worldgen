"""The world rendered into the map picture's texture: the flat layers in 2D and the terrain as one 3D
mesh, so rayscreen sees no difference between them (`map_render`, `terrain_view`). The only modules of
the tool that make GL calls — main thread only.
"""

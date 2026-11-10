"""A terrain generator with a UI: the recipe's sections as forms on the right, the terrain on the left.

Model (`model/`, no raylib), view (`view/`, rayscreen specs; only `map_render` and `terrain_view` make GL calls)
and the controller between them; `app.py` is the window's loop.
"""

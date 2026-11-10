"""A terrain generator with a UI: the recipe's sections as forms on the right, the terrain on the left.

Model (`model/`, no raylib), view (`view/`, rayscreen specs; only `view/gl/` makes GL calls) and the
controller between them (the only code that changes the model); `app.py` is the window's loop.
"""

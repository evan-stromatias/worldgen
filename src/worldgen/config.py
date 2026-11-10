"""Where terrain generation finds its bundled files: the default recipe, kept with the repository's
assets rather than inside the package. And the package's own version, from its installed metadata."""

import importlib.metadata
import sys
from pathlib import Path

DEFAULT_RECIPE = Path(__file__).resolve().parents[2] / "assets" / "recipes" / "default.yaml"

try:
    VERSION = importlib.metadata.version("worldgen")
except importlib.metadata.PackageNotFoundError:  # running from a bare source tree, never installed
    VERSION = "0.0.0"
    print(f"warning: worldgen is not installed as a package, so its version is unknown; writing {VERSION}",
          file=sys.stderr)

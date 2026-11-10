"""The terrain bundle: generate_terrain, export_terrain and load_terrain round-trips."""

import dataclasses

import numpy as np
import pytest
import yaml

from worldgen.config import VERSION
from worldgen.generator import preview
from worldgen.generator.config import ConfigError, load_config
from worldgen.generator.terrain_io import ARRAYS, export_terrain, generate_terrain, load_terrain

ARRAY_NAMES = ("elevation", "water", "sea", "lake_id", "river", "flow_dir", "biome", "color", "temperature",
               "moisture", "sediment")


def test_export_load_round_trip(small_terrain, tmp_path):
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    assert {p.name for p in out.iterdir()} == {"terrain.yaml", "terrain.npz", "preview.png"}

    loaded = load_terrain(out)
    assert loaded.seed == small_terrain.seed and loaded.config == small_terrain.config
    for name in ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(small_terrain, name))
    assert loaded.lakes == small_terrain.lakes and loaded.rivers == small_terrain.rivers


def test_derived_arrays_are_recomputed_exactly(small_terrain, tmp_path):
    """slope, relief and biome_weights are not stored: loading recomputes them with no difference."""
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    loaded = load_terrain(out)
    assert np.abs(loaded.slope - small_terrain.slope).max() == 0.0
    assert np.abs(loaded.relief - small_terrain.relief).max() == 0.0
    assert np.abs(loaded.biome_weights - small_terrain.biome_weights).max() == 0.0
    weights = loaded.biome_weights
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, rtol=1e-5)


def test_export_is_deterministic(small_terrain, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    export_terrain(small_terrain, first)
    export_terrain(small_terrain, second)
    for name in ("terrain.yaml", "terrain.npz", "preview.png"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_the_npz_is_much_smaller_than_the_raw_arrays(small_terrain, tmp_path):
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    raw = sum(getattr(small_terrain, name).nbytes for name in ARRAYS)
    packed = (out / "terrain.npz").stat().st_size
    assert packed < raw / 2  # water, sea, lake_id, river and flow_dir compress to almost nothing


def test_load_terrain_wants_a_bundle(small_terrain, tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_terrain(tmp_path / "missing")
    recipe_only = tmp_path / "recipe"
    recipe_only.mkdir()
    (recipe_only / "terrain.yaml").write_text("world: {size_cells: 128}\n")  # no seed
    with pytest.raises(ConfigError, match="no `seed`"):
        load_terrain(recipe_only)


def test_export_stamps_the_package_version(small_terrain, tmp_path):
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    data = yaml.safe_load((out / "terrain.yaml").read_text())
    assert data["version"] == VERSION


def test_client_data_round_trips_through_the_bundle(small_config, small_terrain, tmp_path):
    """A bundle carries the client recipe that made it: `client` is exported verbatim and recovered on load."""
    client = {"version": 3, "towns": {"count": 5, "capitol": None}, "biome_flora": {"forest": ["oak"]}}
    config = small_config.model_copy(update={"client": client})
    terrain = dataclasses.replace(small_terrain, config=config)
    out = tmp_path / "terrain"
    export_terrain(terrain, out)
    data = yaml.safe_load((out / "terrain.yaml").read_text())
    assert data["client"] == client
    assert load_terrain(out).config.client == client


def test_an_empty_client_section_is_not_written(small_terrain, tmp_path):
    """client defaults to an empty mapping; writing that into every bundle would be noise."""
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    assert "client" not in yaml.safe_load((out / "terrain.yaml").read_text())
    assert load_terrain(out).config.client == {}


def test_a_bundle_from_another_version_still_loads(small_terrain, tmp_path):
    """Compatibility is the client's call: worldgen never checks the bundle's version."""
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    path = out / "terrain.yaml"
    data = yaml.safe_load(path.read_text())
    data["version"] = "99.0.0"
    path.write_text(yaml.dump(data))
    assert load_terrain(out).config.version == "99.0.0"


def test_a_broken_bundle_names_the_version_in_its_error(small_terrain, tmp_path):
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    path = out / "terrain.yaml"
    data = yaml.safe_load(path.read_text())
    data["version"] = "0.0.1"
    path.write_text(yaml.dump(data))
    (out / "terrain.npz").unlink()
    with pytest.raises(ConfigError, match=r"worldgen 0\.0\.1"):
        load_terrain(out)


def test_generate_terrain_matches_the_bundle(small_config, small_terrain, tmp_path):
    """A terrain loaded from its bundle generates nothing new: the same recipe and seed reproduce it."""
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    loaded = load_terrain(out)
    again = generate_terrain(loaded.config, loaded.seed)
    np.testing.assert_array_equal(again.elevation, small_terrain.elevation)
    np.testing.assert_array_equal(again.color, small_terrain.color)


def test_terrain_data_load_and_patch_shape(small_terrain, tmp_path):
    """TerrainData is the class for a bundle on disk: load it, and it says how it tiles into patches."""
    out = tmp_path / "terrain"
    export_terrain(small_terrain, out)
    from worldgen import TerrainData

    terrain = TerrainData.load(out)
    assert terrain.patch_shape == (4, 4)  # 128 cells in 32-cell patches
    assert terrain.patch_at(40.0, 70.0).key == (1, 2)


def test_sampling_the_terrain_at_a_point(small_terrain):
    """A game asks what lies at (x, z): the ground, the water over it, and the biome."""
    x, z = 40.0, 70.0
    assert small_terrain.height_at(x, z) == small_terrain.elevation[70, 40]  # exact on a vertex
    between = small_terrain.height_at(40.5, 70.0)
    assert min(small_terrain.elevation[70, 40:42]) <= between <= max(small_terrain.elevation[70, 40:42])
    assert small_terrain.biome_at(x, z) in {rule.id for rule in small_terrain.config.biomes.rules}
    import numpy as np

    assert np.isnan(small_terrain.water_at(x, z)) or small_terrain.water_at(x, z) > small_terrain.height_at(x, z)


def test_the_bundled_default_recipe(small_terrain):
    from worldgen.config import DEFAULT_RECIPE

    config = load_config(DEFAULT_RECIPE)
    assert config.world.size_cells == 1024 and config.biomes.rules[-1].id == "default"


def test_write_npz_has_fixed_entries(tmp_path):
    """The archive is a deterministic zip: fixed names, fixed timestamps."""
    import zipfile

    preview.write_npz(tmp_path / "a.npz", {"b": np.arange(4), "a": np.zeros(2)})
    with zipfile.ZipFile(tmp_path / "a.npz") as archive:
        assert archive.namelist() == ["a.npy", "b.npy"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())

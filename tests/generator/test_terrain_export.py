"""The raw bundle, read as a client in another language reads it: through terrain.json, with no loader."""

import array
import dataclasses
import json
import math
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from worldgen.config import VERSION
from worldgen.generator.config import ConfigError
from worldgen.generator.terrain_export import RAW_ARRAYS
from worldgen.generator.terrain_io import ARRAYS, bundle_files, export_terrain, load_terrain, stale_files

EXAMPLES = Path(__file__).parents[2] / "examples"


@pytest.fixture(scope="module")
def raw_bundle(small_terrain, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("raw") / "terrain"
    export_terrain(small_terrain, out, "raw")
    return out


def manifest_of(out: Path) -> dict:
    return json.loads((out / "terrain.json").read_text(encoding="utf-8"))


def test_the_raw_bundle_holds_its_files_and_no_npz(raw_bundle):
    files = {str(p.relative_to(raw_bundle)) for p in raw_bundle.rglob("*") if p.is_file()}
    assert files == set(bundle_files("raw"))
    assert "terrain.npz" not in files


def test_every_array_reads_back_through_the_manifest(small_terrain, raw_bundle):
    """Each file is product(shape) * itemsize bytes of the manifest's dtype, little-endian, row-major [z][x]."""
    arrays = manifest_of(raw_bundle)["arrays"]
    assert set(arrays) == {*ARRAYS, "slope", "relief", "biome_weights"}
    for name, entry in arrays.items():
        dtype = np.dtype(entry["dtype"]).newbyteorder("<")
        path = raw_bundle / entry["file"]
        assert path.stat().st_size == math.prod(entry["shape"]) * dtype.itemsize, name
        read = np.fromfile(path, dtype=dtype).reshape(entry["shape"])
        expected = getattr(small_terrain, name)
        assert list(expected.shape) == entry["shape"], name
        np.testing.assert_array_equal(read, expected.astype(dtype), err_msg=name)
    assert np.isnan(np.fromfile(raw_bundle / "arrays/water.bin", "<f8")).any()  # NaN = dry
    sea = np.fromfile(raw_bundle / "arrays/sea.bin", "<u1")
    assert set(np.unique(sea)) <= {0, 1}


def test_elevation_reads_without_numpy(small_terrain, raw_bundle):
    """What a client with only a file read does: little-endian doubles, sample (x, z) at z * width + x."""
    heights = array.array("d")
    heights.frombytes((raw_bundle / "arrays/elevation.bin").read_bytes())  # this host is little-endian
    width = small_terrain.elevation.shape[1]
    for x, z in ((0, 0), (40, 70), (width - 1, width - 1)):
        assert heights[z * width + x] == small_terrain.elevation[z, x]


def test_meta_json_is_the_npz_bundles_meta(small_terrain, raw_bundle, tmp_path):
    npz = tmp_path / "npz"
    export_terrain(small_terrain, npz)
    with zipfile.ZipFile(npz / "terrain.npz") as archive, archive.open("meta.npy") as entry:
        npz_meta = json.loads(str(np.lib.format.read_array(entry)))
    assert json.loads((raw_bundle / "meta.json").read_text(encoding="utf-8")) == npz_meta


def test_the_manifest(small_terrain, raw_bundle):
    text = (raw_bundle / "terrain.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert text == json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    assert manifest["format"] == "worldgen-raw" and manifest["format_version"] == 1
    assert manifest["version"] == VERSION and manifest["seed"] == small_terrain.seed
    assert manifest["world"]["size_cells"] == 128 and manifest["byte_order"] == "little"
    rules = small_terrain.config.biomes.rules
    assert manifest["biomes"] == [{"id": rule.id, "color": list(rule.ground.color)} for rule in rules]
    assert manifest["client"] == {}
    assert manifest["d8"][:3] == [[1, 0], [1, 1], [0, 1]]  # E, SE, S as (dx, dz)
    for name, entry in manifest["arrays"].items():
        if name in ("lake_id", "river", "flow_dir"):
            assert isinstance(entry["nodata"], int), name
        else:
            assert "nodata" not in entry, name
    assert manifest["arrays"]["water"]["nan"] == "dry"
    assert [n for n, entry in manifest["arrays"].items() if "nan" in entry] == ["water"]


def test_the_manifest_carries_the_client_section(small_config, small_terrain, tmp_path):
    client = {"version": 3, "towns": {"count": 5, "capitol": None}}
    terrain = dataclasses.replace(small_terrain, config=small_config.model_copy(update={"client": client}))
    export_terrain(terrain, tmp_path, "raw")
    assert manifest_of(tmp_path)["client"] == client


def test_the_raw_export_is_deterministic(small_terrain, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    export_terrain(small_terrain, first, "raw")
    export_terrain(small_terrain, second, "raw")
    for name in bundle_files("raw"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_a_folder_never_holds_both_formats(small_terrain, tmp_path):
    (tmp_path / "notes.txt").write_text("mine")
    export_terrain(small_terrain, tmp_path, "npz")
    export_terrain(small_terrain, tmp_path, "raw")
    assert not (tmp_path / "terrain.npz").exists() and (tmp_path / "terrain.json").is_file()
    export_terrain(small_terrain, tmp_path, "npz")
    assert {p.name for p in tmp_path.iterdir()} == {"terrain.yaml", "terrain.npz", "preview.png", "notes.txt"}
    assert set(stale_files("npz")) == set(bundle_files("raw")) - set(bundle_files("npz"))


def test_a_raw_bundle_keeps_a_folder_of_its_own_under_arrays(small_terrain, tmp_path):
    """Only the files the raw bundle writes are removed: arrays/ stays when something else is in it."""
    export_terrain(small_terrain, tmp_path, "raw")
    (tmp_path / "arrays" / "mine.bin").write_bytes(b"x")
    export_terrain(small_terrain, tmp_path, "npz")
    assert [p.name for p in (tmp_path / "arrays").iterdir()] == ["mine.bin"]


def test_load_terrain_refuses_a_raw_bundle(raw_bundle):
    with pytest.raises(ConfigError, match="raw export"):
        load_terrain(raw_bundle)


def test_the_raw_arrays_cover_the_stored_ones():
    assert list(RAW_ARRAYS)[:len(ARRAYS)] == list(ARRAYS)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_javascript_example_reads_the_bundle(small_terrain, raw_bundle):
    result = subprocess.run(["node", str(EXAMPLES / "read_raw.mjs"), str(raw_bundle)], capture_output=True, text=True,
                            check=True)
    assert f"elevation at (40, 70): {small_terrain.elevation[70, 40]:.3f} m" in result.stdout
    assert f"biome at (40, 70): {small_terrain.biome_at(40.0, 70.0)}" in result.stdout

"""Patch meshes and textures on the GPU. Main thread only: OpenGL calls must stay on the thread that
owns the context."""

import numpy as np
import pyray as rl

from worldgen.world.mesh import MeshArrays, PatchMeshArrays

MAX_INDEXED_VERTICES = 65535  # raylib meshes use 16-bit indices


class GpuMesh:
    """A raylib mesh uploaded from MeshArrays. raylib frees the arrays itself, so they are copied
    into memory raylib allocated."""

    def __init__(self, arrays: MeshArrays, dynamic: bool = False):
        positions, normals, texcoords, indices = arrays.positions, arrays.normals, arrays.texcoords, arrays.indices
        colors, texcoords2 = arrays.colors, arrays.texcoords2
        if len(positions) > MAX_INDEXED_VERTICES:  # too many for 16-bit indices: draw unindexed
            flat = indices.ravel()
            positions, texcoords = positions[flat], texcoords[flat]
            normals = normals[flat] if normals is not None else None
            colors = colors[flat] if colors is not None else None
            texcoords2 = texcoords2[flat] if texcoords2 is not None else None
            indices = None
        self.ptr = rl.ffi.new("Mesh *")
        mesh = self.ptr[0]
        mesh.vertexCount = len(positions)
        mesh.triangleCount = len(arrays.indices)
        mesh.vertices = _raylib_copy(positions, "float *")
        mesh.texcoords = _raylib_copy(texcoords, "float *")
        if normals is not None:
            mesh.normals = _raylib_copy(normals, "float *")
        if colors is not None:
            mesh.colors = _raylib_copy(colors.astype(np.uint8), "unsigned char *")
        if texcoords2 is not None:
            mesh.texcoords2 = _raylib_copy(texcoords2.astype(np.float32), "float *")
        if indices is not None:
            mesh.indices = _raylib_copy(indices.astype(np.uint16), "unsigned short *")
        rl.upload_mesh(self.ptr, dynamic)
        self.nbytes = arrays.nbytes

    @property
    def mesh(self):
        return self.ptr[0]

    def unload(self) -> None:
        rl.unload_mesh(self.ptr[0])


class GpuTexture:
    """An RGB texture with mipmaps, from (rows, cols, 3) uint8 pixels."""

    def __init__(self, pixels: np.ndarray):
        rows, cols, _ = pixels.shape
        data = rl.ffi.from_buffer("unsigned char[]", np.ascontiguousarray(pixels))
        image = rl.Image(data, cols, rows, 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
        self.texture = rl.load_texture_from_image(image)
        rl.gen_texture_mipmaps(self.texture)
        rl.set_texture_filter(self.texture, rl.TextureFilter.TEXTURE_FILTER_TRILINEAR)
        # Terrain is mostly seen at grazing angles: without this, per-cell detail aliases into streaks
        rl.set_texture_filter(self.texture, rl.TextureFilter.TEXTURE_FILTER_ANISOTROPIC_16X)
        rl.set_texture_wrap(self.texture, rl.TextureWrap.TEXTURE_WRAP_CLAMP)

    def unload(self) -> None:
        rl.unload_texture(self.texture)


class GpuCellTexture:
    """A one-channel texture of per-cell values, (rows, cols) uint8, read exactly: no filtering, no mipmaps."""

    def __init__(self, values: np.ndarray):
        rows, cols = values.shape
        data = rl.ffi.from_buffer("unsigned char[]", np.ascontiguousarray(values, np.uint8))
        image = rl.Image(data, cols, rows, 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE)
        self.texture = rl.load_texture_from_image(image)
        rl.set_texture_filter(self.texture, rl.TextureFilter.TEXTURE_FILTER_POINT)
        rl.set_texture_wrap(self.texture, rl.TextureWrap.TEXTURE_WRAP_CLAMP)

    def unload(self) -> None:
        rl.unload_texture(self.texture)


class GpuPatch:
    """A patch on the GPU: terrain, optional water, colour texture, optional per-cell ground values."""

    def __init__(self, arrays: PatchMeshArrays):
        self.key, self.lod = arrays.key, arrays.lod
        self.terrain = GpuMesh(arrays.terrain)
        self.water = GpuMesh(arrays.water) if arrays.water is not None else None
        self.texture = GpuTexture(arrays.color)
        self.ground = GpuCellTexture(arrays.ground) if arrays.ground is not None else None
        self.nbytes = arrays.nbytes

    def unload(self) -> None:
        self.terrain.unload()
        if self.water is not None:
            self.water.unload()
        self.texture.unload()
        if self.ground is not None:
            self.ground.unload()


def _raylib_copy(array: np.ndarray, ctype: str):
    data = np.ascontiguousarray(array)
    pointer = rl.mem_alloc(data.nbytes)
    rl.ffi.memmove(pointer, rl.ffi.from_buffer(data), data.nbytes)
    return rl.ffi.cast(ctype, pointer)

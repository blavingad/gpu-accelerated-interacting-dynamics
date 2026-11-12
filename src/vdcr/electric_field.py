"""OpenFOAM electric-field loading and JAX interpolation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ElectricFieldGrid:
    values: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray

    @property
    def upper_bound(self) -> np.ndarray:
        return self.origin + self.spacing * (np.asarray(self.values.shape[:3]) - 1)


def load_electric_field_grid(path: str | Path) -> ElectricFieldGrid:
    with np.load(path) as data:
        values = np.asarray(data["E"], dtype=np.float32)
        origin = np.asarray(data["origin"], dtype=np.float32)
        spacing = np.asarray(data["spacing"], dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("Electric field must have shape (nx, ny, nz, 3).")
    if np.any(np.asarray(values.shape[:3]) < 2) or np.any(spacing <= 0):
        raise ValueError("Electric-field grid must have >=2 points/axis and positive spacing.")
    return ElectricFieldGrid(values, origin, spacing)


def trilinear_field(positions, values, origin, spacing):
    """Interpolate a regular vector grid; return zero outside its bounds."""
    shape = jnp.asarray(values.shape[:3], dtype=jnp.int32)
    coordinate = (positions - origin) / spacing
    lower = jnp.floor(coordinate).astype(jnp.int32)
    inside = jnp.all((coordinate >= 0) & (coordinate <= shape - 1), axis=1)
    lower = jnp.clip(lower, 0, shape - 2)
    fraction = jnp.clip(coordinate - lower, 0.0, 1.0)

    ix, iy, iz = lower[:, 0], lower[:, 1], lower[:, 2]
    fx, fy, fz = fraction[:, 0:1], fraction[:, 1:2], fraction[:, 2:3]
    c000 = values[ix, iy, iz]
    c100 = values[ix + 1, iy, iz]
    c010 = values[ix, iy + 1, iz]
    c110 = values[ix + 1, iy + 1, iz]
    c001 = values[ix, iy, iz + 1]
    c101 = values[ix + 1, iy, iz + 1]
    c011 = values[ix, iy + 1, iz + 1]
    c111 = values[ix + 1, iy + 1, iz + 1]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    field = c0 * (1 - fz) + c1 * fz
    return jnp.where(inside[:, None], field, 0.0)

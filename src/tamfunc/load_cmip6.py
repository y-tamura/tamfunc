#%%
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import xarray as xr

ROOT = Path("/nagoya/HighResMIP/Regrid")
DEFAULT_VAR = "zos"
MEAN_VARIANT_LABEL = "variant-mean"

#%%
def resolve_variant_dir(base: Path, variant_label: str | None = None) -> Path:
    if variant_label is not None:
        variant_dir = base / variant_label
        if not variant_dir.is_dir():
            raise FileNotFoundError(f"variant_label directory not found: {variant_dir}")
        return variant_dir

    mean_dir = base / MEAN_VARIANT_LABEL
    if mean_dir.is_dir():
        return mean_dir

    variant_dirs = sorted(
        path for path in base.iterdir()
        if path.is_dir() and path.name != MEAN_VARIANT_LABEL
    )
    if not variant_dirs:
        raise FileNotFoundError(f"no variant directories found under: {base}")
    if len(variant_dirs) > 1:
        names = ", ".join(path.name for path in variant_dirs)
        raise ValueError(
            f"multiple original variant_label directories remain under {base}: {names}. "
            f"Create {MEAN_VARIANT_LABEL} first or pass variant_label explicitly."
        )
    return variant_dirs[0]


#%%
def open_cmip6regrid(
    model: str,
    var: str = DEFAULT_VAR,
    grid: str = "r360x180",
    exp: str = "hist-1950",
    table: str = "Omon",
    root: Path = ROOT,
    variant_label: str | None = None,
) -> xr.Dataset:
    base = root / exp / table / var / model
    variant_dir = resolve_variant_dir(base, variant_label=variant_label)
    paths = sorted(variant_dir.rglob(f"*_{grid}.nc"))
    if not paths:
        raise FileNotFoundError(f"no files matched under: {variant_dir} for grid={grid}")

    return xr.open_mfdataset(
        paths,
        combine="by_coords",
        parallel=True,
    )[[var]]


def open_cmip6regrid_many(
    models: list[str],
    var: str = DEFAULT_VAR,
    grid: str = "r360x180",
    exp: str = "hist-1950",
    table: str = "Omon",
    root: Path = ROOT,
    variant_label: str | None = None,
) -> dict[str, xr.Dataset]:
    return {
        model: open_cmip6regrid(
            model=model,
            var=var,
            grid=grid,
            exp=exp,
            table=table,
            root=root,
            variant_label=variant_label,
        )
        for model in models
    }


def require_grid(ds: xr.Dataset, ny: int, nx: int, var: str = DEFAULT_VAR) -> xr.Dataset:
    sizes = ds[var].sizes
    got = (sizes.get("lat"), sizes.get("lon"))
    expected = (ny, nx)
    if got != expected:
        raise ValueError(f"expected grid {ny}x{nx}, got {got[0]}x{got[1]}")
    return ds
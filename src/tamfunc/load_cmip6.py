#%%
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import xarray as xr

ROOT = Path("/nagoya/HighResMIP/Regrid")
DEFAULT_VAR = "zos"
MEAN_VARIANT_LABEL = "variant-mean"
MEAN_VERSION_TAG = "v00000000"


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


def collect_paths_for_variant(variant_dir: Path, grid: str) -> list[Path]:
    if variant_dir.name == MEAN_VARIANT_LABEL:
        version_dirs = sorted(variant_dir.glob(f"*/{MEAN_VERSION_TAG}"))
        if not version_dirs:
            raise FileNotFoundError(
                f"{MEAN_VARIANT_LABEL} exists under {variant_dir.parent}, but {MEAN_VERSION_TAG} was not found. "
                "Recreate the averages or remove stale variant-mean directories."
            )
    else:
        version_dirs = []
        for grid_dir in sorted(path for path in variant_dir.iterdir() if path.is_dir()):
            candidates = sorted(path for path in grid_dir.iterdir() if path.is_dir())
            if candidates:
                version_dirs.append(candidates[-1])

    paths: list[Path] = []
    for version_dir in version_dirs:
        paths.extend(sorted(version_dir.glob(f"*_{grid}.nc")))
    return paths


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
    paths = collect_paths_for_variant(variant_dir, grid=grid)
    if not paths:
        raise FileNotFoundError(f"no files matched under: {variant_dir} for grid={grid}")
    print(f"opening {len(paths)} files for model {model}")
    return xr.open_mfdataset(
        paths,
        combine="by_coords",
        parallel=False,
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
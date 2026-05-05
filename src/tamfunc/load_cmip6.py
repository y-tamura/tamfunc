#%%
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import xarray as xr

ROOT = Path("/nagoya/HighResMIP/Regrid")
DEFAULT_VAR = "zos"
MEAN_VARIANT_LABEL = "variant-mean"
MEAN_VERSION_TAG = "v00000000"


def original_variant_dirs(base: Path) -> list[Path]:
    variant_dirs = sorted(
        path for path in base.iterdir()
        if path.is_dir() and path.name != MEAN_VARIANT_LABEL
    )
    if not variant_dirs:
        raise FileNotFoundError(f"no variant directories found under: {base}")
    return variant_dirs


def resolve_variant_dir(
    base: Path,
    variant_label: str | None = None,
    prefer_variant_mean: bool = True,
) -> Path:
    if variant_label is not None:
        variant_dir = base / variant_label
        if not variant_dir.is_dir():
            raise FileNotFoundError(f"variant_label directory not found: {variant_dir}")
        return variant_dir

    if prefer_variant_mean:
        mean_dir = base / MEAN_VARIANT_LABEL
        if mean_dir.is_dir():
            return mean_dir

    variant_dirs = original_variant_dirs(base)
    if len(variant_dirs) > 1:
        names = ", ".join(path.name for path in variant_dirs)
        fallback = (
            f"Create {MEAN_VARIANT_LABEL} first or pass variant_label explicitly."
            if prefer_variant_mean
            else "Pass variant_label explicitly or use open_cmip6regrid_variants()."
        )
        raise ValueError(
            f"multiple original variant_label directories remain under {base}: {names}. "
            f"{fallback}"
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
import os
import numpy as np
def open_cmip6regrid(
    model: str,
    var: str = DEFAULT_VAR,
    grid: str = "r360x180",
    exp: str = "hist-1950",
    table: str = "Omon",
    root: Path = ROOT,
    variant_label: str | None = None,
    prefer_variant_mean: bool = True,
) -> xr.DataArray:
    base = root / exp / table / var / model
    variant_dir = resolve_variant_dir(
        base,
        variant_label=variant_label,
        prefer_variant_mean=prefer_variant_mean,
    )
    paths = collect_paths_for_variant(variant_dir, grid=grid)
    if not paths:
        raise FileNotFoundError(f"no files matched under: {variant_dir} for grid={grid}")
    print(f"opening {len(paths)} files for model {model}, variant {variant_dir.name}")
    return xr.open_mfdataset(
        paths,
        combine="by_coords",
        parallel=False,
    )[var]


def open_cmip6regrid_variants(
    model: str,
    var: str = DEFAULT_VAR,
    grid: str = "r360x180",
    exp: str = "hist-1950",
    table: str = "Omon",
    root: Path = ROOT,
) -> dict[str, xr.DataArray]:
    base = root / exp / table / var / model
    return {
        variant_dir.name: open_cmip6regrid(
            model=model,
            var=var,
            grid=grid,
            exp=exp,
            table=table,
            root=root,
            variant_label=variant_dir.name,
        )
        for variant_dir in original_variant_dirs(base)
    }


def open_cmip6regrid_many(
    models: list[str],
    var: str = DEFAULT_VAR,
    grid: str = "r360x180",
    exp: str = "hist-1950",
    table: str = "Omon",
    root: Path = ROOT,
    variant_label: str | None = None,
    prefer_variant_mean: bool = True,
) -> dict[str, xr.DataArray]:
    return {
        model: open_cmip6regrid(
            model=model,
            var=var,
            grid=grid,
            exp=exp,
            table=table,
            root=root,
            variant_label=variant_label,
            prefer_variant_mean=prefer_variant_mean,
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
#%%-----------------------------------------------------------------
# Open Jet latitude
#------------------------------------------------------------------
def jetlat_path(exp: str, current: str, model: str, variant_label: str) -> str:
    fname = f"{current}_latitude_gflow_{model}_{exp}_{variant_label}_720x1440.nc"
    return f"{ROOT}/{exp}/JetLatitude/{current}/{model}/{variant_label}/{fname}"


def iter_variant_labels(exp: str, current: str, model: str, variant_mean: bool = False) -> list[str]:
    model_dir = f"{ROOT}/{exp}/JetLatitude/{current}/{model}"
    if not os.path.isdir(model_dir):
        return []
    variant_labels = sorted(
        variant_label
        for variant_label in os.listdir(model_dir)
        if os.path.isdir(f"{model_dir}/{variant_label}") and variant_label != "old"
    )
    if variant_mean:
        return [variant_label for variant_label in variant_labels if variant_label == VARIANT_MEAN_LABEL]
    return [variant_label for variant_label in variant_labels if variant_label != VARIANT_MEAN_LABEL]


def open_current_model(
    exp: str,
    current: str,
    model: str,
    variant_mean: bool = False,
) -> xr.DataArray | None:
    arrays = []
    variants = []
    for variant_label in iter_variant_labels(exp, current, model, variant_mean=variant_mean):
        path = jetlat_path(exp, current, model, variant_label)
        if not os.path.exists(path):
            print(f"[missing] {path}")
            continue
        da = xr.open_dataarray(path)
        arrays.append(da)
        variants.append(variant_label)

    if not arrays:
        return None

    variant_coord = xr.DataArray(variants, dims="variant_label", name="variant_label")
    return xr.concat(arrays, dim=variant_coord, join="outer")


def open_current(
    exp: str,
    current: str,
    models: list[str],
    variant_mean: bool = False,
) -> dict[str, xr.DataArray]:
    data = {}
    for model in models:
        da = open_current_model(exp, current, model, variant_mean=variant_mean)
        if da is None:
            print(f"[missing current] {current} {model}")
            continue
        data[model] = da
        print(f"[opened] {current} {model}: {dict(da.sizes)}")
    return data

def concat_kuroshio_ke(kuro_da: xr.DataArray, ke_da: xr.DataArray) -> xr.DataArray:
    kuro_da, ke_da = xr.align(kuro_da, ke_da, join="inner", exclude="lon")
    ke_lons = np.round(ke_da.lon.values.astype(float), 6)
    kuro_lons = np.round(kuro_da.lon.values.astype(float), 6)
    kuro_unique = kuro_da.isel(lon=~np.isin(kuro_lons, ke_lons))
    return xr.concat([kuro_unique, ke_da], dim="lon").sortby("lon")


def concat_current_dicts(
    dict_kuroshio_lat: dict[str, xr.DataArray],
    dict_ke_lat: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    jetconcat_lat = {}
    for model in sorted(set(dict_kuroshio_lat) & set(dict_ke_lat)):
        common_variants = np.intersect1d(
            dict_kuroshio_lat[model].variant_label.values,
            dict_ke_lat[model].variant_label.values,
        )
        if len(common_variants) == 0:
            print(f"[skip] {model}: no common variant between Kuroshio and KE")
            continue

        kuro_da = dict_kuroshio_lat[model].sel(variant_label=common_variants)
        ke_da = dict_ke_lat[model].sel(variant_label=common_variants)
        jetconcat_lat[model] = concat_kuroshio_ke(kuro_da, ke_da)
        print(f"[concat] {model}: {dict(jetconcat_lat[model].sizes)}")
    return jetconcat_lat

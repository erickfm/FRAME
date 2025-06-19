#!/usr/bin/env python3
# dataset_check.py — Inspect raw parquet columns and mapping dicts for type mismatches

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Import your categorical maps
from cat_maps import STAGE_MAP, CHARACTER_MAP, ACTION_MAP, PROJECTILE_TYPE_MAP
from dataset import MeleeFrameDatasetWithDelay


def inspect_column(df, col, max_unique=20):
    if col not in df.columns:
        print(f"Column '{col}' not found in DataFrame.")
        return
    series = df[col]
    print(f"\nColumn: {col}")
    print(f"  dtype: {series.dtype}")
    uniques = pd.unique(series.dropna())
    sample = uniques[:max_unique]
    print(f"  unique values (up to {max_unique}): {sample.tolist()}")
    if len(uniques) > max_unique:
        print(f"  ... (+{len(uniques) - max_unique} more unique values)")


def inspect_map(name, m, max_keys=20):
    print(f"\nMapping: {name}")
    if not isinstance(m, dict):
        print(f"  <Not a dict: {type(m)}>\n  value: {m}")
        return
    keys = list(m.keys())
    if not keys:
        print("  <empty dict>")
        return
    print(f"  key type: {type(keys[0])}")
    sample = keys[:max_keys]
    print(f"  sample keys (up to {max_keys}): {sample}")
    if len(keys) > max_keys:
        print(f"  ... (+{len(keys) - max_keys} more keys)")


def main():
    p = argparse.ArgumentParser(
        description="Check dataset column types, unique values and mapping dicts"
    )
    p.add_argument("-f", "--file", default=None,
                   help="Path to a .parquet file (defaults to first in data-dir)")
    p.add_argument("-d", "--data-dir", default="./data",
                   help="Directory containing parquet files")
    p.add_argument("-i", "--index", type=int, default=0,
                   help="Sample index to instantiate dataset mappings")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if args.file:
        parquet_path = Path(args.file)
    else:
        files = sorted(data_dir.glob("*.parquet"))
        if not files:
            raise RuntimeError(f"No .parquet files in {data_dir}")
        parquet_path = files[0]

    print(f"Loading parquet: {parquet_path}\n")
    df = pd.read_parquet(parquet_path)
    if "frame" in df.columns:
        df = df[df["frame"] >= 0]

    # Columns to inspect
    cols = [
        "self_character", "opp_character",
        "self_action",    "opp_action",
    ]
    for col in cols:
        inspect_column(df, col)

    # Inspect fixed global maps
    inspect_map("STAGE_MAP",           STAGE_MAP)
    inspect_map("CHARACTER_MAP",       CHARACTER_MAP)
    inspect_map("ACTION_MAP",          ACTION_MAP)
    inspect_map("PROJECTILE_TYPE_MAP", PROJECTILE_TYPE_MAP)

    # Instantiate dataset to inspect dynamic maps
    print("\nInstantiating dataset to inspect dynamic maps...")
    ds = MeleeFrameDatasetWithDelay(
        parquet_dir=args.data_dir,
        sequence_length=30,
        reaction_delay=1,
    )

    # Built-in enum maps (_enum_maps)
    for name, m in ds._enum_maps.items():
        inspect_map(f"_enum_maps['{name}']", m)

    # Dynamic categorical maps: only dicts ending with '_map'
    dynamic_maps = [attr for attr in vars(ds) if attr.endswith("_map")]
    for attr in dynamic_maps:
        m = getattr(ds, attr)
        if isinstance(m, dict):
            inspect_map(attr, m)

if __name__ == "__main__":
    main()

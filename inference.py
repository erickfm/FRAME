#!/usr/bin/env python3
# inference.py
#
# Real-time Melee inference mirroring MeleeFrameDatasetWithDelay → FramePredictor.

import argparse
import signal
import sys
from pathlib import Path
from collections import deque
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import melee

# ─── your existing maps & model imports ─────────────────────────────────────
from cat_maps import STAGE_MAP, CHARACTER_MAP, ACTION_MAP, PROJECTILE_TYPE_MAP
from model import FramePredictor, ModelConfig
from dataset import MeleeFrameDatasetWithDelay  # only for its spec

# ─── 0) Device & checkpoint load (unchanged) ───────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

ckpt_files = list(Path("./checkpoints").glob("epoch_*.pt"))
ckpt_path  = max(ckpt_files, key=lambda p: p.stat().st_mtime, default=None)
if ckpt_path is None:
    print("No checkpoint found – exiting."); sys.exit(1)

ckpt = torch.load(ckpt_path, map_location=DEVICE)
cfg  = ModelConfig(**ckpt.get("config", {}))
print(f"Reconstructed ModelConfig: {cfg}")

model = FramePredictor(cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

ROLLING_WINDOW = cfg.max_seq_len
MAX_PROJ       = 8

# ─── 1) Re-use Dataset’s grouping & categorical spec ────────────────────────
_spec = MeleeFrameDatasetWithDelay.__new__(MeleeFrameDatasetWithDelay)
_spec.feature_groups     = _spec._build_feature_groups()
_spec._categorical_cols  = [col for _, meta in _spec._walk_groups(return_meta=True)
                             if meta["ftype"] == "categorical"
                             for col in meta["cols"]]
_spec._enum_maps         = {
    "stage":      STAGE_MAP,
    "_character": CHARACTER_MAP,
    "_action":    ACTION_MAP,
    "_type":      PROJECTILE_TYPE_MAP,
}

# ─── 2) rows_to_state_seq: mirror Dataset.__getitem__ exactly ──────────────
def rows_to_state_seq(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # 1) Build DataFrame and drop timestamp
    df = pd.DataFrame(rows).drop(columns=["startAt"], errors="ignore")

    # 2) Fill NaNs for numeric/bool
    num_cols  = [c for c, t in df.dtypes.items() if t.kind in ("i","f")]
    bool_cols = [c for c, t in df.dtypes.items() if t == "bool"]
    df[num_cols]  = df[num_cols].fillna(0.0)
    df[bool_cols] = df[bool_cols].fillna(False)

    # 3) Map categoricals via enum or raw int
    def _map_cat(col: str, x: Any) -> int:
        if col == "stage":
            return _spec._enum_maps["stage"].get(x, 0)
        for suf, m in _spec._enum_maps.items():
            if suf != "stage" and col.endswith(suf):
                return m.get(x, 0)
        try:
            return max(int(x), 0)
        except:
            return 0

    for c in _spec._categorical_cols:
        df[c] = (
            df[c]
            .fillna(0)
            .map(lambda x, col=c: _map_cat(col, x))
            .astype("int64")
        )

    # 4) Prepare synthetic flags and stub missing geometry or button cols in bulk
    new_cols: Dict[str, Any] = {}
    # synthetic Nana flags
    new_cols["self_nana_present"] = (df.get("self_nana_character", 0) > 0).astype("float32")
    new_cols["opp_nana_present"]  = (df.get("opp_nana_character", 0) > 0).astype("float32")
    # ensure all feature spec cols exist
    for _, meta in _spec._walk_groups(return_meta=True):
        for col in meta["cols"]:
            if col not in df.columns:
                if meta["ftype"] == "categorical":
                    new_cols[col] = 0
                else:
                    new_cols[col] = 0.0
    # Merge and defragment
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()

    # 5) Assemble state_seq dict
    state_seq: Dict[str, torch.Tensor] = {}
    for _, meta in _spec._walk_groups(return_meta=True):
        cols, ftype, entity = meta['cols'], meta['ftype'], meta['entity']
        key = f"{entity}_{ftype}" if entity != "global" else ftype

        if ftype == "categorical":
            for col in cols:
                arr = df[col].values.astype("int64")
                state_seq[col] = torch.from_numpy(arr).long().unsqueeze(0)
        else:
            mats = [torch.from_numpy(df[c].astype("float32").values) for c in cols]
            tensor = torch.stack(mats, dim=-1) if len(mats) > 1 else mats[0]
            state_seq[key] = tensor.unsqueeze(0)

    return state_seq

# ─── 3) run_inference wrapper ───────────────────────────────────────────────
@torch.no_grad()
def run_inference(win_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch = rows_to_state_seq(win_rows)
    for k, v in batch.items():
        batch[k] = v.to(DEVICE)
    preds = model(batch)
    return {k: v.cpu().squeeze(0) for k, v in preds.items()}

# ─── 4) Controller output (unchanged) ───────────────────────────────────────
def press_output(controller: melee.Controller, output: Dict[str, torch.Tensor], thresh: float = 0.5):
    mx, my = output["main_xy"].tolist()
    cx, cy = output["c_xy"].tolist()
    controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN, mx, my)
    controller.tilt_analog_unit(melee.enums.Button.BUTTON_C,    cx, cy)
    controller.press_shoulder(melee.enums.Button.BUTTON_L, output["L_val"].item())
    controller.press_shoulder(melee.enums.Button.BUTTON_R, output["R_val"].item())

    idx_to_button = [
        melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
        melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
        melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
        melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_START,
        melee.enums.Button.BUTTON_D_UP, melee.enums.Button.BUTTON_D_DOWN,
        melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,
    ]
    print(output["btn_probs"], idx_to_button)
    for prob, btn in zip(output["btn_probs"], idx_to_button):
        if prob.item() > thresh:
            controller.press_button(btn)
        else:
            controller.release_button(btn)

# ─── 5) Signal handling & main loop (unchanged) ─────────────────────────────
def check_port(value):
    iv = int(value)
    if iv not in {1,2,3,4}:
        raise argparse.ArgumentTypeError("Port must be 1-4")
    return iv

def signal_handler(sig, frame):
    for ctrl in controllers.values():
        ctrl.disconnect()
    console.stop()
    print("Shutting down…")
    sys.exit(0)

if __name__ == "__main__":
    DOLPHIN_APP = "/Users/erick/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app"
    ISO_PATH    = "/Users/erick/Downloads/melee.iso"

    console = melee.Console(path=DOLPHIN_APP, slippi_address="127.0.0.1", fullscreen=False)
    ports = [1, 2]
    controllers = {
        p: melee.Controller(console=console, port=p, type=melee.ControllerType.STANDARD)
        for p in ports
    }

    signal.signal(signal.SIGINT, signal_handler)
    console.run(iso_path=ISO_PATH)

    if not console.connect():
        print("Failed to connect to console."); sys.exit(1)
    for ctrl in controllers.values():
        if not ctrl.connect():
            print("Failed to connect controller."); sys.exit(1)
    print("Console + controllers connected.")

    rows_deque = deque(maxlen=ROLLING_WINDOW)

    while True:
        gs = console.step()
        if gs is None:
            continue

        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            p0, c0 = ports[0], controllers[ports[0]]
            p1, c1 = ports[1], controllers[ports[1]]
            melee.MenuHelper().menu_helper_simple(gs, c0, melee.Character.FALCO,
                                                 melee.Stage.POKEMON_STADIUM,
                                                 cpu_level=1, autostart=1)
            melee.MenuHelper().menu_helper_simple(gs, c1, melee.Character.FALCO,
                                                 melee.Stage.POKEMON_STADIUM,
                                                 cpu_level=0, autostart=0)
            continue

        # build frame dict with proj{k}_frame included
        row: Dict[str, Any] = {}
        row["stage"]    = gs.stage.name if gs.stage else ""
        row["frame"]    = gs.frame
        row["distance"] = gs.distance
        row["startAt"]  = gs.startAt

        for idx, (port, ps) in enumerate(gs.players.items()):
            pref = "self_" if idx == 0 else "opp_"
            row[f"{pref}port"]          = port
            row[f"{pref}character"]     = ps.character.name
            row[f"{pref}action"]        = ps.action.name
            row[f"{pref}action_frame"]  = ps.action_frame
            row[f"{pref}costume"]       = ps.costume

            for btn, st in ps.controller_state.button.items():
                row[f"{pref}btn_{btn.name}"] = int(st)

            row[f"{pref}main_x"], row[f"{pref}main_y"] = ps.controller_state.main_stick
            row[f"{pref}c_x"],     row[f"{pref}c_y"]   = ps.controller_state.c_stick
            row[f"{pref}l_shldr"]   = ps.controller_state.l_shoulder
            row[f"{pref}r_shldr"]   = ps.controller_state.r_shoulder

            row[f"{pref}percent"] = float(ps.percent)
            row[f"{pref}pos_x"]   = float(ps.position.x)
            row[f"{pref}pos_y"]   = float(ps.position.y)

            nana = ps.nana
            npref = f"{pref}nana_"
            if nana:
                row[f"{npref}character"]    = nana.character.name
                row[f"{npref}action"]       = nana.action.name
                row[f"{npref}action_frame"] = nana.action_frame
            else:
                row[f"{npref}character"]    = ""
                row[f"{npref}action"]       = ""
                row[f"{npref}action_frame"] = -1

        for j in range(MAX_PROJ):
            pp = f"proj{j}_"
            if j < len(gs.projectiles):
                p = gs.projectiles[j]
                row[f"{pp}owner"]   = p.owner
                row[f"{pp}type"]    = p.type.name
                row[f"{pp}subtype"] = p.subtype
                row[f"{pp}pos_x"]   = float(p.position.x)
                row[f"{pp}pos_y"]   = float(p.position.y)
                row[f"{pp}speed_x"] = float(p.speed.x)
                row[f"{pp}speed_y"] = float(p.speed.y)
                row[f"{pp}frame"]   = p.frame
            else:
                row[f"{pp}owner"]   = -1
                row[f"{pp}type"]    = ""
                row[f"{pp}subtype"] = -1
                row[f"{pp}pos_x"]   = float("nan")
                row[f"{pp}pos_y"]   = float("nan")
                row[f"{pp}speed_x"] = float("nan")
                row[f"{pp}speed_y"] = float("nan")
                row[f"{pp}frame"]   = -1

        rows_deque.append(row)
        if len(rows_deque) == ROLLING_WINDOW:
            pred = run_inference(list(rows_deque))
            ctrl = controllers[ports[1]]
            ctrl.release_all()
            press_output(ctrl, pred)
            ctrl.flush()

            if gs.frame % 60 == 0:
                print(f"[Frame {gs.frame}] main_xy={pred['main_xy'].tolist()}  "
                      f"btn_probs={pred['btn_probs'].tolist()[:4]}…")

#!/usr/bin/env python3
# inference.py
#
# Drive Dolphin in real-time and feed the last 30 frames into the
# FramePredictor model to estimate *next-frame* controller inputs.
# Assumes checkpoints live in ./checkpoints and model / dataset code
# are updated as per the new architecture (FramePredictor + FrameEncoder).

import argparse
import signal
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Any

import torch
import melee                         # pip install py-slippi-melee

from cat_maps import (
    STAGE_MAP,
    CHARACTER_MAP,
    PROJECTILE_TYPE_MAP,
    ACTION_MAP,
)
from model import FramePredictor, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# 0. Device & model init
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    else "cpu"
)
print(f"Device: {DEVICE}")

# Build config to match training
CFG = ModelConfig(
    max_seq_len=30,
    num_stages=32,
    num_ports=4,
    num_characters=26,
    num_actions=88,
    num_costumes=6,
    num_proj_types=160,
    num_proj_subtypes=40,
)

model = FramePredictor(CFG).to(DEVICE)
ckpt_path = Path("./checkpoints").glob("epoch_*.pt")
ckpt_path = max(ckpt_path, key=lambda p: p.stat().st_mtime, default=None)

if ckpt_path is None:
    print("No checkpoint found in ./checkpoints – exiting.")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint {ckpt_path}")

ROLLING_WINDOW = 30        # how many past frames to feed the model
MAX_PROJ       = 8         # projectile slots


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helper – map raw row dicts → model batch_state
# ─────────────────────────────────────────────────────────────────────────────
def encode_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert string categoricals → integer codes IN-PLACE.
    Unknown strings get code -1.
    """
    row["stage"] = STAGE_MAP.get(row.get("stage"), -1)
    for pref in ("p1_", "p2_"):
        row[f"{pref}character"]      = CHARACTER_MAP.get(row.get(f"{pref}character"), -1)
        row[f"{pref}action"]         = ACTION_MAP.get(row.get(f"{pref}action"), -1)
        row[f"{pref}nana_character"] = CHARACTER_MAP.get(row.get(f"{pref}nana_character"), -1)
        row[f"{pref}nana_action"]    = ACTION_MAP.get(row.get(f"{pref}nana_action"), -1)
    for j in range(MAX_PROJ):
        key = f"proj{j}_type"
        row[key] = PROJECTILE_TYPE_MAP.get(row.get(key), -1)
    return row


def rows_to_batch_state(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Convert the last 30 encoded rows into the exact dictionary of tensors
    FrameEncoder expects.  Shapes: (1, W) or (1, W, D).  Booleans → int.
    """
    W = len(rows)
    batch_state = defaultdict(list)

    for row in rows:
        for k, v in row.items():
            batch_state[k].append(v)

    tensor_state = {}
    for k, seq in batch_state.items():
        val0 = seq[0]
        if isinstance(val0, bool):
            tensor = torch.tensor(seq, dtype=torch.int32).unsqueeze(0)         # (1,W)
        elif isinstance(val0, int):
            tensor = torch.tensor(seq, dtype=torch.int64).unsqueeze(0)         # (1,W)
        else:  # float or nan
            seq = [0.0 if (v is None or (isinstance(v, float) and (v != v))) else v for v in seq]
            tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)       # (1,W)
        tensor_state[k] = tensor
    return tensor_state


@torch.no_grad()
def run_inference(win_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Returns model outputs for a 30-frame window (batch size 1)."""
    batch_state = rows_to_batch_state(win_rows)
    batch_state = {k: v.to(DEVICE) for k, v in batch_state.items()}
    preds = model(batch_state)
    # preds are batch-first; squeeze batch dim
    preds = {k: v.cpu().squeeze(0) for k, v in preds.items()}
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# 2. Controller helpers
# ─────────────────────────────────────────────────────────────────────────────
def press_output(controller: melee.Controller, output: Dict[str, torch.Tensor], thresh=0.7):
    """Apply predicted analog & button outputs to the Dolphin controller."""
    # Analog sticks
    mx, my = output["main_xy"].tolist()
    cx, cy = output["c_xy"].tolist()
    controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN, mx, my)
    controller.tilt_analog_unit(melee.enums.Button.BUTTON_C, cx, cy)
    # Triggers
    controller.press_shoulder(melee.enums.Button.BUTTON_L,  output["L_val"].item())
    controller.press_shoulder(melee.enums.Button.BUTTON_R,  output["R_val"].item())
    # Buttons
    idx_to_button = [
        melee.enums.Button.BUTTON_A,
        melee.enums.Button.BUTTON_B,
        melee.enums.Button.BUTTON_X,
        melee.enums.Button.BUTTON_Y,
        melee.enums.Button.BUTTON_Z,
        melee.enums.Button.BUTTON_L,
        melee.enums.Button.BUTTON_R,
        melee.enums.Button.BUTTON_START,
        melee.enums.Button.BUTTON_D_UP,
        melee.enums.Button.BUTTON_D_DOWN,
        melee.enums.Button.BUTTON_D_LEFT,
        melee.enums.Button.BUTTON_D_RIGHT,
    ]
    probs = output["btn_probs"]
    for p, btn in zip(probs, idx_to_button):
        if p.item() > thresh:
            controller.press_button(btn)
        else:
            controller.release_button(btn)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal handling
# ─────────────────────────────────────────────────────────────────────────────
def check_port(value):
    iv = int(value)
    if iv not in {1, 2, 3, 4}:
        raise argparse.ArgumentTypeError("Port must be 1-4")
    return iv


def signal_handler(sig, frame):
    for ctrl in controllers.values():
        ctrl.disconnect()
    console.stop()
    print("Shutting down…")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Dolphin loop
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # —─ Dolphin paths (EDIT to your setup)
    DOLPHIN_APP = "/Users/erick/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app"
    ISO_PATH    = "/Users/erick/Downloads/melee.iso"

    console = melee.Console(
        path=DOLPHIN_APP,
        slippi_address="127.0.0.1",
        fullscreen=False,
    )

    ports = [1, 2]
    controllers = {
        p: melee.Controller(console=console, port=p, type=melee.ControllerType.STANDARD)
        for p in ports
    }

    signal.signal(signal.SIGINT, signal_handler)
    console.run(iso_path=ISO_PATH)

    if not console.connect():
        print("Failed to connect to console.")
        sys.exit(1)
    for ctrl in controllers.values():
        if not ctrl.connect():
            print("Failed to connect controller.")
            sys.exit(1)
    print("Console + controllers connected.")

    rows_deque = deque(maxlen=ROLLING_WINDOW)

    while True:
        gs = console.step()
        if gs is None:
            continue

        if gs.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            # simple menu helper (autostart Falco ditto)
            port0, ctrl0 = list(controllers.items())[0]
            port1, ctrl1 = list(controllers.items())[1]
            melee.MenuHelper().menu_helper_simple(
                gs, ctrl0, melee.Character.FALCO, melee.Stage.BATTLEFIELD, cpu_level=1
            )
            melee.MenuHelper().menu_helper_simple(
                gs, ctrl1, melee.Character.FALCO, melee.Stage.BATTLEFIELD, autostart=0
            )
            continue

        # ── Build one dict 'row' per frame ───────────────────────────────
        row: Dict[str, Any] = {}
        stage = gs.stage
        row["stage"] = stage.name if stage else ""

        # Global numeric
        row["frame"] = gs.frame
        row["distance"] = gs.distance
        row["startAt"] = gs.startAt

        # Ports & players
        for idx, (port, ps) in enumerate(gs.players.items()):
            pref = "self_" if idx == 0 else "opp_"
            row[f"{pref}port"]       = port
            row[f"{pref}character"]  = ps.character.name
            row[f"{pref}action"]     = ps.action.name
            row[f"{pref}action_frame"] = ps.action_frame
            row[f"{pref}costume"]    = ps.costume

            # Buttons (bool→int)
            for btn, state in ps.controller_state.button.items():
                row[f"{pref}btn_{btn.name}"] = int(state)

            # Analog, triggers
            row[f"{pref}main_x"], row[f"{pref}main_y"] = ps.controller_state.main_stick
            row[f"{pref}c_x"], row[f"{pref}c_y"]       = ps.controller_state.c_stick
            row[f"{pref}l_shldr"] = ps.controller_state.l_shoulder
            row[f"{pref}r_shldr"] = ps.controller_state.r_shoulder

            # Numeric misc
            row[f"{pref}percent"] = float(ps.percent)
            row[f"{pref}pos_x"]   = float(ps.position.x)
            row[f"{pref}pos_y"]   = float(ps.position.y)

            # Nana (if exists)
            nana = ps.nana
            n_pref = f"{pref}nana_"
            if nana:
                row[f"{n_pref}character"]     = nana.character.name
                row[f"{n_pref}action"]        = nana.action.name
                row[f"{n_pref}action_frame"]  = nana.action_frame
            else:
                # sentinel values
                row[f"{n_pref}character"]    = ""
                row[f"{n_pref}action"]       = ""
                row[f"{n_pref}action_frame"] = -1

        # Projectiles
        for j in range(MAX_PROJ):
            pp = f"proj{j}_"
            if j < len(gs.projectiles):
                proj = gs.projectiles[j]
                row[f"{pp}owner"]   = proj.owner
                row[f"{pp}type"]    = proj.type.name
                row[f"{pp}subtype"] = proj.subtype
                row[f"{pp}pos_x"]   = float(proj.position.x)
                row[f"{pp}pos_y"]   = float(proj.position.y)
                row[f"{pp}speed_x"] = float(proj.speed.x)
                row[f"{pp}speed_y"] = float(proj.speed.y)
            else:
                row[f"{pp}owner"]   = -1
                row[f"{pp}type"]    = ""
                row[f"{pp}subtype"] = -1
                row[f"{pp}pos_x"]   = float("nan")
                row[f"{pp}pos_y"]   = float("nan")
                row[f"{pp}speed_x"] = float("nan")
                row[f"{pp}speed_y"] = float("nan")

        # Encode categoricals
        encode_row(row)
        rows_deque.append(row)

        # ── Once we have 30 frames, run inference ───────────────────────
        if len(rows_deque) == ROLLING_WINDOW:
            pred = run_inference(list(rows_deque))
            controller_self = controllers[ports[0]]
            controller_self.release_all()
            press_output(controller_self, pred)
            controller_self.flush()

            # Debug print (optional)
            if gs.frame % 60 == 0:
                print(f"[Frame {gs.frame}] predicted main_xy={pred['main_xy'].tolist()}  "
                      f"btn_probs={pred['btn_probs'].tolist()[:4]}…")

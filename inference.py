#!/usr/bin/env python3
# inference.py  –  FRAME real-time bot
# --------------------------------------------------------------------------
# * Sanitises any NaN/Inf weights on load
# * Mirrors Dataset preprocessing (inc. c-stick categorical)
# * Converts model outputs to Dolphin controller actions
# * Optional DEBUG hooks and asserts
# --------------------------------------------------------------------------

import math, signal, sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import melee

# ─────────────── User flags ───────────────────
DEBUG = False          # True = extra asserts + hooks
PRINT_BAD_ROWS = False # print raw non-finite values
# ───────────────────────────────────────────────

from cat_maps import STAGE_MAP, CHARACTER_MAP, ACTION_MAP, PROJECTILE_TYPE_MAP
from model    import FramePredictor, ModelConfig
from dataset  import MeleeFrameDatasetWithDelay

# ══════════════════════════════════════════════
# 0) Device & checkpoint
# ══════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpts = sorted(Path("./checkpoints").glob("epoch_*.pt"))
if not ckpts:
    sys.exit("No checkpoints found in ./checkpoints")
ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)
ckpt      = torch.load(ckpt_path, map_location=DEVICE)
cfg       = ModelConfig(**ckpt["config"])

model = FramePredictor(cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Loaded", ckpt_path, "→", DEVICE)

# ── 0.a  Sanitise non-finite parameters ------------------------------------
reinit = 0
for name, param in model.named_parameters():
    if not torch.isfinite(param).all():
        reinit += 1
        if param.ndim > 1:  # matrices / pos_emb
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        else:               # biases / 1-d tensors
            nn.init.zeros_(param)
        print(f"[SANITIZE] Re-initialised '{name}'")
if reinit:
    print(f"[SANITIZE] Fixed {reinit} parameter(s); continuing…")

# ══════════════════════════════════════════════
# 0.b Optional debug hooks
# ══════════════════════════════════════════════
if DEBUG:
    def nan_hook(mod, _, out):
        t = out[0] if isinstance(out, tuple) else out
        if not torch.isfinite(t).all():
            print(f"[HOOK] NaN/Inf in {mod._hook_name}")
            raise RuntimeError("Abort on non-finite layer")
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
            m._hook_name = n
            m.register_forward_hook(nan_hook)

# ══════════════════════════════════════════════
# 1) Feature spec clone
# ══════════════════════════════════════════════
_spec = MeleeFrameDatasetWithDelay.__new__(MeleeFrameDatasetWithDelay)
_spec.feature_groups = _spec._build_feature_groups()
_spec._categorical_cols = [
    c for _, m in _spec._walk_groups(return_meta=True)
    if m["ftype"] == "categorical" for c in m["cols"]
]
_spec._enum_maps = {
    "stage": STAGE_MAP,
    "_character": CHARACTER_MAP,
    "_action": ACTION_MAP,
    "_type": PROJECTILE_TYPE_MAP,
    "c_dir": {i: i for i in range(5)},  # 0-4 identity
}

# ══════════════════════════════════════════════
# 2) Helpers
# ══════════════════════════════════════════════
def encode_c_dir(df: pd.DataFrame, p: str, dead=0.15):
    dx = df[f"{p}_c_x"].astype(np.float32) - 0.5
    dy = 0.5 - df[f"{p}_c_y"].astype(np.float32)
    mag = np.hypot(dx, dy)
    cat = np.zeros_like(mag, np.int64)
    active = mag > dead
    horiz  = active & (np.abs(dx) >= np.abs(dy))
    vert   = active & (np.abs(dy)  > np.abs(dx))
    cat[horiz & (dx > 0)] = 4
    cat[horiz & (dx < 0)] = 3
    cat[vert  & (dy > 0)] = 1
    cat[vert  & (dy < 0)] = 2
    df[f"{p}_c_dir"] = cat

def _map_cat(col: str, x: Any) -> int:
    if col == "stage":
        return _spec._enum_maps["stage"].get(x, 0)
    if col.endswith("_c_dir"):
        return int(x) if 0 <= int(x) <= 4 else 0
    for suf, mp in _spec._enum_maps.items():
        if suf != "stage" and col.endswith(suf):
            return mp.get(x, 0)
    try:
        return max(int(x), 0)
    except:  # noqa: E722
        return 0

def assert_finite(t: torch.Tensor, tag: str):
    if DEBUG and not torch.isfinite(t).all():
        raise RuntimeError(f"NaN/Inf in {tag}")

def maybe_print_bad_row(row, fid):
    if PRINT_BAD_ROWS:
        bad = {k:v for k,v in row.items() if isinstance(v,float) and not math.isfinite(v)}
        if bad: print(f"[RAW] non-finite at frame {fid}: {bad}")

# ══════════════════════════════════════════════
# 3) rows → state_seq
# ══════════════════════════════════════════════
def rows_to_state_seq(rows: List[Dict[str, Any]]):
    df = pd.DataFrame(rows).drop(columns=["startAt"], errors="ignore")
    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    for p in ("self", "opp", "self_nana", "opp_nana"):
        if f"{p}_c_x" in df.columns:
            encode_c_dir(df, p)

    num_cols  = [c for c,t in df.dtypes.items() if t.kind in ("i","f")]
    bool_cols = [c for c,t in df.dtypes.items() if t == "bool"]
    df[num_cols]  = df[num_cols].fillna(0.0)
    df[bool_cols] = df[bool_cols].fillna(False)

    miss_cat = [c for c in _spec._categorical_cols if c not in df.columns]
    if miss_cat:
        df = pd.concat([df, pd.DataFrame({c:0 for c in miss_cat}, index=df.index)], axis=1)

    for col in _spec._categorical_cols:
        df[col] = df[col].map(lambda x,c=col: _map_cat(c,x)).astype("int64")

    df["self_nana_present"] = (df.get("self_nana_character",0)>0).astype("float32")
    df["opp_nana_present"]  = (df.get("opp_nana_character",0)>0).astype("float32")

    miss_num = {}
    for _, m in _spec._walk_groups(return_meta=True):
        if m["ftype"] != "categorical":
            for c in m["cols"]:
                if c not in df.columns: miss_num[c] = 0.0
    if miss_num:
        df = pd.concat([df, pd.DataFrame(miss_num, index=df.index)], axis=1)

    seq = {}
    for _, m in _spec._walk_groups(return_meta=True):
        cols, t, ent = m["cols"], m["ftype"], m["entity"]
        key = f"{ent}_{t}" if ent!="global" else t
        if t=="categorical":
            for col in cols:
                x = torch.from_numpy(df[col].values).long().unsqueeze(0)
                assert_finite(x, col); seq[col]=x
        else:
            mats=[torch.from_numpy(df[c].astype(np.float32).values) for c in cols]
            x=(torch.stack(mats,-1) if len(mats)>1 else mats[0]).unsqueeze(0)
            assert_finite(x,key); seq[key]=x
    return seq

@torch.no_grad()
def infer(win_rows):
    batch = rows_to_state_seq(win_rows)
    for k in batch: batch[k] = batch[k].to(DEVICE)
    out = model(batch)
    for k,v in out.items(): assert_finite(v, f"pred[{k}]")
    return {k: v.cpu().squeeze(0) for k,v in out.items()}

# ══════════════════════════════════════════════
# 4) Controller helpers
# ══════════════════════════════════════════════
C_LOOK = {0:(.5,.5),1:(.5,1),2:(.5,0),3:(0,.5),4:(1,.5)}
def clamp(x, d=.5): return min(max(d if not math.isfinite(x) else x,0),1)

def press(ctrl, pred, th=.7):
    mx,my = map(clamp, map(float, pred["main_xy"].tolist()))
    cx,cy = C_LOOK[int(torch.argmax(pred["c_dir_logits"]))]
    l,r   = clamp(pred["L_val"].item(),0), clamp(pred["R_val"].item(),0)

    ctrl.tilt_analog(melee.enums.Button.BUTTON_MAIN, mx,my)
    ctrl.tilt_analog(melee.enums.Button.BUTTON_C, cx,cy)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_L,l)
    ctrl.press_shoulder(melee.enums.Button.BUTTON_R,r)

    buttons=[
        melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B,
        melee.enums.Button.BUTTON_X, melee.enums.Button.BUTTON_Y,
        melee.enums.Button.BUTTON_Z, melee.enums.Button.BUTTON_L,
        melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_START,
        melee.enums.Button.BUTTON_D_UP, melee.enums.Button.BUTTON_D_DOWN,
        melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,
    ]
    for p,b in zip(pred["btn_probs"],buttons):
        (ctrl.press_button if p.item()>th else ctrl.release_button)(b)

# ══════════════════════════════════════════════
# 5) Dolphin loop
# ══════════════════════════════════════════════
def sigint(sig,_):
    for c in ctrls.values(): c.disconnect()
    console.stop(); sys.exit(0)

if __name__=="__main__":
    DOLPHIN_APP = "/Users/erick/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app"
    ISO_PATH    = "/Users/erick/Downloads/melee.iso"

    console = melee.Console(path=DOLPHIN_APP, slippi_address="127.0.0.1", fullscreen=False)
    ports=[1,2]
    ctrls={p: melee.Controller(console,p) for p in ports}

    signal.signal(signal.SIGINT, sigint)
    console.run(iso_path=ISO_PATH)
    if not console.connect(): sys.exit("Dolphin connect fail")
    for c in ctrls.values():
        if not c.connect(): sys.exit("Pad connect fail")

    rows=deque(maxlen=cfg.max_seq_len)

    while True:
        gs=console.step()
        if gs is None: continue

        # menu helper (unchanged) ............................
        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            c0,c1=ctrls[ports[0]],ctrls[ports[1]]
            melee.MenuHelper().menu_helper_simple(gs,c0,melee.Character.FALCO,
                                                 melee.Stage.POKEMON_STADIUM,
                                                 cpu_level=0,autostart=0)
            melee.MenuHelper().menu_helper_simple(gs,c1,melee.Character.FALCO,
                                                 melee.Stage.POKEMON_STADIUM,
                                                 cpu_level=1,autostart=1)
            continue

        # build row ..........................................
        row: Dict[str,Any]={"stage":gs.stage.name if gs.stage else "",
                            "frame":gs.frame,"distance":gs.distance,"startAt":gs.startAt}

        for idx,(port,ps) in enumerate(gs.players.items()):
            pref="self_" if idx==0 else "opp_"
            row.update({
                f"{pref}port":port,
                f"{pref}character":ps.character.name,
                f"{pref}action":ps.action.name,
                f"{pref}action_frame":ps.action_frame,
                f"{pref}costume":ps.costume,
                f"{pref}main_x":ps.controller_state.main_stick[0],
                f"{pref}main_y":ps.controller_state.main_stick[1],
                f"{pref}c_x":ps.controller_state.c_stick[0],
                f"{pref}c_y":ps.controller_state.c_stick[1],
                f"{pref}l_shldr":ps.controller_state.l_shoulder,
                f"{pref}r_shldr":ps.controller_state.r_shoulder,
                f"{pref}percent":float(ps.percent),
                f"{pref}pos_x":float(ps.position.x),
                f"{pref}pos_y":float(ps.position.y),
            })
            for b,s in ps.controller_state.button.items():
                row[f"{pref}btn_{b.name}"]=int(s)

            nana,np=ps.nana,f"{pref}nana_"
            if nana:
                row[f"{np}character"]=nana.character.name
                row[f"{np}action"]=nana.action.name
                row[f"{np}action_frame"]=nana.action_frame
            else:
                row[f"{np}character"]=row[f"{np}action"]=""
                row[f"{np}action_frame"]=-1

        for j in range(8):
            pp=f"proj{j}_"
            if j<len(gs.projectiles):
                p=gs.projectiles[j]
                row.update({
                    f"{pp}owner":p.owner,f"{pp}type":p.type.name,
                    f"{pp}subtype":p.subtype,f"{pp}pos_x":float(p.position.x),
                    f"{pp}pos_y":float(p.position.y),f"{pp}speed_x":float(p.speed.x),
                    f"{pp}speed_y":float(p.speed.y),f"{pp}frame":p.frame})
            else:
                row.update({f"{pp}{fld}":(-1 if fld in ("owner","subtype","frame") else 0.0)
                            for fld in ("owner","type","subtype","pos_x","pos_y",
                                         "speed_x","speed_y","frame")})
        maybe_print_bad_row(row, gs.frame)
        rows.append(row)

        if len(rows)==cfg.max_seq_len:
            pred=infer(list(rows))
            pad=ctrls[ports[0]]
            pad.release_all(); press(pad,pred); pad.flush()

            if gs.frame%60==0:
                print(f"[{gs.frame}] main={pred['main_xy'].tolist()} "
                      f"cdir={int(torch.argmax(pred['c_dir_logits']))}")

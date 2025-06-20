#!/usr/bin/env python3
# inspect_window.py — Audit FRAME windows for key features and targets

import argparse
from dataset import MeleeFrameDatasetWithDelay

# your C-stick index → float mapping
C_DIR_TO_FLOAT = {
    0: (0.5, 0.5),  # neutral
    1: (0.5, 1.0),  # up
    2: (0.5, 0.0),  # down
    3: (0.0, 0.5),  # left
    4: (1.0, 0.5),  # right
}

def main():
    p = argparse.ArgumentParser(
        description="Inspect input features and prediction targets for a dataset window."
    )
    p.add_argument("-i", "--index", type=int, default=0,
                   help="Sample index (0-based)")
    p.add_argument("-d", "--data-dir", default="./data",
                   help="Path to parquet directory")
    p.add_argument("-s", "--seq-len", type=int, default=60,
                   help="Sequence length (must match train)")
    p.add_argument("-r", "--delay", type=int, default=1,
                   help="Reaction delay (must match train)")
    args = p.parse_args()

    # load dataset
    ds = MeleeFrameDatasetWithDelay(
        parquet_dir=args.data_dir,
        sequence_length=args.seq_len,
        reaction_delay=args.delay,
    )

    # pull one window
    state, target = ds[args.index]

    # core inputs
    num     = state["numeric"]          # [T, …]
    self_n  = state["self_numeric"]     # [T, …]
    opp_n   = state["opp_numeric"]      # [T, …]
    sc      = state["self_character"]   # [T]
    oc      = state["opp_character"]    # [T]
    sa      = state["self_action"]      # [T]
    oa      = state["opp_action"]       # [T]
    analog  = state["self_analog"]      # [T,4]: main_x, main_y, l, r
    cdir    = state["self_c_dir"]       # [T]
    btns    = state["self_buttons"].float()  # [T,12]

    T = num.shape[0]
    # unpack
    frames = [int(f) for f in num[:,1].tolist()]
    dist   = num[:,0].tolist()
    sx, sy = self_n[:,0].tolist(), self_n[:,1].tolist()
    ox, oy = opp_n[:,0].tolist(), opp_n[:,1].tolist()
    ss, os_ = self_n[:,3].tolist(), opp_n[:,3].tolist()
    ax, ay = analog[:,0].tolist(), analog[:,1].tolist()
    al, ar = analog[:,2].tolist(), analog[:,3].tolist()
    # map categorical → floats
    cfx = [C_DIR_TO_FLOAT[d][0] for d in cdir.tolist()]
    cfy = [C_DIR_TO_FLOAT[d][1] for d in cdir.tolist()]
    b   = btns.tolist()

    # print header
    print(
        "T frame self_char opp_char self_act opp_act "
        "self_x self_y opp_x opp_y self_stock opp_stock distance "
        "analog_x analog_y analog_L analog_R C_dir C_x   C_y   buttons"
    )

    # per-frame rows
    for t in range(T):
        print(
            f"{t:2d} "
            f"{frames[t]:5d} {sc[t]:9d} {oc[t]:8d} {sa[t]:8d} {oa[t]:8d} "
            f"{sx[t]:7.3f} {sy[t]:7.3f} {ox[t]:7.3f} {oy[t]:7.3f} "
            f"{ss[t]:10.1f} {os_[t]:10.1f} {dist[t]:9.3f} "
            f"{ax[t]:8.3f} {ay[t]:8.3f} {al[t]:8.3f} {ar[t]:8.3f} "
            f"{cdir[t]:2d} {cfx[t]:6.3f} {cfy[t]:6.3f} {b[t]}"
        )

    # prediction targets at frame = last_input_frame + delay
    tf = frames[-1] + args.delay
    print(f"\nTargets (frame {tf}):")
    print(f"  main_x  = {target['main_x'].item():.3f}")
    print(f"  main_y  = {target['main_y'].item():.3f}")
    print(f"  l_shldr = {target['l_shldr'].item():.3f}")
    print(f"  r_shldr = {target['r_shldr'].item():.3f}")
    cv = target['c_dir'].tolist(); ci = cv.index(max(cv))
    print(f"  c_dir   = {cv} (idx {ci})")
    print(f"  btns    = {target['btns'].tolist()}")

if __name__ == "__main__":
    main()

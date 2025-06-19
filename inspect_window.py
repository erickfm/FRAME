#!/usr/bin/env python3
# inspect_window.py — Audit FRAME windows for key features: positions, stocks, distance, self analog, C-stick, buttons and targets

import argparse
from dataset import MeleeFrameDatasetWithDelay

def main():
    p = argparse.ArgumentParser(
        description=(
            "Inspect input features (frame, chars, actions, positions, stocks, distance, "
            "self analog, C-stick, buttons) and prediction targets for a dataset window."
        )
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

    # get state window and target
    state, target = ds[args.index]

    # extract core tensors
    num = state["numeric"]          # [T,20]
    self_num = state["self_numeric"]  # [T,22]
    opp_num  = state["opp_numeric"]   # [T,22]
    sc = state["self_character"]
    oc = state["opp_character"]
    sa = state["self_action"]
    oa = state["opp_action"]
    analog = state["self_analog"]     # [T,4]
    cstick = state["self_c_dir"]      # [T]
    btns = state["self_buttons"].float()  # [T,12]

    # build lists
    T = num.shape[0]
    frames = [int(f) for f in num[:,1].tolist()]
    dist   = num[:,0].tolist()
    sx = self_num[:,0].tolist(); sy = self_num[:,1].tolist()
    ox = opp_num[:,0].tolist(); oy = opp_num[:,1].tolist()
    ss = self_num[:,3].tolist(); os_ = opp_num[:,3].tolist()
    ax = analog[:,0].tolist(); ay = analog[:,1].tolist()
    al = analog[:,2].tolist(); ar = analog[:,3].tolist()
    cd = cstick.tolist()
    b  = btns.tolist()  # list of 12-length lists

    # header
    header = (
        "T frame self_char opp_char self_act opp_act self_x self_y opp_x opp_y "
        "self_stock opp_stock distance analog_x analog_y analog_L analog_R C_dir buttons"
    )
    print(header)

    # rows
    for t in range(T):
        print(
            f"{t:2d} {frames[t]:5d} {sc[t]:9d} {oc[t]:8d} {sa[t]:8d} {oa[t]:8d} "
            f"{sx[t]:7.3f} {sy[t]:7.3f} {ox[t]:7.3f} {oy[t]:7.3f} "
            f"{ss[t]:10.1f} {os_[t]:10.1f} {dist[t]:9.3f} "
            f"{ax[t]:8.3f} {ay[t]:8.3f} {al[t]:8.3f} {ar[t]:8.3f} "
            f"{cd[t]:2d} {b[t]}"
        )

    # prediction targets
    tf = frames[-1] + args.delay
    print(f"\nTargets (frame {tf}):")
    print(f"  main_x  = {target['main_x'].item():.3f}")
    print(f"  main_y  = {target['main_y'].item():.3f}")
    print(f"  l_shldr = {target['l_shldr'].item():.3f}")
    print(f"  r_shldr = {target['r_shldr'].item():.3f}")
    cv = target['c_dir'].tolist(); ci = cv.index(max(cv))
    print(f"  c_dir   = {cv} (idx {ci})")
    bv = target['btns'].tolist()
    print(f"  btns    = {bv}")

if __name__ == "__main__":
    main()

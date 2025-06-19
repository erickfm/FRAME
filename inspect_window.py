#!/usr/bin/env python3
# inspect_window.py — Audit FRAME sample windows for key features

import argparse
from dataset import MeleeFrameDatasetWithDelay

def main():
    p = argparse.ArgumentParser(
        description="Inspect key features (frame, chars, actions, x/y, stocks, distance) for a sample window."
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

    state, _ = ds[args.index]

    # extract tensors
    global_num = state["numeric"]       # [T, 20]
    self_num   = state["self_numeric"]  # [T, 22]
    opp_num    = state["opp_numeric"]   # [T, 22]
    self_char  = state["self_character"]
    opp_char   = state["opp_character"]
    self_act   = state["self_action"]
    opp_act    = state["opp_action"]

    # deduced indices (cast frames to int to avoid format errors)
    frames     = [int(x) for x in global_num[:, 1].tolist()]
    distance   = global_num[:, 0].tolist()
    self_x     = self_num[:, 0].tolist()
    self_y     = self_num[:, 1].tolist()
    opp_x      = opp_num[:, 0].tolist()
    opp_y      = opp_num[:, 1].tolist()
    self_stock = self_num[:, 3].tolist()
    opp_stock  = opp_num[:, 3].tolist()

    # print header
    header = (
        "T  frame  self_char  opp_char  self_act  opp_act  "
        "self_x  self_y  opp_x  opp_y  self_stock  opp_stock  distance"
    )
    print(header)

    # print each timestep
    for t in range(len(frames)):
        print(
            f"{t:2d} {frames[t]:6d} {self_char[t]:11d} {opp_char[t]:9d} "
            f"{self_act[t]:9d} {opp_act[t]:8d} "
            f"{self_x[t]:7.3f} {self_y[t]:7.3f} "
            f"{opp_x[t]:7.3f} {opp_y[t]:7.3f} "
            f"{self_stock[t]:10.1f} {opp_stock[t]:10.1f} "
            f"{distance[t]:8.3f}"
        )

if __name__ == "__main__":
    main()

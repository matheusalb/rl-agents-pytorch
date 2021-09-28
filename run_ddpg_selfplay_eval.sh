#!/bin/bash
for i in 0; do
  python evaluation.py  --checkpoint_eval ./selfplay_evaluation/goalkeeper/gk_policy/checkpoint_gk002000000.pth --path_players ./selfplay_evaluation/goalkeeper/atk_policies  --cuda -n "DDPG_VSSMAEVAL-v2_0 $i" -e VSSSelfPlay-v0
done

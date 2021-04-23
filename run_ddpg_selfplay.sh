#!/bin/bash
for i in 0 1 2 3 4; do
  python train_ddpg_selfplay.py --cuda -n "DDPG_VSSSelfPlayAtkGk-v0_0$i" -e VSSSelfPlay-v0
done

#!/bin/bash
for i in 0 1 2 3 4; do
  python train_ddpg_gk_3v3_fira.py --cuda -n "DDPG_VSSGK3V3_RSIMUPDATED-v0_0$i" -e VSSGK3V3-v0 --checkpoint models/gk_ft2.pth
done

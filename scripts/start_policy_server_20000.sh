#!/bin/bash
# 启动策略服务器，使用20000步的checkpoint

# 使用20000步的checkpoint
uv run scripts/serve_policy1.py \
    policy:local-checkpoint \
    --policy.path checkpoints/pi05_pick_blue_bottle_libero_downsample4x/downsample4x_right_arm_finetune_30k/20000 \
    --policy.config pi05_pick_blue_bottle_libero_downsample4x \
    --port 8000



















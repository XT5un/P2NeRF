#!/bin/bash

# Don't use multiple GPUs for training
export CUDA_VISIBLE_DEVICES=0

scans=( office0 office1 office2 office3 office4 room0 room1 )

for scan_name in ${scans[@]}
    do
    python train.py \
        --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
        --gin_bindings "Config.replica_scene = '$scan_name'" \
        --gin_bindings "Config.expname = 'replica-$scan_name-full-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/replica/$scan_name/p2nerf/full'" \
        --gin_bindings "Config.project = 'replica'"

    python eval.py \
        --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
        --gin_bindings "Config.replica_scene = '$scan_name'" \
        --gin_bindings "Config.expname = 'replica-$scan_name-full-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/replica/$scan_name/p2nerf/full'" \
        --gin_bindings "Config.project = 'replica'"
    done

# # office0 office1
# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office0'" \
#     --gin_bindings "Config.expname = 'replica-office0-full-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office0/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office0'" \
#     --gin_bindings "Config.expname = 'replica-office0-full-eval'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office0/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office1'" \
#     --gin_bindings "Config.expname = 'replica-office1-full-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office1/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=0 python eval.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office1'" \
#     --gin_bindings "Config.expname = 'replica-office1-full-eval'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office1/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'"

# # office2 office3
# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office2'" \
#     --gin_bindings "Config.expname = 'replica-office2-full-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office2/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office2'" \
#     --gin_bindings "Config.expname = 'replica-office2-full-eval'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office2/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office3'" \
#     --gin_bindings "Config.expname = 'replica-office3-full-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office3/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=1 python eval.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office3'" \
#     --gin_bindings "Config.expname = 'replica-office3-full-eval'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office3/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'"

# # office4 room0
# CUDA_VISIBLE_DEVICES=2 python train.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office4'" \
#     --gin_bindings "Config.expname = 'replica-office4-full-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office4/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=2 python eval.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'office4'" \
#     --gin_bindings "Config.expname = 'replica-office4-full-eval'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/office4/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=2 python train.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'room0'" \
#     --gin_bindings "Config.expname = 'replica-room0-full-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/room0/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=2 python eval.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'room0'" \
#     --gin_bindings "Config.expname = 'replica-room0-full-eval'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/room0/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'"

# # room1 room2
# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'room1'" \
#     --gin_bindings "Config.expname = 'replica-room1-full-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/room1/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=3 python eval.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'room1'" \
#     --gin_bindings "Config.expname = 'replica-room1-full-eval'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/room1/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'room2'" \
#     --gin_bindings "Config.expname = 'replica-room2-full-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/room2/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'" && \
# CUDA_VISIBLE_DEVICES=3 python eval.py \
#     --gin_configs configs/p2nerf/replica_p2nerf_full.gin \
#     --gin_bindings "Config.replica_scene = 'room2'" \
#     --gin_bindings "Config.expname = 'replica-room2-full-eval'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/replica/room2/p2nerf/full'" \
#     --gin_bindings "Config.project = 'replica'"
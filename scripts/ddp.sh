#!/bin/bash

# Don't use multiple GPUs for training
export CUDA_VISIBLE_DEVICES=0

scans=( scene0710_00 scene0758_00 scene0781_00 )

for scan_name in ${scans[@]}
    do
    python train.py \
        --gin_configs configs/p2nerf/ddp_p2nerf_full.gin \
        --gin_bindings "Config.replica_scene = '$scan_name'" \
        --gin_bindings "Config.expname = 'DDP-scan$scan_name-full-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/DDP/p2nerf/scan$scan_name/full'" \
        --gin_bindings "Config.project = 'DDP'"
    done

for scan_name in ${scans[@]}
    do
    python eval.py \
        --gin_configs configs/p2nerf/ddp_p2nerf_full.gin \
        --gin_bindings "Config.replica_scene = 'scan$scan_name'" \
        --gin_bindings "Config.expname = 'DDP-scan$scan_name-full-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/DDP/p2nerf/scan$scan_name/full'" \
        --gin_bindings "Config.project = 'DDP'"
    done

#!/usr/bin/env bash
ARCH=$1
export PATH=~/.local/bin/:$PATH

GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
srun -n1 -p <your partition> --gres=gpu:1 --ntasks-per-node=1 --mpi=pmi2 \
    --job-name=rc --cpus-per-task=5 --preempt \
python -u examples/test.py -a ${ARCH} --resume $2 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
    --query-list $3 \
    --gallery-list $4 \
    --validate_feat fusion --config ./scripts/config_ablation5.yaml \
    --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 --test_feat_type f \
    --root $5 \
    --test_task_type $6
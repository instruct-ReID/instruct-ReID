#!/bin/sh
ARCH=$1
NUM_GPUs=$2
DESC=$3
SEED=0

if [[ $# -eq 4 ]]; then
  port=${4}
else
  port=23456
fi

ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH


GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
srun -n${NUM_GPUs} -p <your partition> --gres=gpu:${NUM_GPUs} --ntasks-per-node=${NUM_GPUs} --mpi=pmi2 \
    --job-name=rc --cpus-per-task=5 --preempt \
python -u examples/train.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.001 --alpha 3.0 --optimizer SGD --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config ./scripts/market.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None \
  --query-list <your query dataset txt path> \
  --gallery-list <your gallery dataset txt path> \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your dataset root>
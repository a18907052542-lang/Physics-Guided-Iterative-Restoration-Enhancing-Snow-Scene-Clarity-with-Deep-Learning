#!/usr/bin/env bash
# Run the full PAID-SnowNet experimental pipeline.
#
# Requires:
#   - DATA_ROOT environment variable pointing at Snow100K
#   - CKPT  (optional) path to a trained checkpoint
#
# Usage:
#   DATA_ROOT=/path/to/Snow100K CKPT=./checkpoints/best.pth bash scripts/run_all_experiments.sh

set -e

if [ -z "$DATA_ROOT" ]; then
    echo "ERROR: please set DATA_ROOT to your Snow100K directory."
    exit 1
fi

CKPT_ARG=""
if [ -n "$CKPT" ]; then
    CKPT_ARG="--checkpoint $CKPT"
fi

mkdir -p results figs

echo "=== 1. main test ==="
python test.py --data_root "$DATA_ROOT" $CKPT_ARG --save_dir ./results

echo "=== 2. ablation studies (Tables 3,4,5,6,12,13,14,15) ==="
python experiments/ablation_study.py --data_root "$DATA_ROOT" $CKPT_ARG \
    --save_dir ./results/ablation --max_batches 50

echo "=== 3. loss-function ablation (Table 16) ==="
python experiments/ablation_loss.py --data_root "$DATA_ROOT" \
    --save_dir ./results/ablation --epochs 5 \
    ${CKPT:+--init_checkpoint "$CKPT"}

echo "=== 4. scene-type evaluation (Table 11) ==="
python experiments/scene_type_eval.py --data_root "$DATA_ROOT" $CKPT_ARG \
    --save_dir ./results/scene_type

echo "=== 5. robustness evaluation (Tables 17, 18, 19) ==="
python experiments/robustness_eval.py --data_root "$DATA_ROOT" $CKPT_ARG \
    --save_dir ./results/robustness --max_batches 50

echo "=== 6. downstream evaluation ==="
echo "   (set DOWNSTREAM_MODE=real to use ultralytics YOLOv8 + DeepLabV3 + ResNet-50)"
python experiments/downstream_eval.py --data_root "$DATA_ROOT" $CKPT_ARG \
    --save_dir ./results/downstream \
    --mode "${DOWNSTREAM_MODE:-proxy}"

echo "=== 7. runtime analysis (Table 21) ==="
python experiments/runtime_analysis.py $CKPT_ARG --save_dir ./results/runtime

echo "=== 8. convergence analysis (Theorem 1 + Fig. 5/8) ==="
python analysis/convergence_analysis.py --data_root "$DATA_ROOT" $CKPT_ARG \
    --save_dir ./results/convergence

echo "=== 9. visualization ==="
python visualization/plot_physics_decomposition.py \
    --physics_dir ./results/physics --output ./figs/fig3.png || true
python visualization/plot_convergence.py \
    --results ./results/convergence/convergence_results.json \
    --output_fig5 ./figs/fig5.png --output_fig8 ./figs/fig8.png
python visualization/plot_ablation_modules.py \
    --results ./results/ablation/ablation_results.json --output ./figs/fig7.png
python visualization/plot_robustness.py \
    --results ./results/robustness/robustness_results.json --output ./figs/fig9.png
python visualization/plot_pareto.py \
    --results ./results/ablation/ablation_results.json \
    ${COMPETITOR_DATA:+--competitor_data "$COMPETITOR_DATA"} \
    --output ./figs/fig11.png
python visualization/plot_downstream_correlation.py \
    --results ./results/downstream/downstream_results.json --output ./figs/fig12.png

echo "Done. Results -> ./results, figures -> ./figs"

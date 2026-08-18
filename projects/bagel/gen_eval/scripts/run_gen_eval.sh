#!/bin/bash
# Generation evaluation pipeline for BAGEL-7B-MoT on DPG-Bench / Qwen-Image-Bench.
# Usage:
#   bash run_gen_eval.sh base dpg          # base model, DPG-Bench only
#   bash run_gen_eval.sh base all          # base model, both benchmarks
#   bash run_gen_eval.sh lora dpg          # + t2i LoRA adapter
#   LIMIT=32 bash run_gen_eval.sh base dpg # smoke test with 32 prompts
set -uo pipefail

# mplug 评分模型缓存（目录规整后位置）
export MODELSCOPE_CACHE=/root/data/projects/bagel/cache/modelscope
export HF_HOME=/root/data/projects/bagel/hf_home

MODEL_TAG=${1:-base}      # base | lora
BENCH=${2:-dpg}           # dpg | qib | all
LIMIT=${LIMIT:-0}
GEN_ROOT=/root/data/projects/bagel/gen_eval/images
SCRIPTS=/root/data/projects/bagel/gen_eval/scripts
T2I_LORA=/root/data/projects/bagel/results/lora_overfit_t2i/checkpoints/final/lora.safetensors

if [ "$MODEL_TAG" = "lora" ]; then export LORA_PATH="$T2I_LORA"; else unset LORA_PATH; fi

run_gen() {
  local bench=$1 out_dir=$2
  mkdir -p "$out_dir"
  echo "[gen-pipe] generating $bench ($MODEL_TAG) -> $out_dir limit=$LIMIT"
  CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/generate_t2i.py" \
    --benchmark "$bench" --out_dir "$out_dir" --image_size 512 \
    --limit "$LIMIT" --shard 0 --num_shards 2 \
    > "$out_dir/gen_shard0.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/generate_t2i.py" \
    --benchmark "$bench" --out_dir "$out_dir" --image_size 512 \
    --limit "$LIMIT" --shard 1 --num_shards 2 \
    > "$out_dir/gen_shard1.log" 2>&1 &
  local p1=$!
  wait $p0; local rc0=$?
  wait $p1; local rc1=$?
  echo "[gen-pipe] $bench generation done rc=($rc0,$rc1)"
  # CLIPScore (local rule-based)
  python "$SCRIPTS/clip_score_eval.py" --image_dir "$out_dir" --benchmark "$bench" \
    > "$out_dir/clip_eval.log" 2>&1
  cat "$out_dir/clip_eval.log" | tail -1
}

score_dpg() {
  local out_dir=$1
  echo "[gen-pipe] official DPG scoring (mPLUG VQA) on $out_dir"
  CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/score_dpg_mplug.py" \
    --image_dir "$out_dir" --resolution 512 \
    > "$out_dir/dpg_score.log" 2>&1
  tail -6 "$out_dir/dpg_results.txt" 2>/dev/null
}

for bench in $( [ "$BENCH" = "all" ] && echo "dpg qib" || echo "$BENCH" ); do
  out_dir="$GEN_ROOT/${MODEL_TAG}_${bench}"
  run_gen "$bench" "$out_dir"
  [ "$bench" = "dpg" ] && score_dpg "$out_dir"
done
echo "[gen-pipe] ALL DONE tag=$MODEL_TAG bench=$BENCH"

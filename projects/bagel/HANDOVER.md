# Bagel 目标机交接文档（下载与环境搭建）

> 适用场景：代码与训练/评测产出已通过 rsync 传到目标机（腾讯云 A800×2）的 `/root/bagel`，
> 本文列出需要在目标机**下载**的大文件及其相对目录位置。
> 目标根目录约定：`/root/bagel`，下文所有路径均**相对于该目录**。
> 磁盘注意：`/` 仅剩 ~60G，P0/P1 下载项（约 31G）放 `/root/bagel` 没问题；
> P2 大数据集（41G/42G）请改放 `/tank`（14T 可用）并相应调整软链或脚本路径。

## 0. 带宽实测参考（2026-08-18）

| 链路 | 实测速度 |
|---|---|
| 目标机 → huggingface.co 官方 | ~307 B/s（不可用，必须走镜像） |
| 目标机 → hf-mirror.com | ~1.4 MB/s |
| 目标机 → modelscope.cn | 可达（未测速，建议优先测） |

- 所有 HF 下载必须设置 `export HF_ENDPOINT=https://hf-mirror.com`
- 建议每个下载用 `setsid nohup ... > xxx.log 2>&1 &` 后台跑 + 重试循环

## 1. BAGEL-7B-MoT 基座权重（28G）→ `models/BAGEL-7B-MoT/`

来源：**ModelScope** `bytedance-Seed/BAGEL-7B-MoT`（本机即从 ModelScope 下载，目录里有 configuration.json 为证）

```bash
pip install modelscope
modelscope download --model bytedance-Seed/BAGEL-7B-MoT \
  --local_dir /root/bagel/models/BAGEL-7B-MoT
```

校验：`ema.safetensors` ≈ 27G、`ae.safetensors` ≈ 320M、有 `config.json`。
用途：LoRA 训练基座（train 脚本 `--model-path`）、推理。

## 2. Qwen-Image-Bench 提示词（176M）→ `gen_eval/prompts/`

来源：HF dataset `Qwen/Qwen-Image-Bench`（走 hf-mirror）。
仓库里已有下载脚本，改一下 local_dir 即可：`gen_eval/scripts/dl_qib.py`

```bash
export HF_ENDPOINT=https://hf-mirror.com
python gen_eval/scripts/dl_qib.py   # 记得把脚本内 local_dir 改为 /root/bagel/gen_eval/prompts
```

产物：`gen_eval/prompts/qwen_image_bench_hf_v0518.jsonl`（176M，该文件已被 gitignore，必须下载）。
qib_prompts.jsonl 已随仓库推送，无需下载。

## 3. MME-RealWorld-Base64 数据集（blob 实际 ~41G）→ `hf_home/`（建议放 /tank）

来源：HF dataset `yifanzhang114/MME-RealWorld-Base64`。
必须用 `HF_HOME` 指到项目内，保证与本机缓存布局一致（/ 空间不够时先下到 /tank 再软链）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/bagel/hf_home    # 或 /tank/bagel_data/hf_home 后软链到 /root/bagel/hf_home
huggingface-cli download --repo-type dataset yifanzhang114/MME-RealWorld-Base64
```

校验：`hf_home/hub/datasets--yifanzhang114--MME-RealWorld-Base64/blobs/` 总量 ≈ 41G。
注意：主文件 MME-RealWorld.tsv 单文件 33G，下载脚本要有断点续传/重试（huggingface-cli 自带续传）。

## 4. mplug 评分模型（2.8G）→ `cache/modelscope/`

来源：ModelScope `damo/mplug_visual-question-answering_coco_large_en`。
gen_eval/scripts/run_gen_eval.sh 已导出 `MODELSCOPE_CACHE=<项目>/cache/modelscope`，
目标机上把该路径改为 /root/bagel 新路径后直接跑即可，首次运行会自动下载：

```bash
export MODELSCOPE_CACHE=/root/bagel/cache/modelscope
modelscope download --model damo/mplug_visual-question-answering_coco_large_en \
  --local_dir /root/bagel/cache/modelscope/models/damo--mplug_visual-question-answering_coco_large_en/snapshots/master
```

## 5. LMUData 评测数据集（42G）→ `LMUData/`（仅跑 VLMEvalKit 基准时需要，建议放 /tank）

由 VLMEvalKit 自动下载。先克隆第三方仓库（已 gitignore，不在本仓库内）：

```bash
git clone https://github.com/open-compass/VLMEvalKit.git /root/bagel/VLMEvalKit
```

跑评测时设置 `LMUData` 目录并按 VLMEvalKit 文档执行，数据集会自动落到
`/root/bagel/LMUData/`（本机口径；空间不足时放 /tank 再软链）。

## 6. flash-attn（通常无需处理）

rsync 已带来现成 wheel：`fa_build/flash_attn-2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
匹配条件：Python 3.10 / torch 2.6 / CUDA 12 / cxx11abi=FALSE。目标机环境一致则直接：

```bash
pip install fa_build/flash_attn-2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 7. 下载优先级建议

| 优先级 | 内容 | 大小 | 理由 |
|---|---|---|---|
| P0 | BAGEL-7B-MoT | 28G | 训练/推理必需 |
| P0 | qwen_image_bench jsonl | 176M | gen-eval 必需且仓库没有 |
| P1 | mplug | 2.8G | gen-eval 打分必需 |
| P2 | MME-RealWorld | 41G | 仅 MME-RW 评测需要 |
| P2 | LMUData | 42G | 仅 VLMEvalKit 基准需要 |

## 8. 环境快照（本机口径，供对齐）

- Python 3.10.12，torch 2.6（cu12，cxx11abi=FALSE），flash-attn 2.7.4.post1
- 评测/训练脚本内硬编码绝对路径均为 `/root/data/projects/bagel/...`，
  目标机部署后需全局替换为 `/root/bagel/...`：
  `grep -rl "/root/data/projects/bagel" scripts gen_eval Bagel | xargs sed -i "s|/root/data/projects/bagel|/root/bagel|g"`
- 本文件随仓库推送（HANDOVER.md），rsync 后直接可读。

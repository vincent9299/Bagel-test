# Bagel 目标机交接文档（下载与环境搭建）

> 适用场景：代码与训练/评测产出已通过 rsync 传到目标机（腾讯云 A800×2）的 `/tank/bagel`，
> 本文列出需要在目标机**下载**的大文件及其落盘路径。
> 目标根目录约定：`/tank/bagel`（/tank 有 14T，所有下载项都直接放这里，无需软链）。

## 0. 带宽实测参考（2026-08-18）

| 链路 | 实测速度 |
|---|---|
| 目标机 → huggingface.co 官方 | ~307 B/s（不可用，HF 只作为最后兜底且必须走镜像） |
| 目标机 → hf-mirror.com | ~1.4 MB/s |
| 目标机 → modelscope.cn | 可达（国内 CDN，预期最快，未测速，开工前先测） |

**下载源选择策略（铁律）：能走 ModelScope 的一律先走 ModelScope，ModelScope 没有的再走 hf-mirror，HF 官方域名禁用。**

- ModelScope 下载：`pip install modelscope` 后用 `modelscope download`（数据集加 `--dataset` 参数）
- HF 兜底下载必须设置 `export HF_ENDPOINT=https://hf-mirror.com`
- 建议每个下载用 `setsid nohup ... > xxx.log 2>&1 &` 后台跑 + 重试循环

## 0.1 下载落盘路径总表（绝对路径，直接下到对应位置）

| 优先级 | 内容 | 来源 | 精确落盘路径（目标机绝对路径） | 大小 |
|---|---|---|---|---|
| P0 | BAGEL-7B-MoT 权重 | **ModelScope** `bytedance-Seed/BAGEL-7B-MoT` | `/tank/bagel/models/BAGEL-7B-MoT/` | 28G |
| P0 | Qwen-Image-Bench 提示词 | **ModelScope dataset** `Qwen/Qwen-Image-Bench`（官方同步镜像） | `/tank/bagel/gen_eval/prompts/qwen_image_bench_hf_v0518.jsonl`（单文件） | 176M |
| P1 | mplug 评分模型 | **ModelScope** `damo/mplug_visual-question-answering_coco_large_en` | `/tank/bagel/cache/modelscope/models/damo--mplug_visual-question-answering_coco_large_en/snapshots/master/` | 2.8G |
| P2 | MME-RealWorld-Base64 | 先在 ModelScope 搜镜像，找不到再 hf-mirror `yifanzhang114/MME-RealWorld-Base64` | `/tank/bagel/hf_home/`（用 HF_HOME 控制） | 41G |
| P2 | LMUData | VLMEvalKit 自动下载（其内部源自行配置镜像） | `/tank/bagel/LMUData/` | 42G |
| P2 | VLMEvalKit 仓库 | GitHub `open-compass/VLMEvalKit` | `/tank/bagel/VLMEvalKit/`（git clone） | 20M |
| P2 | ELLA 第三方仓库 | GitHub `TencentARC/ELLA` | `/tank/bagel/gen_eval/ELLA/`（git clone，本机原本也是克隆使用） | 13M |

> /tank 空间充足（14T），全部下载项合计约 114G，直接落盘即可。

## 1. BAGEL-7B-MoT 基座权重（28G）→ `models/BAGEL-7B-MoT/`

来源：**ModelScope** `bytedance-Seed/BAGEL-7B-MoT`（本机即从 ModelScope 下载，目录里有 configuration.json 为证）

```bash
pip install modelscope
modelscope download --model bytedance-Seed/BAGEL-7B-MoT \
  --local_dir /tank/bagel/models/BAGEL-7B-MoT
```

校验：`ema.safetensors` ≈ 27G、`ae.safetensors` ≈ 320M、有 `config.json`。
用途：LoRA 训练基座（train 脚本 `--model-path`）、推理。

## 2. Qwen-Image-Bench 提示词（176M）→ `gen_eval/prompts/`

来源：**ModelScope dataset `Qwen/Qwen-Image-Bench`**（官方宣布与 HF 同步开源，优先走这里）：

```bash
modelscope download --dataset Qwen/Qwen-Image-Bench \
  qwen_image_bench_hf_v0518.jsonl \
  --local_dir /tank/bagel/gen_eval/prompts
```

若 ModelScope 上找不到该文件，兜底走 hf-mirror（仓库里已有脚本 `gen_eval/scripts/dl_qib.py`，改 local_dir 为 `/tank/bagel/gen_eval/prompts`）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python gen_eval/scripts/dl_qib.py
```

产物：`gen_eval/prompts/qwen_image_bench_hf_v0518.jsonl`（176M，该文件已被 gitignore，必须下载）。
qib_prompts.jsonl 已随仓库推送，无需下载。

## 3. MME-RealWorld-Base64 数据集（blob 实际 ~41G）→ `hf_home/`

来源：HF dataset `yifanzhang114/MME-RealWorld-Base64`。开工前先在 ModelScope 搜 `MME-RealWorld`，
如有 Base64 版镜像则优先用 `modelscope download --dataset`；找不到再走 hf-mirror。
走 HF 时必须用 `HF_HOME` 控制落盘位置，保证与本机缓存布局一致：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tank/bagel/hf_home
huggingface-cli download --repo-type dataset yifanzhang114/MME-RealWorld-Base64
```

注意：若从 ModelScope 下载，下完后需手工摆成上述 HF 缓存布局（datasets--yifanzhang114--MME-RealWorld-Base64/snapshots+blobs），
或者把评测脚本的数据读取路径改为 ModelScope 落盘路径（本机 gen_eval/scripts/dl_mme_rw.py 可作参考）。

校验：`/tank/bagel/hf_home/hub/datasets--yifanzhang114--MME-RealWorld-Base64/blobs/` 总量 ≈ 41G。
注意：主文件 MME-RealWorld.tsv 单文件 33G，下载脚本要有断点续传/重试（huggingface-cli 自带续传）。

## 4. mplug 评分模型（2.8G）→ `cache/modelscope/`

来源：ModelScope `damo/mplug_visual-question-answering_coco_large_en`。
gen_eval/scripts/run_gen_eval.sh 已导出 `MODELSCOPE_CACHE=<项目>/cache/modelscope`，
目标机上把该路径改为 /tank/bagel 新路径后直接跑即可，首次运行会自动下载：

```bash
export MODELSCOPE_CACHE=/tank/bagel/cache/modelscope
modelscope download --model damo/mplug_visual-question-answering_coco_large_en \
  --local_dir /tank/bagel/cache/modelscope/models/damo--mplug_visual-question-answering_coco_large_en/snapshots/master
```

## 5. LMUData 评测数据集（42G）→ `LMUData/`（仅跑 VLMEvalKit 基准时需要）

由 VLMEvalKit 自动下载。先克隆第三方仓库（已 gitignore，不在本仓库内）：

```bash
git clone https://github.com/open-compass/VLMEvalKit.git /tank/bagel/VLMEvalKit
```

之后按 VLMEvalKit 文档跑评测，数据集会自动落到
`/tank/bagel/LMUData/`（与本机口径一致）。另需克隆 ELLA（gen-eval 依赖，本机为 gitignore 的第三方仓库）：

```bash
git clone https://github.com/TencentARC/ELLA.git /tank/bagel/gen_eval/ELLA
```

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
  目标机部署后需全局替换为 `/tank/bagel/...`：
  `grep -rl "/root/data/projects/bagel" scripts gen_eval Bagel | xargs sed -i "s|/root/data/projects/bagel|/tank/bagel|g"`
- 本文件随仓库推送（HANDOVER.md），rsync 后直接可读。

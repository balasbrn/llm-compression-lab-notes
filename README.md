# LLM Compression Lab Notes

**How much accuracy can I retain while making a model smaller and faster?**

My experiment notes from NVIDIA's *The Art of Compressing LLMs*. The first milestone is an accuracy baseline for the original Nemotron model.

## Lab environment

| Item | Observed value |
| :--- | :--- |
| GPU | 1 × NVIDIA H100 NVL |
| GPU memory visible to the lab | 95,830 MiB (93.6 GiB) |
| NVIDIA driver | 570.86.15 |
| CUDA version reported by `nvidia-smi` | 13.0 |

Hardware snapshot taken in the quantization session on November 9, 2026. The CUDA value comes from the driver display. Runtime and toolkit versions may differ.

## Baseline results

![5-shot baseline accuracy for ARC-Easy, PIQA, and WinoGrande](assets/baseline-accuracy.svg)

| Benchmark | Accuracy | Length-normalized accuracy |
| :--- | ---: | ---: |
| ARC-Easy | 81.36% | 82.70% |
| PIQA | 77.15% | 77.75% |
| WinoGrande | 67.17% | — |

**Setup.** NVIDIA Nemotron 3 Nano 4B in BF16, evaluated with the Hugging Face backend of LM Evaluation Harness. Each task used five examples in the prompt, with batch size 32 and no example limit. The chart shows raw accuracy.

## Why prune a pretrained model?

Pruning removes selected parts of a trained model. **Depth pruning** removes entire layers. **Width pruning** removes structures inside layers, such as attention heads or feed-forward neurons.

- **The original idea.** Le Cun and colleagues estimated how much removing each weight would increase training loss, then removed weights with the smallest estimated impact. [Optimal Brain Damage (1989)](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)
- **The LLM application.** NVIDIA combined structured pruning with teacher-guided retraining to produce 8B and 4B models from Nemotron-4 15B. The study reports up to 40× fewer training tokens per smaller model compared with training from scratch. [Compact Language Models via Pruning and Knowledge Distillation (2024)](https://arxiv.org/pdf/2407.14679)

**What the lab tests.** Rerun ARC-Easy, PIQA, and WinoGrande after pruning and compare each score with its BF16 baseline. This measures the immediate accuracy change and gives a reference for evaluating how much accuracy distillation recovers.

## Pruning results

The depth-pruned checkpoint has 36 layers and 3.50B parameters, down from 3.97B (about 12% fewer parameters). The width-pruned checkpoint has 3.44B parameters (about 13% fewer than the original). All runs used BF16, five examples per task, and batch size 32.

| Benchmark | Original | Depth-pruned (Δ) | Width-pruned (Δ) | NAS (Δ) |
| :--- | ---: | ---: | ---: | ---: |
| ARC-Easy | 81.36% | 67.13% (−14.23 pp) | 71.21% (−10.15 pp) | 67.21% (−14.15 pp) |
| PIQA | 77.15% | 69.48% (−7.67 pp) | 71.44% (−5.71 pp) | 69.31% (−7.84 pp) |
| WinoGrande | 67.17% | 60.14% (−7.03 pp) | 63.38% (−3.79 pp) | 60.22% (−6.95 pp) |

The width-pruned checkpoint had fewer parameters and scored higher than the depth-pruned checkpoint on all three tasks. NAS scored within 0.17 percentage points of depth pruning on each task. Width pruning had the highest accuracy of the three pruned checkpoints on every benchmark. Changes use raw accuracy and are in percentage points relative to the original.

## Distillation results

After 40 iterations of logits distillation, the 3.44B width-pruned student improved on all three BF16 5-shot benchmarks.

| Benchmark | Original | Before distillation | After distillation | Gain |
| :--- | ---: | ---: | ---: | ---: |
| ARC-Easy | 81.36% | 71.21% | 76.26% | +5.05 pp |
| PIQA | 77.15% | 71.44% | 74.27% | +2.83 pp |
| WinoGrande | 67.17% | 63.38% | 66.22% | +2.84 pp |

Gains are in percentage points and use raw accuracy.

## FP8 quantization results

The original Nemotron checkpoint was quantized to FP8 with CNN/DailyMail calibration and served through vLLM with an FP8 KV cache.

| Benchmark | Original BF16 | Original FP8 | Change |
| :--- | ---: | ---: | ---: |
| ARC-Easy | 81.36% | 81.19% | −0.17 pp |
| PIQA | 77.15% | 76.82% | −0.33 pp |
| WinoGrande | 67.17% | 67.56% | +0.39 pp |

These are raw 5-shot accuracy scores. The FP8 run used the vLLM completions API with batch size 16. The BF16 run used the Hugging Face backend with batch size 32. The scores were close across these setups. A matched serving benchmark is needed to measure a speedup.

## Serving throughput on H100

AIPerf measured aggregate output tokens per second with vLLM. Both runs used streaming completions, 512-token synthetic inputs, a 256-token output target, and 30-second runs at each concurrency. The FP8 run used the original Nemotron quantized to FP8, with an FP8 KV cache.

| Concurrency | BF16 tokens/s | FP8 tokens/s | FP8 / BF16 |
| :--- | ---: | ---: | ---: |
| 1 | 79.66 | 96.25 | 1.21× |
| 10 | 1686.35 | 2091.72 | 1.24× |
| 30 | 3339.07 | 4221.79 | 1.26× |
| 100 | 3313.80 | 3721.74 | 1.12× |

FP8 delivered higher aggregate throughput at all four concurrencies. Both configurations reached their highest measured throughput at concurrency 30. The CSV exports also reported the following p95 latencies, in milliseconds.

| Concurrency | Request latency BF16 → FP8 | Time to first token BF16 → FP8 |
| :--- | ---: | ---: |
| 1 | 11,142.67 → 8,136.30 | 10,349.79 → 7,576.43 |
| 10 | 1,196.18 → 926.12 | 120.81 → 79.06 |
| 30 | 3,167.82 → 1,441.28 | 1,757.24 → 219.78 |
| 100 | 14,595.84 → 14,397.14 | 11,076.77 → 11,079.44 |

Concurrency 1 had only 11 BF16 and 14 FP8 requests, so a slow request can strongly affect p95. At concurrency 100, first-token p95 was essentially unchanged despite higher output throughput. The CSVs do not establish request error rates.

## Experiment roadmap

| Stage | What I want to measure | Status |
| :--- | :--- | :--- |
| BF16 baseline | Reference accuracy on three tasks | Measured |
| Pruning | Accuracy change after removing model capacity | Depth, width, and NAS measured |
| Distillation | Accuracy recovered through teacher supervision | Measured |
| Quantization | Effect of lower precision on accuracy and memory | Accuracy measured |
| Inference benchmarking | Latency and throughput under the same serving workload | Throughput and latency measured |

## Interview talking point

I measured the original model before applying compression. Using the same evaluation setup for each variant lets me compare accuracy in percentage points. A separate serving benchmark will show whether those accuracy trade-offs buy measurable improvements in latency or throughput.

## FE1 · Single-feature ablation

**Status.** Planned. Results will be added after the experiment.

**Question.** Does one input feature improve the student's ability to reproduce the teacher's decisions when the other inputs are available?

Train a baseline student with all features and a second student with one feature removed. Keep the architecture, training settings, data split, and decision threshold fixed. Evaluate both on the same held-out examples against the same teacher labels.

| Model | Missed teacher vetoes | Extra vetoes |
| :--- | ---: | ---: |
| All features | Pending | Pending |
| One feature removed | Pending | Pending |

Report counts and rates. Divide missed vetoes by teacher vetoes and extra vetoes by teacher allows. If both rates stay similar, the feature may add little under this setup. Overlapping features can still matter when removed together.

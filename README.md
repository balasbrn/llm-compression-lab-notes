# LLM Compression Lab Notes

**How much accuracy can I retain while making a model smaller and faster?**

My experiment notes from NVIDIA's *The Art of Compressing LLMs*. The first milestone is an accuracy baseline for the original Nemotron model.

## Baseline results

![5-shot baseline accuracy for ARC-Easy, PIQA, and WinoGrande](assets/baseline-accuracy.svg)

| Benchmark | Accuracy | Length-normalized accuracy |
| :--- | ---: | ---: |
| ARC-Easy | 81.36% | 82.70% |
| PIQA | 77.15% | 77.75% |
| WinoGrande | 67.17% | — |

**Setup.** NVIDIA Nemotron 3 Nano 4B in BF16, evaluated with the Hugging Face backend of LM Evaluation Harness. Each task used five examples in the prompt, with batch size 32 and no example limit. The chart shows raw accuracy.

## Experiment roadmap

| Stage | What I want to measure | Status |
| :--- | :--- | :--- |
| BF16 baseline | Reference accuracy on three tasks | Measured |
| Pruning | Accuracy change after removing model capacity | Planned |
| Distillation | Accuracy recovered through teacher supervision | Planned |
| Quantization | Effect of lower precision on accuracy and memory | Planned |
| Inference benchmarking | Latency and throughput under the same serving workload | Planned |

## Interview talking point

I measured the original model before applying compression. Using the same evaluation setup for each variant lets me compare accuracy in percentage points. A separate serving benchmark will show whether those accuracy trade-offs buy measurable improvements in latency or throughput.

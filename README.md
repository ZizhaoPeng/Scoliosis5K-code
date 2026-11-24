# Scoliosis5K

Scoliosis5K contains 4,998 corridor-walking recordings from 3,010 participants at nine clinical and school sites. Each recording pairs RGB crops with silhouettes and 25 two-dimensional pose landmarks. This repository contains the reference implementation for *Scoliosis5K: Multimodal Gait Analysis via Clip-Conditioned RGB-Silhouette Fusion*.

[Project website](https://zizhaopeng.github.io/Scoliosis5K-site/) · [Request dataset access](https://zizhaopeng.github.io/Scoliosis5K-site/#access) · [Preprint (arXiv v1)](https://arxiv.org/abs/2611.26849v1) · [Model and evaluation](REPRODUCIBILITY.md)

## Dataset access

Use the [dataset access page](https://zizhaopeng.github.io/Scoliosis5K-site/#access) to download the agreement and submit a research request. Approved researchers receive access to the encrypted, head-masked RGB and paired silhouette/pose archives. Include any request for the study fold assignments in your application.

## Quick start

Create an environment with Python 3.11–3.14. The demo uses generated arrays and runs on a CPU without downloading model weights.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m scoliosis5k.demo
python -m pytest
```

The demo prints motion-gate dimensions and paired and fused logits for one synthetic window. The tests cover the adapter, training gradients, checkpoint round-tripping, regional fitting, and input preparation.

## Model code

| Component | Entry point |
| --- | --- |
| Paired RGB and silhouette preprocessing | `scoliosis5k.preprocessing.prepare_window` |
| DINOv2 motion-conditioned Q/V adaptation | `scoliosis5k.motion.MotionConditionedLoRADenseAdapter` |
| Paired classifier and training step | `scoliosis5k.paired.build_public_motion_paired`, `make_optimizer`, `train_step` |
| Fold-centered silhouette distillation targets | `scoliosis5k.distillation.masked_token_mean`, `normalize_teacher_vectors` |
| Native-frame PE-Spatial descriptors | `scoliosis5k.pe.load_pe_spatial`, `extract_regions` |
| Participant-balanced regional fitting and score fusion | `scoliosis5k.regional.fit_regional`, `predict_regional`, `fuse_regional` |

See [model inputs, dependency setup, and evaluation](REPRODUCIBILITY.md) for the complete tensor interface.

## Study

The model reads 24-frame RGB and silhouette windows. A regional RGB classifier reads one higher-resolution frame per window.

The internal study used 1,628 recordings from 988 participants, with five participant-disjoint folds and three training seeds. Averaging over the three seeds gave 83.77% participant accuracy and 79.91% macro-F1. The final branch combination was evaluated on the same folds used to inspect the component results. The official dataset test partition was held out.

## Citation and license

The current manuscript lists Zizhao Peng, Zihan Wang, Zirui Zhou, Zheng Lv, Yan Wang, Ping Li and Fengwei An. The public preprint (v1) lists six authors. [Citation metadata](CITATION.cff) · [arXiv v1](https://arxiv.org/abs/2611.26849v1)

The source code and its software documentation are licensed under the [MIT License](LICENSE). Upstream backbones and pretrained weights retain their original licenses. Dataset access and use are governed by the data-use agreement.

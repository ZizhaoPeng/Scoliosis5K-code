# Reproducibility

## Run with upstream encoders

Install the repository as shown in the README. For the pretrained backbones, make the upstream source checkouts importable on `PYTHONPATH`. Install `torchvision`, `timm`, `einops`, and `huggingface-hub` alongside PyTorch when using PE-Spatial.

| Backbone | Source revision | Pretrained checkpoint |
| --- | --- | --- |
| DINOv2 ViT-S/14, without registers | [7764ea0](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) | [dinov2_vits14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) |
| PE-Spatial-B16-512 | [3e352cc](https://github.com/facebookresearch/perception_models/tree/3e352cca660658d4b5c90f42a7808b11469e4c66) | [PE-Spatial-B16-512.pt](https://huggingface.co/facebook/PE-Spatial-B16-512/resolve/9ea59c16f2d31b1bac766e9b54d94e78bb03ee93/PE-Spatial-B16-512.pt) |

Load them by path with `build_public_motion_paired(arm="D", seed=73111, checkpoint=...)` and `load_pe_spatial(checkpoint=...)`. The original backbone weights load strictly. The pretrained DINOv2 checkpoint initializes the frozen backbone. To obtain the adapted classifier, train its motion factors and paired head, or load an authorized matching fold-and-seed delta with `load_delta`. The regional branch also requires a fitted reference model for that fold.

## Windows and training

- Prepare paired clips with `prepare_window(rgb, silhouettes)`. Supply 24 slots in saved order, with `None` for a padded slot. RGB is uint8 HWC and silhouette is uint8 64×64, with binary values encoded as 0 and 255. The output has `aux_images` of shape `[1,24,3,224,149]`, `silhouettes` of shape `[1,24,64,64]`, and two boolean masks of shape `[1,24]`. It also returns the native RGB crop closest to one-based slot 12, breaking ties toward the earlier frame.
- For each training fold, compute the frozen DINOv2 teacher by running `build_dinov2_dense_adapter` and wrapping its backbone with `ResolutionDINOv2DenseAdapter`. Pass its dense tokens and validity mask to `masked_token_mean`. Average the resulting per-window vectors over the fold's training candidates and pass that center to `normalize_teacher_vectors`. Keep the target from the same indexed window during paired training.
- Train on one sampled window per training participant per epoch, using eight candidates per participant, batch size eight, and 60 epochs for each fold and seed. Stack windows along the batch dimension and add `labels` as a long tensor of shape `[B]` (0 negative, 1 neutral, 2 positive). Use `train_step` with the same-window centered target. The loss sums three-class cross-entropy and silhouette-to-RGB cosine distance.
- For the regional branch, run `extract_regions` with the CPU encoder returned by `load_pe_spatial` to obtain an `[8,768]` descriptor. Fit with `fit_regional(train_regions, train_valid, labels, people)`, using at most two distinct training windows per participant. Supply arrays of shape `[people,2,8,768]` and `[people,2]`, unique nonempty participant keys, and at least five participants with valid windows per class. The fit selects ridge regularization and temperature within its training fold. Fitted regional files contain training descriptors and are subject to the study's access terms.

## Participant-level scoring

For a trained model in eval mode, put video logits in a float32 array `[people,windows,3]` and regional descriptors in `[people,windows,8,768]`. Run `predict_regional`, then `fuse_regional`, passing the boolean regional-validity mask to both. Use the corresponding-region output under key `"P"`. It computes `0.65 * video + 0.35 * temperature * regional` for supported windows and uses the video logits for missing regional frames. Preserve all sampled slots, including repeats. Average the 20 window logits per participant and take the class argmax.

For the study protocol, keep all recordings from one participant in one fold and pool held-out predictions across the five folds separately for seeds 73111, 73112, and 73113. The recorded analysis used native RGB. Head-masked RGB supplied for controlled access can change model inputs and results.

The synthetic checks validate the software path. Reproducing the reported figures requires [controlled research access](README.md#dataset-access) to the original study inputs, fixed fold and window selections, and fitted artifacts or independent retraining under the same protocol.

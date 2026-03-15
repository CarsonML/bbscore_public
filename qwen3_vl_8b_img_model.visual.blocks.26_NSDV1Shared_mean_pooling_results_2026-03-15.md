# Qwen3-VL-8B-IMG `model.visual.blocks.26` NSDV1Shared Mean Pooling Results

## Config

- Model: `qwen3_vl_8b_img`
- Layer: `model.visual.blocks.26`
- Benchmark: `NSDV1Shared`
- Metric: `ridge`
- Representation method: `mean pooling over per-image patch activations`
- Aggregation mode: `none`
- Timestamp from run output: `2026-03-15T09:05:07.206261`

## Command

```bash
python run.py --model qwen3_vl_8b_img --layer model.visual.blocks.26 --benchmark NSDV1Shared --metric ridge
```

## Summary Metrics

- Final ceiled Pearson: `0.7660841402977353`
- Final unceiled Pearson: `0.5130076993261812`
- Final ceiled R2: `0.148172381542396`
- Final unceiled R2: `0.09294789297146902`

## Fold Medians

- Median unceiled Pearson by fold:
  `0.4993163554114887, 0.5205625143411419, 0.49181309946850155, 0.529324433780492, 0.502522261699496, 0.5265707319396882, 0.5047427931274133, 0.5080012289033331, 0.5292003334672861, 0.5180232411229713`
- Median ceiled Pearson by fold:
  `0.7630139790861088, 0.7509575693085533, 0.7239500458282292, 0.7795096193159887, 0.7763399858676847, 0.7950437553071432, 0.7383926124745268, 0.7644261461008435, 0.8031949349841627, 0.7660127547041115`
- Median unceiled R2 by fold:
  `0.0850279413754243, 0.09912520974455569, 0.0798893613337674, 0.10013699232749446, 0.08443500111650881, 0.102976669361035, 0.09193758197277502, 0.08814235484006505, 0.10075801306332693, 0.09704980457973739`
- Median ceiled R2 by fold:
  `0.13485636258614186, 0.15773772294732613, 0.12527576695924084, 0.15356867075488306, 0.1389680728746966, 0.16652217891023383, 0.1466847067453991, 0.14475320727775481, 0.16406074345053867, 0.14929638291774505`

## Interpretation Notes

- This run was produced before switching Qwen to preserve per-image patch tokens.
- The representation here compresses each image to a single vector by averaging patch activations within `model.visual.blocks.26`.
- Results appear stable across folds and clearly above zero, which suggests the layer carries meaningful predictive signal for `NSDV1Shared`.
- This file is intended as a reference point for comparing mean pooling against the later per-image patch-token version.

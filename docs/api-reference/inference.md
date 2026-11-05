# Inference API

Utilities for exporting pretrained SRGAN checkpoints and running tiled inference on Sentinel-2
imagery.

## Core helpers

::: opensr_srgan.inference
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - load_model
        - run_sen2_inference
        - main
      filters:
        - "!^_"
      members_order: source

## Prediction and matching methods

The model exposes three related entry points:

| Method | Inputs and behavior |
| --- | --- |
| `model.forward(normalized_lr)` | Raw generator output; no normalization or matching. |
| `model.predict_step(lr)` | Configured normalization, EMA inference, selected matching, and output in supplied input units on CPU. Requires evaluation mode. |
| `model.postprocess(sr, lr)` | Apply `Postprocessing.method` to existing SR/LR tensors already in the same units, without gradients or normalization. |

`Postprocessing.method` accepts `histogram` (including the missing-field default)
and `back_projection`. Histogram matching occurs before denormalization inside
`predict_step`; back-projection occurs afterward. Raw training losses and
validation metrics are unchanged. See the [inference guide](../inference.md#prediction-matching)
for enabling either method and the [model API](model.md#lightning-module) for the
SRGAN method documentation.

## Pretrained model factory

::: opensr_srgan._factory
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - load_from_config
        - load_inference_model
      filters:
        - "!^_"
      members_order: source

# Utility Helpers

Reusable helpers for logging, radiometric preprocessing, tensor conversion, and distributed-safe
side effects.

## Package exports

::: opensr_srgan.utils
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^_"
      members_order: source

## Logging helpers

::: opensr_srgan.utils.logging_helpers
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^__"
      members_order: source

## Radiometric transforms

::: opensr_srgan.utils.radiometrics
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^__"
      members_order: source

## Back-projection

Used when `Postprocessing.method: back_projection` is selected. The model's
`postprocess(sr, lr)` method infers the spatial scale; standalone calls use
`BackProjectionConfig`. Both inputs must share a value range. See the
[inference examples](../inference.md#prediction-matching).

::: opensr_srgan.utils.back_projection
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - BackProjectionConfig
        - BlurDownsample
        - back_projection
      members_order: source

## Tensor conversions

::: opensr_srgan.utils.tensor_conversions
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^__"
      members_order: source

## Model summarisation

::: opensr_srgan.utils.model_descriptions
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - print_model_summary
      members_order: source

## Distributed coordination

::: opensr_srgan.utils.gpu_rank
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^__"
      members:
        - _is_global_zero
      members_order: source

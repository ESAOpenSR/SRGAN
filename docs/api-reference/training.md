# Training API

Functions for launching PyTorch Lightning training runs and configuring trainer parameters from
OmegaConf-based experiment files.

## High-level entry points

::: opensr_srgan.train
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - train
      filters:
        - "!^_"
      members_order: source

## Hydra entry point

::: opensr_srgan.train_hydra
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - main
      filters:
        - "!^_"
      members_order: source

## Trainer configuration helpers

::: opensr_srgan.utils.build_trainer_kwargs
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - build_lightning_kwargs
      filters:
        - "!^_"
      members_order: source

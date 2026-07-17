# Hydra experiment configs

Hydra is the recommended entry point when you want to run repeatable experiments without copying and editing a full YAML file for every small change. The old single-file config workflow still works; Hydra is an additional layer that composes the same `Data`, `Model`, `Training`, `Generator`, `Discriminator`, `Optimizers`, `Schedulers`, and `Logging` sections that the SRGAN code already understands.

## Mental model

The legacy command loads one complete YAML file:

```bash
python -m opensr_srgan.train --config opensr_srgan/configs/config_training_example.yaml
```

The Hydra command starts from a small default config, selects reusable config groups, applies an optional experiment preset, and then applies command-line overrides:

```bash
python -m opensr_srgan.train_hydra experiment=10m Training.max_epochs=5 Logging.wandb.enabled=false
```

After composition, `opensr_srgan.train_hydra` forwards the resolved OmegaConf object to the same `opensr_srgan.train.train()` function used by the legacy workflow. This keeps the model, data, checkpoint, and logger code paths shared.

## Config layout

Hydra configs live in `opensr_srgan/configs/hydra/`:

| Path | Purpose |
| --- | --- |
| `train.yaml` | Default Hydra entry config. It selects the bundled example dataset, small CPU-safe training settings, CSV logging, and `experiment: null`. |
| `data/` | Dataset and dataloader presets such as `example`, `sen2naip`, and `lrhr_folder`. |
| `model/` | Combined `Model`, `Generator`, and `Discriminator` presets. |
| `training/` | Combined `Training`, `Optimizers`, and `Schedulers` presets. |
| `logging/` | CSV and W&B logging presets. |
| `experiment/` | Full experiment presets that reproduce the shipped legacy YAMLs. |
| `hydra/default.yaml` | Hydra's own run-directory and working-directory settings. |

The important convention is that each group writes into the existing top-level SRGAN schema. For example, `model/10m_esrgan.yaml` contributes `Model`, `Generator`, and `Discriminator`, not a new lowercase `model` object.

## Choosing a config

Use `experiment=...` when you want a known full setup:

```bash
python -m opensr_srgan.train_hydra experiment=example
python -m opensr_srgan.train_hydra experiment=10m
python -m opensr_srgan.train_hydra experiment=20m
```

Use direct overrides when you only want to change one or two values for a run:

```bash
python -m opensr_srgan.train_hydra experiment=10m Training.max_epochs=20
python -m opensr_srgan.train_hydra experiment=10m Generator.n_blocks=8 Optimizers.optim_g_lr=5e-5
python -m opensr_srgan.train_hydra experiment=20m Training.gpus=[0,1] Logging.wandb.enabled=false
```

Use group overrides when you want to mix reusable pieces:

```bash
python -m opensr_srgan.train_hydra data=example model=example_srresnet training=cpu_debug logging=csv
python -m opensr_srgan.train_hydra data=sen2naip model=10m_esrgan training=10m logging=wandb_10m
```

If a field does not already exist in the composed config, Hydra requires a `+` prefix. For normal SRGAN fields this is rarely needed because the presets define the expected sections up front.

## Output directories

Hydra creates a run directory under:

```text
logs/<wandb-project>/<timestamp>
```

The Hydra entry point copies that directory into `Logging.output_dir` before calling the existing trainer. The trainer then writes checkpoints and the resolved `config.yaml` there.

Hydra is configured with:

```yaml
hydra:
  job:
    chdir: false
```

This means relative paths keep behaving as if you launched the command from the repository root. Dataset paths such as `example_dataset/` therefore do not silently start resolving inside the run directory.

## Inspecting configs

Use Hydra's built-in config printing when you want to check exactly what will be passed to training:

```bash
python -m opensr_srgan.train_hydra --cfg job experiment=10m
python -m opensr_srgan.train_hydra --cfg hydra experiment=10m
```

The first command prints the resolved SRGAN job config. The second prints Hydra's runtime config, including output-directory settings.

## When to use legacy YAML

Use the legacy YAML entry point when you already have a complete hand-written config file or when a notebook/script expects a path. For the bundled example dataset, use:

```bash
python -m opensr_srgan.train --config opensr_srgan/configs/config_training_example.yaml
```

Use Hydra when you want quick overrides, reusable experiment pieces, or clean per-run config snapshots without creating many near-duplicate files. Use `experiment=10m` or `experiment=20m` after configuring the matching real datasets. Both paths call the same trainer once the config is loaded.

## References

Hydra composes these configs through a Defaults List. See the Hydra Defaults List documentation for the underlying mechanism: <https://hydra.cc/docs/1.1/advanced/defaults_list/>. The SRGAN entry point also uses an explicit `version_base` and `config_path`, following Hydra's upgrade guidance: <https://hydra.cc/docs/1.3/upgrades/version_base/>.

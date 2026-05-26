# Getting started

This guide walks through installing dependencies, configuring datasets, and launching your first ESA OpenSR experiment. The stack supports Python 3.10-3.12, PyTorch Lightning, and Weights & Biases for experiment tracking.

## Try it in Colab first

For the fastest start, open the interactive notebook in Google Colab and run through the introduction without setting up a local environment.

<p align="center">
  <a href="https://colab.research.google.com/drive/16W0FWr6py1J8P4po7JbNDMaepHUM97yL?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

> 💡 **Only need inference?** Install the published package instead: `python -m pip install opensr-srgan`. It exposes `load_from_config` and `load_inference_model` so you can instantiate models without cloning the repository. Continue with the rest of this guide when you want to train, fine-tune, or otherwise modify the codebase.

## 1. Install the environment

1. **Create a virtual environment.**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Python dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
3. **Authenticate logging backends (optional but recommended).**
   * Run `wandb login` to capture metrics and images in your W&B workspace.
   * Start `tensorboard --logdir logs/` if you prefer local dashboards.

## 2. Gather training data

The repository now ships with a single, ready-to-use example dataset so you can verify the full training loop without preparing custom manifests. Fetch it with the bundled helper:

```python
from opensr_srgan.data.example_data.download_example_dataset import get_example_dataset

get_example_dataset()  # downloads into ./example_dataset/
```

The script downloads `example_dataset.zip` from the Hugging Face Hub, extracts it to `example_dataset/`, and removes the archive once extraction finishes. The configuration only needs to specify the dataset type:

```yaml
Data:
  dataset_type: ExampleDataset
```

When you are ready to integrate your own collections, follow the guidance in [Data](data.md) to add a new dataset class and register it with the selector.

## 3. Configure the experiment

Use of the provided YAML presets or copy and edit one:

```bash
cp opensr_srgan/configs/config_10m.yaml opensr_srgan/configs/my_experiment.yaml
```

Update at least the following fields:

* `Data.dataset_type`: Keep `ExampleDataset` for the bundled sample or switch to your custom key once you register a new dataset.
* `Generator.scaling_factor`: Set the desired upscaling (e.g., `4` or `8`).
* `Model.load_checkpoint`: Provide a path if you want to initialize only model weights from an existing checkpoint.
* `Model.continue_training`: Provide a path if you want to fully resume interrupted training (optimizer/scheduler/EMA/global-step state).
* `Training.Losses.perceptual_metric`: Switch to `lpips` if you installed the optional dependency.

See [Configuration](configuration.md) for a full breakdown of available options.

## 4. Launch training

Run the training script with your customised config:

```bash
python -m opensr_srgan.train --config opensr_srgan/configs/my_experiment.yaml
```

Prefer to stay inside Python? Import the helper exposed by the package:

```python
from opensr_srgan import train

train("opensr_srgan/configs/my_experiment.yaml")
```

Both entry points will:

1. Instantiate the `SRGAN_model` Lightning module from the YAML file.
2. Build the appropriate dataset pair and wrap it in a `LightningDataModule`.
3. Configure Weights & Biases and TensorBoard loggers alongside checkpointing and learning-rate monitoring callbacks.
4. Start alternating generator/discriminator optimisation according to your warm-start schedule.

Training resumes automatically if `Model.continue_training` points to a Lightning checkpoint. If you interrupt training, always use the `Model.continue_training` flag to pass the generated checkpoint, since that restores all optimizers, schedulers, EMA etc. Do not set `Model.load_checkpoint` and `Model.continue_training` at the same time.

## 5. Run validation or inference

* **Validation metrics** are logged at the end of each epoch, including L1, SAM, PSNR/SSIM (from the content loss helper), and
  discriminator statistics.
* **Qualitative monitoring** is available through Weights & Biases image panels when `Logging.num_val_images` is greater than zero.
* **Inference** on new low-resolution tiles can reuse the Lightning module.
  * **When working from the PyPI package:**
    ```python
    from opensr_srgan import load_from_config, load_inference_model

    # Option A – bring your own config + checkpoint (local path or URL)
    custom_model = load_from_config(
        config_path="opensr_srgan/configs/config_10m.yaml",
        checkpoint_uri="https://example.com/checkpoints/srgan.ckpt",
        map_location="cuda",  # optional
    )

    # Option B – grab the published RGB-NIR/SWIR presets from Hugging Face
    preset_model = load_inference_model("RGB-NIR", map_location="cpu")
    ```
  * **When working from source:**
    ```python
    from opensr_srgan.model.SRGAN import SRGAN_model

    model = SRGAN_model("your_config.yaml")
    model.load_weights_from_checkpoint("path/to/checkpoint.ckpt")
    sr_tiles = model.predict_step(lr_tiles)
    ```
  In all cases the helpers automatically normalise Sentinel-2 ranges, apply histogram matching, and denormalise outputs for
  easier comparison with the source imagery.

## 6. Create Data Pipeline

* **SR Sen2 Tiles**: Use `opensr-utils` to crop, SR, patch, and overlap whole Sentinel-2 tiles. (Note: Currently only supports RGB-NIR.)

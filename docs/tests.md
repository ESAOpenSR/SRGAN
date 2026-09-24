# Tests

## Automated test suite

From the repository root, install the test dependencies and run the suite:

```bash
python -m pip install -e ".[tests]"
python -m pytest -q
```

The automated suite covers model components, training logic, data handling, inference, and deployment utilities. CI runs it with CPU-only PyTorch. The separate GPU integration check below is opt-in and requires two CUDA GPUs.

## Optional two-GPU DDP smoke test

The standalone integration check in `tests/manual/ddp_smoke.py` runs the real Lightning manual-optimization path on two CUDA GPUs with small, seeded random LR/HR tensors. It is not collected by `pytest` or run by ordinary CI. From the repository root, with the package and its dependencies installed, run:

```bash
CUDA_VISIBLE_DEVICES=0,1 timeout --kill-after=10s 180s \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/manual/ddp_smoke.py
```

`timeout` is the GNU/Linux utility: it bounds the whole run, including startup and collective hangs. A timeout or any nonzero exit status is a failure; a successful run prints `PASS rank=0` and `PASS rank=1` and exits with status zero. Two visible CUDA GPUs are required; missing hardware fails explicitly rather than silently skipping. No datasets, pretrained weights, or external logging services are needed, and any training output uses temporary directories.

The test uses the production trainer builder's `ddp_find_unused_parameters_true` strategy and checks, on both ranks:

* Five epochs of 20 training batches per rank (100 batches total): 25 generator-only batches followed by 75 adversarial batches. Pretraining crosses the first epoch boundary before switching to adversarial training.
* Explicitly enabled linear generator learning-rate warmup over 30 batches, including checks of the learning rate on every batch and that the warmup scheduler stops advancing afterward.
* Generator parameters change each batch; discriminator parameters remain fixed during pretraining and change during adversarial training.
* Finite losses and parameters, with matching generator/discriminator parameters across ranks after every batch, despite different distributed data shards.
* An adversarial weight that ramps from zero to its configured maximum over 40 optimizer steps, spanning an epoch boundary, and stays there afterward.
* One EMA update per batch and matching EMA shadow parameters across ranks.
* Two validation batches per rank after each epoch, finite validation metrics, EMA weight application and exact restoration of live generator parameters.
* Both plateau schedulers advance once per validation epoch. Their patience is deliberately longer than this run, so learning-rate reductions are not tested.
* The expected optimizer-step count: 25 generator-only steps plus 150 alternating discriminator/generator steps (175 total per rank). Lightning's `global_step` counts optimizer steps, so it advances twice per adversarial batch.

Each rank prints epoch progress, including its batch count, optimizer-step count, generator learning rate, and adversarial weight. The later epochs check continued training after both warmup and adversarial ramping have completed.

Gradient-norm clipping is enabled through `Optimizers.gradient_clip_val=0.5` in the manual training step. This check uses `32-true` precision and one node. It does not validate AMP, gradient accumulation, multi-node execution, checkpoint resume, every architecture/loss combination, or long-run convergence. Ordinary BatchNorm running buffers are not required to match across data shards; the synchronization assertions cover parameters, including EMA shadow parameters. The standard trainer builder does not currently forward `precision`, `accumulate_grad_batches`, or `num_nodes` from configuration; the manual loop updates optimizers every batch.

## Troubleshooting

### Environment and GPU setup

Run these checks in the same environment and GPU allocation used for the test:

```bash
python -m pip check
python -c "import torch, pytorch_lightning as pl; print('PyTorch:', torch.__version__); print('Lightning:', pl.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count())"
nvidia-smi
```

| Symptom | What to check |
| --- | --- |
| `No module named opensr_srgan` or `No module named pytest` | Activate the intended environment, change to the repository root, and run `python -m pip install -e ".[tests]"`. Use that same `python` for testing. |
| `Requires torchrun --nproc_per_node=2 and two visible CUDA GPUs` | Use the full launch command above. Running the script directly creates only one process. Check that two GPUs are allocated and visible to PyTorch. |
| CUDA is unavailable or fewer than two GPUs are visible | Check the installed PyTorch build, NVIDIA driver, container GPU access, and `CUDA_VISIBLE_DEVICES`. The CPU-only PyTorch installation used by CI cannot run this GPU check. On a cluster, request a two-GPU allocation before launching. |
| Invalid CUDA device index | The example assumes physical GPUs `0,1` are available. Adapt the visible-device selection to your machine; under a scheduler, preserve its assigned `CUDA_VISIBLE_DEVICES`. The two selected devices are addressed as logical devices `0,1` inside the process. |
| CUDA out of memory | Check `nvidia-smi` for competing workloads and use GPUs with free memory. This test uses a small model and synthetic tensors; if it still fails on otherwise idle GPUs, retain the traceback and environment details. |
| `timeout: command not found` | The example uses GNU `timeout`. Use an equivalent process timeout or scheduler wall-time limit on systems without it; keep a time limit so a distributed hang does not run indefinitely. |

### Distributed startup failures or hangs

DDP needs communication between the two local processes. A socket error such as `Operation not permitted` can indicate that a sandbox or container prevents local communication. Run in an environment that permits it. A GPU allocation alone does not guarantee that the process communication is permitted.

If the run hangs or times out, inspect the first traceback from **either rank**. One worker can fail while the other waits in a collective operation; the final launcher error may only summarize that earlier failure. Compare the last epoch progress lines from both ranks. Increasing the timeout is appropriate when the run is still progressing on a slow or busy machine, but does not fix a deadlock.

To capture the complete output, redirect both streams and preserve the exit code:

```bash
CUDA_VISIBLE_DEVICES=0,1 timeout --kill-after=10s 180s \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/manual/ddp_smoke.py > ddp-smoke.log 2>&1
test_status=$?
cat ddp-smoke.log
echo "Test exit status: $test_status"
```

### Failed assertions and expected warnings

A parameter mismatch, non-finite loss, unexpected optimizer update, or failed warmup/EMA assertion means the check failed, even if some epochs completed. Keep the assertion and its traceback when investigating; removing it or relaxing the synchronization tolerance would weaken the validation. Reproduce using the unchanged test configuration first if you have customized the script.

Warnings about a low DataLoader worker count or logging without a configured logger are expected for this small test: it deliberately uses `num_workers=0` and disables external logging. Success still requires both rank-specific `PASS` messages **and** exit status zero. Other warnings should be assessed from their contents rather than suppressed indiscriminately.

If a selected combination of pytest files fails during collection with `pytorch_lightning has no attribute LightningModule` or `pytorch_lightning is not a package`, check whether `tests/test_utils/test_trainer_kwargs.py` was collected first. That file installs a lightweight Lightning stub when Lightning has not already been imported. Until its test isolation is improved, run that file in a separate pytest process from model/training tests. The standalone DDP check runs in fresh processes and does not import this stub.

When reporting a failure, include the repository commit (`git rev-parse HEAD`), launch command, exit status, GPU models, PyTorch/Lightning versions, and the full log from both ranks. State whether failure occurred during startup, pretraining, the adversarial transition, or validation.

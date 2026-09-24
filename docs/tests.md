# Tests

## Automated test suite

From the repository root, install the test dependencies and run the suite:

```bash
python -m pip install -e ".[tests]"
python -m pytest -q
```

The automated suite covers model components, training logic, data handling,
inference, and deployment utilities. CI runs it with CPU-only PyTorch. The
separate GPU integration check below is opt-in and requires two CUDA GPUs.

## Optional two-GPU DDP smoke test

The standalone integration check in `tests/manual/ddp_smoke.py` runs the real
Lightning manual-optimization path on two CUDA GPUs with small, seeded random
LR/HR tensors. It is not collected by `pytest` or run by ordinary CI. From the
repository root, with the package and its dependencies installed, run:

```bash
CUDA_VISIBLE_DEVICES=0,1 timeout --kill-after=10s 180s \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/manual/ddp_smoke.py
```

`timeout` is the GNU/Linux utility: it bounds the whole run, including startup
and collective hangs. A timeout or any nonzero exit status is a failure; a
successful run prints `PASS rank=0` and `PASS rank=1` and exits with status zero.
Two visible CUDA GPUs are required; missing hardware fails explicitly rather
than silently skipping. No datasets, pretrained weights, or external logging
services are needed, and any training output uses temporary directories.

The test uses the production trainer builder's `ddp_find_unused_parameters_true`
strategy and checks, on both ranks:

* Five epochs of 20 training batches per rank (100 batches total): 25
  generator-only batches followed by 75 adversarial batches. Pretraining crosses
  the first epoch boundary before switching to adversarial training.
* Explicitly enabled linear generator learning-rate warmup over 30 batches,
  including checks of the learning rate on every batch and that the warmup
  scheduler stops advancing afterward.
* Generator parameters change each batch; discriminator parameters remain fixed
  during pretraining and change during adversarial training.
* Finite losses and parameters, with matching generator/discriminator parameters
  across ranks after every batch, despite different distributed data shards.
* An adversarial weight that ramps from zero to its configured maximum over 40
  optimizer steps, spanning an epoch boundary, and stays there afterward.
* One EMA update per batch and matching EMA shadow parameters across ranks.
* Two validation batches per rank after each epoch, finite validation metrics,
  EMA weight application and exact restoration of live generator parameters.
* Both plateau schedulers advance once per validation epoch. Their patience is
  deliberately longer than this run, so learning-rate reductions are not tested.
* The expected optimizer-step count: 25 generator-only steps plus 150
  alternating discriminator/generator steps (175 total per rank). Lightning's
  `global_step` counts optimizer steps, so it advances twice per adversarial batch.

Each rank prints epoch progress, including its batch count, optimizer-step
count, generator learning rate, and adversarial weight. The later epochs check
continued training after both warmup and adversarial ramping have completed.

Gradient-norm clipping is enabled through `Optimizers.gradient_clip_val=0.5` in
the manual training step. This check uses `32-true` precision and one node. It
does not validate AMP, gradient accumulation, multi-node execution, checkpoint
resume, every architecture/loss combination, or long-run convergence. Ordinary
BatchNorm running buffers are not required to match across data shards; the
synchronization assertions cover parameters, including EMA shadow parameters.
The standard trainer builder does not currently forward `precision`,
`accumulate_grad_batches`, or `num_nodes` from configuration; the manual loop
updates optimizers every batch.


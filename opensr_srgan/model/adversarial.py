"""Distributed RaGAN statistics and resolution-independent R1 regularization."""
import torch
import torch.distributed as dist


def relativistic_means(real, fake, *, global_batch=False):
    """Compute batch/patch means, retaining cross-rank mean gradients."""
    if not global_batch:
        return real.mean(), fake.mean()
    statistics = torch.stack((real.float().sum(), fake.float().sum(),
                              real.new_tensor(real.numel(), dtype=torch.float32),
                              fake.new_tensor(fake.numel(), dtype=torch.float32)))
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if torch.is_grad_enabled() and statistics.requires_grad:
            from torch.distributed.nn.functional import all_reduce
            statistics = all_reduce(statistics, op=dist.ReduceOp.SUM)
        else:
            statistics = statistics.detach().clone()
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    return statistics[0] / statistics[2], statistics[1] / statistics[3]


def normalized_patch_r1(discriminator, real_images):
    """Half squared input-gradient norm of each image's mean PatchGAN logit."""
    with torch.autocast(device_type=real_images.device.type, enabled=False):
        inputs = real_images.detach().float().requires_grad_(True)
        logits = discriminator(inputs)
        scores = logits.flatten(1).mean(1)
        gradient = torch.autograd.grad(scores.sum(), inputs, create_graph=True)[0]
        return 0.5 * gradient.flatten(1).square().sum(1).mean()


def image_logits(logits):
    return logits.detach().float().flatten(1).mean(1)


def critic_auc(real_scores, fake_scores):
    """Mann–Whitney AUC with average ranks for ties, using unsquashed logits."""
    if not real_scores.numel() or not fake_scores.numel():
        raise ValueError("Critic AUC requires real and generated scores")
    scores = torch.cat((real_scores, fake_scores)).double()
    labels = torch.cat((torch.ones_like(real_scores), torch.zeros_like(fake_scores)))
    order = scores.argsort()
    _, counts = torch.unique_consecutive(scores[order], return_counts=True)
    average_ranks = counts.cumsum(0).double() - (counts.double() - 1) / 2
    ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = (ranks * labels[order]).sum()
    n, m = real_scores.numel(), fake_scores.numel()
    return ((positive_rank_sum - n * (n + 1) / 2) / (n * m)).float()

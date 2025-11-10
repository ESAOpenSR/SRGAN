import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
import wandb


def sen2_test_step(self):
    from opensr_srgan.data.sen2_test.sen2_test_dataset import Sentinel2TestDataSet

    ds_test = Sentinel2TestDataSet()
    ims = []
    for i in range(4):
        im = ds_test.__getitem__(i)
        ims.append(im)
    ims = torch.stack(ims).to(self.device)
    if self.training == True:
        self.eval()
    sr_ims = self.predict_step(ims)
    sr_ims = sr_ims.cpu().numpy()

    # Detach + convert to numpy safely
    lr_ims = ims.cpu().numpy()

    # keep only RGB
    sr_ims = sr_ims[:, :3, :, :]
    lr_ims = lr_ims[:, :3, :, :]

    # to HWC
    sr_ims = sr_ims.transpose(0, 2, 3, 1)
    lr_ims = lr_ims.transpose(0, 2, 3, 1)

    # Optional: normalize for visualization (0–1)
    def norm(im):
        im = im*3.5  # assuming input is 0-1 scaled for Sentinel-2
        im = im.clip(0, 1)
        return im

    lr_ims = [norm(im) for im in lr_ims]
    sr_ims = [norm(im) for im in sr_ims]

    #lr_ims = lr_ims[:,:64,:64,:]
    #sr_ims = sr_ims[:,:256,:256,:]

    # Plot
    fig, axes = plt.subplots(2, len(lr_ims), figsize=(4 * len(lr_ims), 6))

    for i in range(len(lr_ims)):
        # top row: LR
        axes[0, i].imshow(lr_ims[i])
        axes[0, i].set_title(f"LR {i}")
        axes[0, i].axis("off")

        # bottom row: SR
        axes[1, i].imshow(sr_ims[i])
        axes[1, i].set_title(f"SR {i}")
        axes[1, i].axis("off")

    fig.tight_layout()

    # render to PIL
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    pil_image = Image.open(buf).convert("RGB").copy()
    buf.close()
    plt.close(fig)

    # Cleanup
    del ds_test, ims, sr_ims, lr_ims, fig, axes, buf

    # Log to wandb if enabled
    if self.config.Logging.wandb.enabled:
        if self.logger != None and hasattr(self.logger, "experiment"):
            self.logger.experiment.log(
                {"S2 SR": wandb.Image(pil_image)}
            )  # upload to dashboard

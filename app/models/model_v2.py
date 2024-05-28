import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    #model.cuda()
    model.eval()
    return model

class DiffusionModel:

    def __init__(self, config_path,model_path, n_iter=1, n_samples=1, scale=2.0, ddim_steps=10, eta=0.0, H = 512, W = 512):

        config = OmegaConf.load(config_path)

        self.n_iter = n_iter
        self.n_samples = n_samples
        self.scale = scale
        self.ddim_steps = ddim_steps
        self.ddim_eta = eta
        self.H = H
        self.W = W

        self.model = load_model_from_config(config, model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.sampler = PLMSSampler(self.model, device = self.device)

    def generate_with_prompt(self, prompt):
        all_samples=list()
        with torch.no_grad():
            with self.model.ema_scope():
                uc = None
                if self.scale != 1.0:
                    uc = self.model.get_learned_conditioning(self.n_samples * [""])
                for n in trange(self.n_iter, desc="Sampling"):
                    c = self.model.get_learned_conditioning(self.n_samples * [prompt])
                    shape = [4, self.H//8, self.W//8]
                    samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=self.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=self.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=self.ddim_eta)

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    all_samples.append(x_samples_ddim)


        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=self.n_samples)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        return grid.astype(np.uint8)

    def generate_without_prompt(self):
        return self.generate_with_prompt("")
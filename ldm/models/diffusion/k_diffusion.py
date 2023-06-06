from itertools import islice

import torch
import k_diffusion as K

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

class KSchedule(K.external.DiscreteSchedule):
    def __init__(self, model, quantize=True):
        self.device = model.device
        model_sigmas = ((1 - model.alphas_cumprod) / model.alphas_cumprod) ** 0.5
        super().__init__(model_sigmas, quantize)

    @property
    def model_sigmas(self):
        return self.sigmas

    def _conv_sigma_max(self, sigma_max):
        if sigma_max is None:
            return self.sigma_max
        if type(sigma_max) != torch.Tensor:
            return torch.tensor([sigma_max], device=self.device)
        if sigma_max.ndim == 0:
            return torch.tensor([sigma_max.item()], device=self.device)
        return sigma_max.to(self.device)

    def get_sigmas(self, n, sigma_max=None):
        """Variant of K.external.DiscreteSchedule.get_sigmas with sigma_max added
        as a parameter."""
        t_max = self.sigma_to_t(sigma_max.cuda()).item() if sigma_max is not None else len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return K.sampling.append_zero(self.t_to_sigma(t)).to('cpu')

    def get_sigmas_karras(self, n, sigma_max=None, **kwargs):
        sigma_max = self._conv_sigma_max(sigma_max)
        return K.sampling.get_sigmas_karras(n, self.sigma_min.cpu(), sigma_max.cpu(), **kwargs)

    def get_sigmas_exponential(self, n, sigma_max=None, **kwargs):
        sigma_max = self._conv_sigma_max(sigma_max)
        return K.sampling.get_sigmas_exponential(n, self.sigma_min.cpu(), sigma_max.cpu(), **kwargs)

    def get_sigmas_polyexponential(self, n, sigma_max=None, **kwargs):
        sigma_max = self._conv_sigma_max(sigma_max)
        return K.sampling.get_sigmas_polyexponential(n, self.sigma_min.cpu(), sigma_max.cpu(), **kwargs)

class KWrapper(torch.nn.Module):
    def __init__(self, model, schedule, steps, cond, uncond, scale=1.0):
        super().__init__()

        self.model = model
        self.vdiff = model.parameterization == 'v'

        self.schedule = schedule
        self.sigma_data = 1.0

        self.total_steps = steps
        self.current_step = 0

        self.cond = cond
        self.uncond = uncond
        self.scale = scale

        self.cond_batch_size = min(2, len(self.cond))

    @property
    def device(self):
        return self.model.device

    def get_scalings(self, sigma):
        # From crowsonkb/k-diffusion
        # https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py
        if self.vdiff:
            c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
            c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
            return c_out, c_in
        else:
            c_out = -sigma
            c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
            return c_out, c_in

    def get_skip_scaling(self, sigma):
        if self.vdiff:
            return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        else:
            return torch.tensor([1.0], device=self.model.device)

    @torch.no_grad()
    def get_eps(self, *args, **kwargs):
        # From crowsonkb/k-diffusion
        return self.model.apply_model(*args, **kwargs)

    @torch.no_grad()
    def predict(self, x, sigma, **kwargs):
        # FIXME: Doesn't work for v-diffusion
        return x + self.predict_eps(x, sigma, **kwargs)

    @torch.no_grad()
    def predict_eps(self, x, sigma, cond=None, **kwargs):
        """Same as predict, but doesn't add x, thus predicting eps and not x0.
        From crowsonkb/k-diffusion"""
        c_out, c_in = [K.utils.append_dims(s, x.ndim) for s in self.get_scalings(sigma)]

        cond = {
            'c_concat': [],
            'c_crossattn': [cond]
        }

        eps = self.get_eps(x * c_in, self.schedule.sigma_to_t(sigma), cond=cond, **kwargs)
        return eps * c_out

    def _get_all_conds(self):
        # Returns a concatenated list of all conditionings, including
        # the "unconditioning", taking into account the special cases
        # where the CFG scale is 1.0 or 0.0 and the unconditioning and
        # conditioning, respectively, are not needed.
        if self.scale == 1.0 and len(self.cond) == 1:
            all_conds = [*self.cond]
        elif self.scale == 0.0 and len(self.cond) <= 1:
            all_conds = [self.uncond]
        else:
            all_conds = [self.uncond, *self.cond]
        return all_conds

    def _get_cond_uncond_noises(self, noises, cond_weights):
        if self.scale == 1.0 and len(self.cond) == 1:
            cond_noise = sum(
                [noises[i] * cond_weights[i] for i in range(len(noises))])
            cond_weight_sum = sum(cond_weights)
            uncond_noise = torch.zeros_like(cond_noise, device=cond_noise.device)
        elif self.scale == 0.0 and len(self.cond) <= 1:
            uncond_noise = noises[0]
            cond_weight_sum = 0
            cond_noise = torch.zeros_like(uncond_noise, device=uncond_noise.device)
        else:
            uncond_noise, other_noises = noises[0], noises[1:]
            cond_weight_sum = sum(cond_weights[1:])
            cond_noise = sum(
                [other_noises[i] * cond_weights[i + 1] for i in range(len(other_noises))])
        return cond_noise, cond_weight_sum, uncond_noise

    def _get_cfg_noise(self, cond_noise, cond_weight_sum, uncond_noise):
        #return (1 - self.scale * cond_weight_sum) * uncond_noise + self.scale * cond_noise
        return uncond_noise + (cond_noise - uncond_noise) * self.scale * cond_weight_sum

    @torch.no_grad()
    def forward(self, x, sigma):
        # Speedup paths for scale in (0.0, 1.0)
        all_conds = self._get_all_conds()

        noises = []
        cond_weights = []

        for i, cond_batch in enumerate(chunk(all_conds, self.cond_batch_size)):
            cond_in = torch.cat([c for c in cond_batch])
            x_in = torch.cat([x] * len(cond_batch))
            sigma_in = torch.cat([sigma] * len(cond_batch))
            cond_weights.extend([1.0 for c in cond_batch])

            predicted_noises = self.predict_eps(x_in, sigma_in, cond=cond_in)\
                .chunk(self.cond_batch_size)
            noises.extend(predicted_noises)

        # Speedup paths for scale in (0.0, 1.0)
        cond_noise, cond_weight_sum, uncond_noise = self._get_cond_uncond_noises(noises, cond_weights)

        # Calculate the final noise after classifier-free guidance
        noise = self._get_cfg_noise(cond_noise, cond_weight_sum, uncond_noise)

        # Get the skip scale factor (equal to 1 if not using a v-diffusion model)
        skip_scale = K.utils.append_dims(self.get_skip_scaling(sigma), x.ndim)

        return x * skip_scale + noise

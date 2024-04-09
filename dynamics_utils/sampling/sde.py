# %%
from typing import Tuple, Union

import torch
import torchsde
from torchsde import BrownianInterval


class SDESampler(torch.nn.Module):
    def __init__(
        self,
        drift,
        diffusion,
        epsilon,
        device="cpu",
        sde_type="stratonovich",
        noise_type="diagonal",
    ):
        super(SDESampler, self).__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.epsilon = epsilon
        self.device = device
        self.sde_type = sde_type
        self.noise_type = noise_type

        # Check that the SDE type and noise type are valid.
        assert self.sde_type in ["ito", "stratonovich"]
        assert self.noise_type in ["diagonal", "additive", "scalar", "general"]

    def f(self, t, x):
        eps_t = torch.atleast_2d(self.epsilon(t)).T
        return self.drift(t, x) + eps_t * self.diffusion(
            t, x
        )  # * torch.sqrt(2 * eps_t)

    def g(self, t, x):
        eps_t = torch.atleast_2d(self.epsilon(t)).T
        g = torch.sqrt(2 * eps_t) * torch.ones_like(x)

        # Expand tensor for adding Brownian motion
        if self.noise_type != "diagonal":
            g = g[:, None]
        return g

    def sample(self, x0, t0=0.0, tf=1.0, dt=0.01, method="heun"):
        x0 = x0.to(self.device)
        ts = torch.linspace(t0, tf, int((tf - t0) / dt)).to(self.device)
        batch_size, n_atoms, n_dim = x0.shape
        x0_s = x0.reshape(batch_size, n_atoms * n_dim)
        bm = self.brownian_motion(batch_size, n_atoms * n_dim, t0, tf)
        xt = torchsde.sdeint(
            sde=self,
            y0=x0_s,
            ts=ts,
            method=method,
            bm=bm,
            dt=dt,
        )
        xt = xt.reshape(len(ts), batch_size, n_atoms, n_dim)
        return xt

    def brownian_motion(self, batch_size, feature_size, t0=0.0, tf=1.0):
        return BrownianInterval(
            t0,
            tf,
            size=(batch_size, feature_size),
            device=self.device,
        )


class LangevinSDE(torch.nn.Module):
    def __init__(
        self,
        kT,
        m,
        d,
        potential_grad,
        noise_type="diagonal",
        sde_type="stratonovich",
        seed=None,
    ):
        super().__init__()
        self.kT = kT
        self.m = m
        self.d = d
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.potential_grad = potential_grad
        if seed is not None:
            torch.manual_seed(seed)

    def f(self, t, x):
        # Drift term: -h * grad V / (m * d)
        return -self.potential_grad(x) / (self.m * self.d)

    def g(self, t, x):
        # Diffusion term: sqrt(2 * h * kT / (m * d))
        return torch.atleast_1d(
            torch.sqrt(torch.tensor(2.0 * self.kT / (self.m * self.d)))
        )[:, None]

    def sample(self, x0, n_steps, t0=0.0, tf=1.0, method="heun"):
        ts = torch.linspace(t0, tf, n_steps)
        trajectory = torchsde.sdeint(self, x0, ts, method=method)
        return trajectory.squeeze()

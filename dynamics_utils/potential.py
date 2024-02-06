import numpy as np


def step_high_friction_langevin(x, grad, dt=0.1, kT=15.0, mGamma=1000.0):
    return (
            x
            - dt / mGamma * grad(x)
            + np.random.standard_normal(x.shape) * np.sqrt((2.0 * kT * dt) / (mGamma))
    )


class OneDimensionalDoubleWellPotential:
    def __init__(self, a=5.0, b=1.0):
        self.a = a
        self.b = b

    def __call__(self, x):
        """Calculate the double well potential for a single particle."""
        return self.a * (x**2 - 1)**2 + self.b * x

    def grad(self, x):
        """Calculate the gradient (first derivative) of the double well potential."""
        return 4 * self.a * x * (x**2 - 1) + self.b


class LangevinSampler:
    def __init__(self, potential, x0=0.0, dt=0.1, kT=15.0, mGamma=1000.0, seed = None):
        self.potential = potential
        self.x = np.array([x0])
        self.dt = dt
        self.kT = kT
        self.mGamma = mGamma
        np.random.seed(seed)

    def step(self):
        self.x = step_high_friction_langevin(self.x, self.potential.grad, self.dt, self.kT, self.mGamma)

    def run(self, nsteps):
        x = np.zeros((nsteps, *self.x.shape))
        for i in range(nsteps):
            self.step()
            x[i] = self.x
        return x

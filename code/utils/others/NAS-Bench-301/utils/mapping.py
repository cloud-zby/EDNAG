"""
This module contains classes of fitness mapping function.
"""

import torch
import copy
import numpy as np
import matplotlib.pyplot as plt


class Identity:
    """Identity fitness mapping function."""

    def __init__(self, l2_factor=0.0):
        self.l2_factor = l2_factor

    def l2(self, x):
        return torch.norm(x, dim=-1) ** 2

    def forward(self, x):
        return x

    def __call__(self, x):
        return self.forward(x) * torch.exp(-1.0 * self.l2(x) * self.l2_factor)


class Energy(Identity):
    """Fitness mapping function that treats the fitness as energy.

    Args:
        temperature: float, the temperature of the system.

    Returns:
        p: torch.Tensor, the probability of the fitness. Compute by exp(-x / temperature).
    """

    def __init__(self, temperature=1.0, l2_factor=0.0):
        super().__init__(l2_factor=l2_factor)
        self.temperature = temperature

    def forward(self, x):
        eps = 5.0
        power = x / self.temperature
        power = power - power.max() + eps
        p = torch.exp(power)
        return p


class Power(Identity):
    """Fitness mapping function that returns the power of the fitness.

    Args:
        power: float, the power of the fitness.
        temperature: float, the temperature of the system.

    Returns:
        p: torch.Tensor, the probability of the fitness. Compute by (x / temperature) ** power.
    """

    def __init__(self, power=1.0, temperature=1.0, l2_factor=0.0):
        super().__init__(l2_factor=l2_factor)
        self.power = power
        self.temperature = temperature

    def forward(self, x):
        return torch.pow(x / self.temperature, self.power)


class Exp(Identity):
    """
    Fitness mapping function.
    Args:
        temperature: float, , ; , /
        max_fitness_val: float,
    Returns:
        p: torch.Tensor, [0, 1]
    """

    def __init__(self, temperature=50.0, max_fitness_val=100.0):
        super().__init__()
        self.temperature = temperature
        self.max_fitness_val = max_fitness_val

    def forward(self, x):
        # x0.0, 0.0
        if torch.all(x == 0.0):
            return torch.zeros_like(x)
        return torch.exp(x / self.temperature) / torch.exp(
            torch.tensor(self.max_fitness_val) / self.temperature
        )


class ReScale:
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor):
        y = copy.deepcopy(x)
        for i in range(x.shape[0]):
            if x[i] < 30:
                y[i] = 0.1 * x[i]
            elif x[i] < 50:
                y[i] = 3 + 1.0 * (x[i] - 30)
            elif x[i] < 70:
                y[i] = 23 + 2.0 * (x[i] - 50)
            elif x[i] < 75:
                y[i] = 63 + 3.0 * (x[i] - 70)
            elif x[i] < 80:
                y[i] = 78 + 5.0 * (x[i] - 75)
            elif x[i] < 85:
                y[i] = 103 + 10.0 * (x[i] - 80)
            elif x[i] < 90:
                y[i] = 153 + 30.0 * (x[i] - 85)
            elif x[i] < 95:
                y[i] = 303 + 40.0 * (x[i] - 90)
            else:
                y[i] = 503 + 50.0 * (x[i] - 95)

            if x[i] < 0.01 or x[i] > 99999:
                y[i] = 0.0

        return y

    def __call__(self, x):
        return self.forward(x)


def plot_mapping_fn(scale_fn, x_min=0, x_max=100, num_points=500):
    x = np.linspace(x_min, x_max, num_points)
    x = torch.tensor(x).float()
    y = scale_fn(x).detach().numpy()
    plt.plot(x, y)
    plt.xlabel("Original Fitness")
    plt.ylabel("Rescaled fitness")
    plt.title(f"{scale_fn.__class__.__name__} Fitness Mapping Function")
    plt.show()


if __name__ == "__main__":
    plot_mapping_fn(ReScale())
    plot_mapping_fn(Exp())
    plot_mapping_fn(Power(power=2.0))
    plot_mapping_fn(Energy(temperature=1.0))
    plot_mapping_fn(Identity())

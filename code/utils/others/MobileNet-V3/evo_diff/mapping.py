"""
This module contains classes of fitness mapping function.
"""

import torch


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
        temperature: float, 该值越大, 就越线性; 该值越小, 下凸性/上凸性越强
        max_fitness_val: float, 最大适应度值
    Returns:
        p: torch.Tensor, 将适应度值映射到[0, 1]区间
    """

    def __init__(self, temperature=50.0, max_fitness_val=100.0):
        super().__init__()
        self.temperature = temperature
        self.max_fitness_val = max_fitness_val

    def forward(self, x):
        # 如果x全都为0.0, 则返回全为0.0的张量
        if torch.all(x == 0.0):
            return torch.zeros_like(x)
        return torch.exp(x / self.temperature) / torch.exp(
            torch.tensor(self.max_fitness_val) / self.temperature
        )

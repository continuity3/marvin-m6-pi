# coding=utf-8
# Activation functions for transformers_replace
import torch
import torch.nn.functional as F


def gelu_pytorch_tanh(x):
    """GELU activation function using tanh approximation."""
    return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu(x):
    """GELU activation function."""
    return F.gelu(x)


def gelu_new(x):
    """New GELU activation function."""
    return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_fast(x):
    """Fast GELU activation function."""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def relu(x):
    """ReLU activation function."""
    return F.relu(x)


def silu(x):
    """SiLU activation function (also known as Swish)."""
    return F.silu(x)


def swish(x):
    """Swish activation function (same as SiLU)."""
    return F.silu(x)


def mish(x):
    """Mish activation function."""
    return x * torch.tanh(F.softplus(x))


def linear(x):
    """Linear activation function (no-op)."""
    return x


# Activation function mapping
ACT2FN = {
    "gelu": gelu,
    "gelu_pytorch_tanh": gelu_pytorch_tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "relu": relu,
    "silu": silu,
    "swish": swish,
    "mish": mish,
    "linear": linear,
}












"""Pick a ShiuBrain implementation: 'cpu' (sparse SciPy, brain_lif), 'cuda'
(Triton + CUDA graphs, brain_cuda) or 'auto' (cuda when torch sees a GPU)."""

from .brain_lif import ShiuBrain

BACKENDS = ("auto", "cpu", "cuda")


def resolve_backend(backend: str = "auto") -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend != "auto":
        return backend
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_brain(backend: str = "auto", seed: int = 0, **kwargs) -> ShiuBrain:
    backend = resolve_backend(backend)
    if backend == "cuda":
        from .brain_cuda import ShiuBrainCUDA
        return ShiuBrainCUDA(seed=seed, **kwargs)
    return ShiuBrain(seed=seed, **kwargs)


def describe(brain: ShiuBrain) -> str:
    base = getattr(brain, "backend", "cpu:sparse-scipy")
    if getattr(brain, "version", "783") != "783":
        base = f"{base} dataset:{brain.version}"
    ablate = getattr(brain, "ablate", "none")
    if ablate != "none":
        base = f"{base} ABLATED:{ablate}"
    if getattr(brain, "sensory_input", "free") != "free":
        base = f"{base} sensory:{brain.sensory_input}"
    if getattr(brain, "signs", "predicted") != "predicted":
        base = f"{base} signs:{brain.signs}"
    if getattr(brain, "aminergic", "excitatory") != "excitatory":
        base = f"{base} aminergic:{brain.aminergic}"
    if getattr(brain, "weight_scale", 1.0) != 1.0:
        base = f"{base} weights x{brain.weight_scale:g}"
    return base

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "shiu_model"


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the fetched connectome data (scripts/fetch_assets.py)")
    config.addinivalue_line("markers", "cuda: needs torch with a CUDA device")


def pytest_collection_modifyitems(config, items):
    has_data = DATA.exists()
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    for item in items:
        if "data" in item.keywords and not has_data:
            item.add_marker(pytest.mark.skip(reason="connectome data not fetched"))
        if "cuda" in item.keywords and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))

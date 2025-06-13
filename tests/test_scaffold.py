import torch

import tp_overlap


def test_cpu_torch_and_version():
    assert tp_overlap.__version__ == "0.1.0"
    assert torch.distributed.is_gloo_available()

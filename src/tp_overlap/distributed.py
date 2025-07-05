from datetime import timedelta

import torch.distributed as dist


def initialize_process_group(backend: str, init_method: str, rank: int, world_size: int) -> None:
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

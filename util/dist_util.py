'''
Adapted from mmcv/runner/dist_utils.py from open-mmlab. For further details:
    
    https://github.com/open-mmlab/mmcv/blob/f61295d944c7e10c7e1e16f6ca6f0b352d8e3545/mmcv/runner/dist_utils.py#L45

'''
import os
import subprocess

import torch
import torch.multiprocessing as mp
from torch import distributed as dist

def init_distributed_mode():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    
    is_slurm_job = "SLURM_JOB_ID" in os.environ
    if is_slurm_job:
        _init_dist_slurm()
    else:
        _init_dist_pytorch()
    
    return get_dist_info()


def _init_dist_pytorch():
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend='nccl')


def _init_dist_slurm(port=None):
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    gpu_to_work_on = rank % torch.cuda.device_count()
    return rank, world_size, gpu_to_work_on
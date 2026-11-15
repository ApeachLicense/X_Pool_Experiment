"""Run first_experiment.py with TF32 turned off for matmul and cuDNN.

PyTorch 1.7-1.11 compute float32 matmuls and convolutions in TF32 on Ampere GPUs by
default. Running a check through this file shows whether a numerical gap comes from
TF32. Usage is the same as first_experiment.py, for example:

    python run_with_tf32_off.py gradcheck --source-root ../xpool
"""
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import first_experiment  # noqa: E402  (imported after the flags on purpose)


if __name__ == '__main__':
    print('TF32 allowed: matmul {}, cudnn {}'.format(
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), flush=True)
    sys.exit(first_experiment.main())

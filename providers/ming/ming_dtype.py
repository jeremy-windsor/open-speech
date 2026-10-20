"""Compute dtype for the patched upstream Ming code.

The Dockerfile replaces Ming's hard-coded torch.bfloat16 with ming_dtype.MING_DTYPE,
and the worker sets it before it loads a model.
"""

import torch

MING_DTYPE = torch.bfloat16

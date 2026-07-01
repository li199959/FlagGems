# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from flag_gems.fused.FLA.chunk import chunk_gated_delta_rule_fwd
from flag_gems.fused.FLA.fused_gdn_prefill_post_conv import fused_post_conv_prep
from flag_gems.fused.FLA.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
from flag_gems.fused.FLA.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "fused_post_conv_prep",
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_sigmoid_gating_delta_rule_update",
]

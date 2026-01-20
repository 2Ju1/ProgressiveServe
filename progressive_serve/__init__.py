"""
ProgressiveServe Package (vLLM v0/v1 자동 감지)
"""

from .alpha_gated_layer import AlphaGatedLayer
from .progressive_llama_alpha import ProgressiveLlamaModelAlpha

# vLLM v0/v1 자동 감지
try:
    from vllm.v1.sample.sampler import Sampler
    # v1이 있다면 v1 버전 사용 
    from .progressive_llama_for_causal_lm_alpha import ProgressiveLlamaForCausalLMAlpha
    print("vLLM v1 detected")
except ImportError:
    # v1이 없으면 v0 버전 사용
    from .progressive_llama_for_causal_lm_alpha_v0 import ProgressiveLlamaForCausalLMAlpha
    print("vLLM v0 detected")

__all__ = [
    "AlphaGatedLayer",
    "ProgressiveLlamaModelAlpha",
    "ProgressiveLlamaForCausalLMAlpha",
]
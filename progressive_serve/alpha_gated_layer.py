"""
Alpha Gating Layer for ProgressiveServe (vLLM v0 Compatible)

CUDA Graph 호환 동적 레이어 활성화
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.cuda.nvtx as nvtx

class AlphaGatedLayer(nn.Module):
    """
    Alpha Gating을 사용한 동적 레이어 활성화 (vLLM v0)
    
    핵심 아이디어:
        y = x + alpha * F(x)
        
    - alpha = 0: 레이어 비활성 (Pass through)
    - alpha = 1: 레이어 활성 (Normal operation)
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        initial_alpha: float = 0.0,
    ):
        """
        Args:
            base_layer: 실제 LlamaDecoderLayer
            initial_alpha: 초기 alpha 값 (0.0 = 비활성)
        """
        super().__init__()
        
        self.layer = base_layer
        
        # Alpha gate
        self.register_buffer('alpha', torch.tensor(initial_alpha))
        
        self._is_active = initial_alpha > 0.5
    
    def forward(self, positions, hidden_states, residual):

        
        # base layer가 residual 관리 
        nvtx.range_push("BaseLayer")
        delta, updated_residual = self.layer(positions, hidden_states, residual)
        nvtx.range_pop()
        
        # alpha gating 적용
        nvtx.range_push("AlphaMultiply")
        gated_delta = self.alpha * delta
        nvtx.range_pop()
    
        return gated_delta, updated_residual
    
    def activate(self):
        """레이어 활성화 (alpha = 1)"""
        self.alpha.fill_(1.0)
        self._is_active = True
        print(f" Layer activated (alpha = 1.0)")
    
    def deactivate(self):
        """레이어 비활성화 (alpha = 0)"""
        self.alpha.fill_(0.0)
        self._is_active = False
        print(f" Layer deactivated (alpha = 0.0)")
    
    def is_active(self) -> bool:
        """활성화 여부 확인"""
        return self._is_active
    
    def get_alpha(self) -> float:
        """현재 alpha 값"""
        return self.alpha.item()
    
    @property
    def is_alpha_gated(self) -> bool:
        """AlphaGatedLayer 식별용"""
        return True

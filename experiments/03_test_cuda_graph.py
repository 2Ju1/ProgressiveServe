#!/usr/bin/env python3
"""
Progressive Stage 테스트 
"""

import sys
import os
import time
import torch


sys.path.insert(0, "/workspace/vllm_test")
sys.path.insert(0, "/acpl-ssd20/1218/A")
sys.path.insert(0, "/home/devewha/Juwon/vllm_test")

# vLLM import
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry
    from progressive_serve.progressive_llama_for_causal_lm_alpha_v0 import ProgressiveLlamaForCausalLMAlpha
    print("vLLM and Custom Model imported successfully")
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)

# 모델 레지스트리 등록
ModelRegistry.register_model(
    "ProgressiveLlamaForCausalLM",
    ProgressiveLlamaForCausalLMAlpha
)

def main():
    print("\n" + "="*80)
    print("Progressive Stage Test - Single Instance (Fixed)")
    print("="*80 + "\n")
    
    # 설정 경로
    model_path = "/acpl-ssd20/1218/A"
    stage2_path = "/acpl-ssd20/1218/checkpoints/stage2_layers_B.safetensors"
    stage3_path = "/acpl-ssd20/1218/checkpoints/stage3_layers_C.safetensors"

    # ========================================
    # 1. 초기화
    # ========================================
    print("[1/7] Initializing vLLM...")
    start_init = time.time()
    
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            enforce_eager=False # Nsight 분석을 위해 False(Graph 사용) 유지
        )
    except Exception as e:
        print(f"Failed to initialize vLLM: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    init_time = time.time() - start_init
    print(f"Initialization complete: {init_time:.2f}s\n")
    
    # 모델 객체 직접 접근
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        print(f"Model accessed: {type(model).__name__}\n")
    except Exception as e:
        print(f"Failed to access model: {e}")
        sys.exit(1)
    
    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0,
    )
    prompt = "What is the capital of France?"
    
    # ========================================
    # 2. Stage 1 추론
    # ========================================
    print("[2/7] Running Stage 1 inference...")
    start_s1 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    s1_time = time.time() - start_s1
    print(f"   Stage 1 complete: {s1_time:.4f}s")
    print(f"   Output: {outputs[0].outputs[0].text[:50]}...\n")
    
    # ========================================
    # 3. Stage 1 상태 확인
    # ========================================
    print("[3/7] Checking Stage 1 status...")
    try:
        model.print_status()
    except:
        try:
            info = model.get_stage_info()
            print(f"   Current stage: {info.get('stage')}\n")
        except:
            print(" Status method not found.\n")
    
    # ========================================
    # 4. Stage 2로 전환
    # ========================================
    print("[4/7] Transitioning to Stage 2...")
    start_transition = time.time()
    
    try:
        model.advance_to_stage2(layer_b_checkpoint=stage2_path)
        transition_time = time.time() - start_transition
        print(f" Stage 2 transition: {transition_time:.2f}s\n")
    except Exception as e:
        print(f" Stage 2 transition failed: {e}")
        sys.exit(1)
    
    # ========================================
    # 5. Stage 2 추론
    # ========================================
    print("[5/7] Running Stage 2 inference...")
    start_s2 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    s2_time = time.time() - start_s2
    print(f"   Stage 2 complete: {s2_time:.4f}s")
    print(f"   Output: {outputs[0].outputs[0].text[:50]}...\n")
    
    # ========================================
    # 6. Stage 3로 전환
    # ========================================
    print("[6/7] Transitioning to Stage 3...")
    start_transition = time.time()
    
    try:
        model.advance_to_stage3(layer_c_checkpoint=stage3_path)
        transition_time = time.time() - start_transition
        print(f" Stage 3 transition: {transition_time:.2f}s\n")
    except Exception as e:
        print(f" Stage 3 transition failed: {e}")
        sys.exit(1)
    
    # ========================================
    # 7. Stage 3 추론
    # ========================================
    print("[7/7] Running Stage 3 inference...")
    start_s3 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    s3_time = time.time() - start_s3
    print(f"   Stage 3 complete: {s3_time:.4f}s")
    print(f"   Output: {outputs[0].outputs[0].text[:50]}...\n")
    
    print("="*80)
    print("ALL STAGES COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()
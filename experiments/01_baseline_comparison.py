"""
Baseline vs Progressive 비교 
- 4가지 실행 모드: baseline만, progressive만, both, reversed
- vLLM 로그 파싱 및 저장
- 모드별 JSON 누적 저장
"""

import sys
import json
import os
import time
import uuid
import socket
import logging
import re
import argparse
from datetime import datetime
from typing import Dict, Optional

# Python path 설정
sys.path.insert(0, "/workspace/vllm_test")
sys.path.insert(0, "/acpl-ssd20/1218/A")
sys.path.insert(0, "/home/devewha/Juwon/vllm_test")

# vLLM imports
from progressive_serve.progressive_llama_for_causal_lm_alpha_v0 import ProgressiveLlamaForCausalLMAlpha
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# 모델 등록
ModelRegistry.register_model(
    "ProgressiveLlamaForCausalLM",
    ProgressiveLlamaForCausalLMAlpha
)


# JSON 파일 경로
JSON_FILES = {
    "baseline": "01_baseline_only.json",
    "progressive": "01_progressive_only.json",
    "both": "01_both_baseline_first.json",
    "reversed": "01_both_progressive_first.json"
}


class VLLMLogParser(logging.Handler):
    """vLLM 로그 파싱"""
    
    def __init__(self, save_raw_logs: bool = True, max_raw_logs: int = 100):
        super().__init__()
        self.save_raw_logs = save_raw_logs
        self.max_raw_logs = max_raw_logs
        self.logs = {
            "weight_loading_time": None,
            "model_loading_gb": None,
            "model_loading_time": None,
            "memory_profiling_time": None,
            "total_gpu_memory_gb": None,
            "gpu_memory_utilization": None,
            "model_weights_gb": None,
            "kv_cache_gb": None,
            "cuda_blocks": None,
            "cpu_blocks": None,
            "max_concurrency": None,
            "graph_capturing_time": None,
            "graph_capturing_memory_gb": None,
            "init_engine_time": None,
            "raw_logs": [] if save_raw_logs else None
        }
    
    def emit(self, record):
        msg = record.getMessage()
        
        # raw_logs 저장 
        if self.save_raw_logs and self.logs["raw_logs"] is not None:
            if len(self.logs["raw_logs"]) < self.max_raw_logs:
                self.logs["raw_logs"].append(msg)
        
        # Weight loading time
        m = re.search(r"Loading weights took ([\d.]+) seconds", msg)
        if m:
            self.logs["weight_loading_time"] = float(m.group(1))
        
        # Model loading
        m = re.search(r"Model loading took ([\d.]+) GB and ([\d.]+) seconds", msg)
        if m:
            self.logs["model_loading_gb"] = float(m.group(1))
            self.logs["model_loading_time"] = float(m.group(2))
        
        # Memory profiling
        m = re.search(r"Memory profiling takes ([\d.]+) seconds", msg)
        if m:
            self.logs["memory_profiling_time"] = float(m.group(1))
        
        # GPU memory 
        m = re.search(r"total_gpu_memory \(([\d.]+)GiB\) x gpu_memory_utilization \(([\d.]+)\)", msg)
        if m:
            self.logs["total_gpu_memory_gb"] = float(m.group(1))
            self.logs["gpu_memory_utilization"] = float(m.group(2))
        
        # Model weights
        m = re.search(r"model weights take ([\d.]+)GiB", msg)
        if m:
            self.logs["model_weights_gb"] = float(m.group(1))
        
        # KV Cache 
        m = re.search(r"KV Cache is ([\d.]+)GiB", msg)
        if m:
            self.logs["kv_cache_gb"] = float(m.group(1))
        
        # CUDA/CPU blocks
        m = re.search(r"# cuda blocks: (\d+), # CPU blocks: (\d+)", msg)
        if m:
            self.logs["cuda_blocks"] = int(m.group(1))
            self.logs["cpu_blocks"] = int(m.group(2))
        
        # Max concurrency
        m = re.search(r"Maximum concurrency.*?: ([\d.]+)x", msg)
        if m:
            self.logs["max_concurrency"] = float(m.group(1))
        
        # Graph capturing
        m = re.search(r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB", msg)
        if m:
            self.logs["graph_capturing_time"] = float(m.group(1))
            self.logs["graph_capturing_memory_gb"] = float(m.group(2))
        
        # Init engine time
        m = re.search(r"init engine.*?took ([\d.]+) seconds", msg)
        if m:
            self.logs["init_engine_time"] = float(m.group(1))



class BaselineComparison:
    """Baseline vs Progressive 비교"""
    
    def __init__(self):
        self.results = {}
    
    def setup_logging(self, save_raw_logs: bool = True, max_raw_logs: int = 100):
        logger = logging.getLogger("vllm")
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            if isinstance(handler, VLLMLogParser):
                logger.removeHandler(handler)
        
        # 새로운 파서 생성 및 추가
        log_parser = VLLMLogParser(save_raw_logs=save_raw_logs, max_raw_logs=max_raw_logs)
        logger.addHandler(log_parser)
        
        return log_parser
    
    def measure_model(
        self, 
        model_type: str,
        model_path: str,
        total_layers: int = 32,
        active_layers: Optional[int] = None,
        interface: str = NFS_INTERFACE,
        save_raw_logs: bool = True,
        max_raw_logs: int = 100
    ) -> Dict:
        """모델 측정 (baseline 또는 progressive)"""
        
        # 로깅 설정 - 새로운 파서 받기
        log_parser = self.setup_logging(save_raw_logs=save_raw_logs, max_raw_logs=max_raw_logs)
        
        print("\n" + "="*80)
        print(f"{model_type.upper()}: {os.path.basename(model_path)}")
        print(f"Model Path: {model_path}")
        print("="*80)
        
        # [1] Cold Start 측정
        print("\n[1] Measuring Cold Start Time...")
        cold_start_begin = time.time()
        net_monitor.start()
        
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="float16",
            trust_remote_code=True,
        )
        
        cold_start_time = time.time() - cold_start_begin
        network_metrics = net_monitor.stop()
        
        print(f" Cold Start Time: {cold_start_time:.2f} seconds")
        if network_metrics and not network_metrics.get('error', False):
            print(f" Network Throughput: {network_metrics['throughput_mbs']:.2f} MB/s "
                  f"({network_metrics['throughput_mbps']:.2f} Mbps)")
            print(f" Data Received: {network_metrics['bytes_received_gb']:.2f} GB")
        elif network_metrics and network_metrics.get('error', False):
            print(f" Network metrics collection failed: {network_metrics.get('error_message', 'Unknown error')}")
        
        # [2] TTFT 측정 - 처리 시간을 따로 콜드스타트로 분리(hydraserve등 정의 맞출것)
        print("\n[2] Measuring TTFT (Time To First Token)...")
        ttft_start = time.time()
        
        outputs = llm.generate(
            prompts=["What is the capital of France?"],
            sampling_params=SamplingParams(max_tokens=1, temperature=0)
        )
        
        ttft = time.time() - ttft_start
        print(f" TTFT: {ttft:.4f} seconds")
        
        # [3] Throughput 측정
        print("\n[3] Measuring Throughput...")
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms."
        ]
        
        throughput_start = time.time()
        
        outputs = llm.generate(
            prompts=test_prompts,
            sampling_params=SamplingParams(max_tokens=50, temperature=0)
        )
        
        throughput_duration = time.time() - throughput_start
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        output_throughput = total_output_tokens / throughput_duration
        
        print(f" Output Throughput: {output_throughput:.2f} tokens/sec")
        
        # [4] GPU Memory
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f" GPU Memory: {gpu_memory:.2f} GB")
        
        print("\n" + "="*80)
        print(f"{model_type.upper()} MEASUREMENT COMPLETE")
        print("="*80)
        
        # 결과 정리
        result = {
            "model_type": model_type,
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "cold_start_time": cold_start_time,
            "ttft": ttft,
            "output_throughput": output_throughput,
            "gpu_memory_gb": gpu_memory,
            "total_layers": total_layers,
            "active_layers": active_layers if active_layers else total_layers,
            "vllm_logs": log_parser.logs  
        }
        
        # Progressive인 경우 추가 정보
        if active_layers and active_layers < total_layers:
            result["activation_progress"] = f"{(active_layers/total_layers)*100:.0f}%"
            result["layers_pruned"] = total_layers - active_layers
        
        # 로그 핸들러 정리 (메모리 누수 방지)
        logger = logging.getLogger("vllm")
        logger.removeHandler(log_parser)
        
        return result
    
    def generate_comparison_table(self, baseline_result: Dict, progressive_result: Dict):
        """비교 결과 테이블 생성"""
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        # 개선율 계산
        cold_start_improvement = (
            (baseline_result["cold_start_time"] - progressive_result["cold_start_time"]) 
            / baseline_result["cold_start_time"] * 100
        )
        
        ttft_improvement = (
            (baseline_result["ttft"] - progressive_result["ttft"]) 
            / baseline_result["ttft"] * 100
        )
        
        memory_reduction = (
            (baseline_result["gpu_memory_gb"] - progressive_result["gpu_memory_gb"]) 
            / baseline_result["gpu_memory_gb"] * 100
        )
        
        
        
        print(f"\n{'Metric':<35} {'Baseline':<20} {'Progressive':<20} {'Improvement':<15}")
        print("-" * 90)
        print(f"{'Cold Start Time (s)':<35} {baseline_result['cold_start_time']:<20.2f} {progressive_result['cold_start_time']:<20.2f} {cold_start_improvement:>12.1f}%")
        print(f"{'TTFT (s)':<35} {baseline_result['ttft']:<20.4f} {progressive_result['ttft']:<20.4f} {ttft_improvement:>12.1f}%")
        print(f"{'Output Throughput (tok/s)':<35} {baseline_result['output_throughput']:<20.2f} {progressive_result['output_throughput']:<20.2f} {'-':>15}")
        print(f"{'GPU Memory (GB)':<35} {baseline_result['gpu_memory_gb']:<20.2f} {progressive_result['gpu_memory_gb']:<20.2f} {memory_reduction:>12.1f}%")
        print(f"{'Active Layers':<35} {baseline_result['active_layers']:<20} {progressive_result['active_layers']:<20} {'-':>15}")
        
       
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        print(f"Cold Start Reduction: {cold_start_improvement:.1f}%")
        print(f"TTFT Reduction: {ttft_improvement:.1f}%")
        print(f"Memory Savings: {memory_reduction:.1f}%")
        if network_reduction is not None:
            print(f"Network Data Reduction: {network_reduction:.1f}%")
        print(f"Model Size Reduction: {progressive_result.get('layers_pruned', 0)} layers pruned")
        print("="*80)


def save_experiment(mode: str, experiment_data: Dict):
    """실험 결과를 JSON 파일에 누적 저장"""
    
    json_file = JSON_FILES[mode]
    
    # 기존 파일 읽기
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = {
            "mode": mode,
            "description": get_mode_description(mode),
            "experiments": []
        }
    
    # 실험 ID 추가
    experiment_data["experiment_id"] = str(uuid.uuid4())
    experiment_data["hostname"] = socket.gethostname()
    
    # 누적 저장
    data["experiments"].append(experiment_data)
    
    # 파일 저장
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n Results saved to: {json_file}")
    print(f"   Total experiments in file: {len(data['experiments'])}")


def get_mode_description(mode: str) -> str:
    """모드 설명 반환"""
    descriptions = {
        "baseline": "Baseline model only (full Llama-2-7b with 32 layers)",
        "progressive": "Progressive model only (Stage 1 with 24 active layers)",
        "both": "Both models tested (Baseline first, then Progressive)",
        "reversed": "Both models tested (Progressive first, then Baseline)"
    }
    return descriptions.get(mode, "Unknown mode")


def main():
    """메인 실행 함수"""
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Baseline vs Progressive Comparison")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "progressive", "both", "reversed"],
        required=True,
        help="Execution mode: baseline (only baseline), progressive (only progressive), "
             "both (baseline→progressive), reversed (progressive→baseline)"
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="/acpl-ssd20/Llama-2-7b",
        help="Path to baseline model"
    )
    parser.add_argument(
        "--progressive-path",
        type=str,
        default="/acpl-ssd20/1218/A",
        help="Path to progressive model"
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=NFS_INTERFACE,
        help=f"Network interface to monitor (default: {NFS_INTERFACE})"
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Disable network monitoring"
    )
    parser.add_argument(
        "--no-raw-logs",
        action="store_true",
        help="Don't save raw vLLM logs (save space)"
    )
    parser.add_argument(
        "--max-raw-logs",
        type=int,
        default=100,
        help="Maximum number of raw logs to save (default: 100)"
    )
    
    args = parser.parse_args()
    

    print("Baseline vs Progressive Comparison")
    print(f"Mode: {args.mode.upper()}")
    print(f"Description: {get_mode_description(args.mode)}")

    
    comparison = BaselineComparison()
    experiment_data = {
        "mode": args.mode,
        "timestamp": datetime.now().isoformat()
    }
    
    baseline_result = None
    progressive_result = None
    
    # 모드별 실행
    if args.mode == "baseline":
        # Baseline만 실행
        print("Measuring Baseline Only")
        
        baseline_result = comparison.measure_model(
            model_type="baseline",
            model_path=args.baseline_path,
            total_layers=32,
            interface=args.interface,
            save_raw_logs=not args.no_raw_logs,
            max_raw_logs=args.max_raw_logs
        )
        experiment_data["baseline"] = baseline_result
    
    elif args.mode == "progressive":
        # Progressive만 실행
        print("Measuring Progressive Only")

        progressive_result = comparison.measure_model(
            model_type="progressive",
            model_path=args.progressive_path,
            total_layers=32,
            active_layers=24,
            interface=args.interface,
            save_raw_logs=not args.no_raw_logs,
            max_raw_logs=args.max_raw_logs
        )
        experiment_data["progressive"] = progressive_result
    
    elif args.mode == "both":
        # Baseline → Progressive
        print("STEP 1: Measuring Baseline")
        
        baseline_result = comparison.measure_model(
            model_type="baseline",
            model_path=args.baseline_path,
            total_layers=32,
            interface=args.interface,
            save_raw_logs=not args.no_raw_logs,
            max_raw_logs=args.max_raw_logs
        )
        experiment_data["baseline"] = baseline_result
        
        # GPU 메모리 클리어
        torch.cuda.empty_cache()
        time.sleep(2)
        print("STEP 2: Measuring Progressive")

        
        progressive_result = comparison.measure_model(
            model_type="progressive",
            model_path=args.progressive_path,
            total_layers=32,
            active_layers=24,
            interface=args.interface,
            save_raw_logs=not args.no_raw_logs,
            max_raw_logs=args.max_raw_logs
        )
        experiment_data["progressive"] = progressive_result
    
    elif args.mode == "reversed":
        # Progressive → Baseline
        print("STEP 1: Measuring Progressive")
        
        progressive_result = comparison.measure_model(
            model_type="progressive",
            model_path=args.progressive_path,
            total_layers=32,
            active_layers=24,
            interface=args.interface,
            save_raw_logs=not args.no_raw_logs,
            max_raw_logs=args.max_raw_logs
        )
        experiment_data["progressive"] = progressive_result
        
        # GPU 메모리 클리어
        torch.cuda.empty_cache()
        time.sleep(2)
        print("STEP 2: Measuring Baseline")
        
        baseline_result = comparison.measure_model(
            model_type="baseline",
            model_path=args.baseline_path,
            total_layers=32,
            interface=args.interface,
            save_raw_logs=not args.no_raw_logs,
            max_raw_logs=args.max_raw_logs
        )
        experiment_data["baseline"] = baseline_result
    
    # 비교 테이블 생성 (both 또는 reversed 모드인 경우)
    if baseline_result and progressive_result:
        comparison.generate_comparison_table(baseline_result, progressive_result)
    
    # 결과 저장
    save_experiment(args.mode, experiment_data)
    
    print("\nExperiment Complete!")
    print(f"   Mode: {args.mode}")
    print(f"   Results saved to: {JSON_FILES[args.mode]}")


if __name__ == "__main__":
    main()

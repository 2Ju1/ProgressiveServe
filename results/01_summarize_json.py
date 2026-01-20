#!/usr/bin/env python3
"""
JSON 실험 결과 요약 스크립트
"""
import json
import os
from datetime import datetime
from typing import List, Dict

# pandas 설치 확인
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


JSON_FILES = {
    "baseline": "results/01_baseline_only.json",
    "progressive": "results/01_progressive_only.json",
    "both": "results/01_both_baseline_first.json",
    "reversed": "results/01_both_progressive_first.json"
}


def extract_metrics(experiment: Dict, model_type: str) -> Dict:
    """실험 데이터에서 핵심 메트릭 추출"""
    
    model_data = experiment.get(model_type, {})
    if not model_data:
        return None
    
    vllm_logs = model_data.get("vllm_logs", {})
    
    # timestamp 파싱
    timestamp = experiment.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(timestamp)
        time_str = dt.strftime("%m/%d %H:%M:%S")
    except:
        time_str = timestamp[:19] if len(timestamp) >= 19 else timestamp
    
    metrics = {
        "timestamp": time_str,
        "model": model_type,
        "cold_start": model_data.get("cold_start_time", 0),
        "weight_loading": vllm_logs.get("weight_loading_time", 0),
        "model_loading": vllm_logs.get("model_loading_time", 0),
        "init_engine": vllm_logs.get("init_engine_time", 0),
        "ttft": model_data.get("ttft", 0),
        "throughput": model_data.get("output_throughput", 0),
        "gpu_memory": model_data.get("gpu_memory_gb", 0),
        "active_layers": model_data.get("active_layers", 32),
    }
    
    return metrics


def load_json_file(filepath: str) -> Dict:
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f" Error reading {filepath}: {e}")
        return None


def print_simple_table(data: List[Dict], title: str):
    
    print(f"\n{'='*120}")
    print(f"{title}")
    print(f"{'='*120}")
    
    # 헤더
    headers = ["Time", "Model", "Cold Start", "Weight Load", "Model Load", "Init Eng", "TTFT", "Throughput", "GPU Mem", "Layers"]
    widths = [16, 12, 12, 12, 12, 10, 10, 12, 10, 8]
    
    header_line = ""
    for h, w in zip(headers, widths):
        header_line += f"{h:<{w}}"
    print(header_line)
    print("-" * 120)
    
    # 데이터
    for row in data:
        line = ""
        line += f"{row['timestamp']:<16}"
        line += f"{row['model']:<12}"
        line += f"{row['cold_start']:<12.2f}"
        line += f"{row['weight_loading']:<12.2f}"
        line += f"{row['model_loading']:<12.2f}"
        line += f"{row['init_engine']:<10.2f}"
        line += f"{row['ttft']:<10.4f}"
        line += f"{row['throughput']:<12.2f}"
        line += f"{row['gpu_memory']:<10.2f}"
        line += f"{row['active_layers']:<8}"
        print(line)


def print_pandas_table(data: List[Dict], title: str):
    """pandas로 깔끔한 표 출력"""
    if not data:
        print(f" No data for {title}")
        return
    
    df = pd.DataFrame(data)
    

    column_order = [
        'timestamp', 'model', 'cold_start', 'weight_loading', 
        'model_loading', 'init_engine', 'ttft', 'throughput', 
        'gpu_memory', 'active_layers'
    ]
    df = df[column_order]
    

    df.columns = [
        'Time', 'Model', 'Cold Start', 'Weight Load', 'Model Load', 
        'Init Eng', 'TTFT', 'Throughput', 'GPU Mem', 'Layers'
    ]
    

    pd.options.display.float_format = '{:.2f}'.format
    
    print(f"\n{'='*120}")
    print(f"{title}")
    print(f"{'='*120}")
    print(df.to_string(index=False))


def analyze_file(filepath: str, mode: str):
    """JSON 파일 분석 및 표 출력"""
    
    data = load_json_file(filepath)
    if not data:
        return
    
    experiments = data.get("experiments", [])
    if not experiments:
        print(f"\n No experiments found in {filepath}")
        return
    
    # 모든 실험에서 메트릭 추출
    all_metrics = []
    
    for exp in experiments:
        # baseline과 progressive 둘 다 확인
        if "baseline" in exp:
            metrics = extract_metrics(exp, "baseline")
            if metrics:
                all_metrics.append(metrics)
        
        if "progressive" in exp:
            metrics = extract_metrics(exp, "progressive")
            if metrics:
                all_metrics.append(metrics)
    
    if not all_metrics:
        print(f"\n No valid metrics found in {filepath}")
        return
    
    # 표 출력
    title = f"{mode.upper()} - {os.path.basename(filepath)} ({len(experiments)} experiments)"
    
    if HAS_PANDAS:
        print_pandas_table(all_metrics, title)
    else:
        print_simple_table(all_metrics, title)


def print_comparison_summary(all_data: Dict):
    """전체 파일의 비교 요약"""
    
    if not all_data:
        return
    
    print(f"\n\n{'='*120}")
    print("COMPARISON SUMMARY")
    print(f"{'='*120}")
    
    summary = []
    
    for mode, filepath in JSON_FILES.items():
        data = all_data.get(mode)
        if not data or not data.get("experiments"):
            continue
        
        # 가장 최근 실험
        latest = data["experiments"][-1]
        
        # baseline과 progressive 비교
        for model_type in ["baseline", "progressive"]:
            if model_type in latest:
                metrics = extract_metrics(latest, model_type)
                if metrics:
                    metrics["file"] = mode
                    summary.append(metrics)
    
    if not summary:
        print(" No data to compare")
        return
    
    if HAS_PANDAS:
        df = pd.DataFrame(summary)
        column_order = [
            'file', 'model', 'cold_start', 'weight_loading', 
            'ttft', 'throughput', 'active_layers'
        ]
        df = df[[col for col in column_order if col in df.columns]]
        
        df.columns = [c.replace('_', ' ').title() for c in df.columns]
        
        pd.options.display.float_format = '{:.3f}'.format
        print(df.to_string(index=False))
    else:
        print_simple_table(summary, "Summary")


def calculate_improvements():
    """Baseline vs Progressive 평균 개선율 계산 """
    
    print(f"\n\n{'='*120}")
    print("IMPROVEMENT ANALYSIS (Baseline → Progressive)")
    print(f"{'='*120}")
    
    # 각 실험의 개선율 수집
    cold_start_improvements = []
    weight_loading_improvements = []
    ttft_improvements = []
    
    baseline_cold_starts = []
    progressive_cold_starts = []
    baseline_weight_loadings = []
    progressive_weight_loadings = []
    baseline_ttfts = []
    progressive_ttfts = []
    
    total_experiments = 0
    file_experiment_counts = {}
    
    print(f"\nCollecting experiments from all JSON files...\n")
    
    # 모든 JSON 파일에서 실험 수집
    for mode, filepath in JSON_FILES.items():
        data = load_json_file(filepath)
        if not data or not data.get("experiments"):
            continue
        
        experiments = data["experiments"]
        file_experiments = 0
        
        for exp in experiments:
            baseline_metrics = extract_metrics(exp, "baseline")
            progressive_metrics = extract_metrics(exp, "progressive")
            
            # baseline과 progressive 둘 다 있어야 비교 가능
            if not baseline_metrics or not progressive_metrics:
                continue
            
            # 개선율 계산
            cold_start_imp = (baseline_metrics["cold_start"] - progressive_metrics["cold_start"]) / baseline_metrics["cold_start"] * 100
            weight_loading_imp = (baseline_metrics["weight_loading"] - progressive_metrics["weight_loading"]) / baseline_metrics["weight_loading"] * 100
            ttft_imp = (baseline_metrics["ttft"] - progressive_metrics["ttft"]) / baseline_metrics["ttft"] * 100
            
            cold_start_improvements.append(cold_start_imp)
            weight_loading_improvements.append(weight_loading_imp)
            ttft_improvements.append(ttft_imp)
            
            baseline_cold_starts.append(baseline_metrics["cold_start"])
            progressive_cold_starts.append(progressive_metrics["cold_start"])
            baseline_weight_loadings.append(baseline_metrics["weight_loading"])
            progressive_weight_loadings.append(progressive_metrics["weight_loading"])
            baseline_ttfts.append(baseline_metrics["ttft"])
            progressive_ttfts.append(progressive_metrics["ttft"])
            
            file_experiments += 1
            total_experiments += 1
        
        if file_experiments > 0:
            file_experiment_counts[mode] = file_experiments
    
    if not cold_start_improvements:
        print(" No valid experiments found for comparison")
        return
    
    # 파일별 실험 개수 표시
    print(f"Found {total_experiments} comparable experiment(s):")
    for mode, count in file_experiment_counts.items():
        print(f"  - {mode:15} : {count} experiment(s)")
    print()
    
    # 각 실험 상세 출력
    exp_num = 1
    for mode, filepath in JSON_FILES.items():
        data = load_json_file(filepath)
        if not data or not data.get("experiments"):
            continue
        
        experiments = data["experiments"]
        
        for exp in experiments:
            baseline_metrics = extract_metrics(exp, "baseline")
            progressive_metrics = extract_metrics(exp, "progressive")
            
            if not baseline_metrics or not progressive_metrics:
                continue
            
            cold_start_imp = (baseline_metrics["cold_start"] - progressive_metrics["cold_start"]) / baseline_metrics["cold_start"] * 100
            weight_loading_imp = (baseline_metrics["weight_loading"] - progressive_metrics["weight_loading"]) / baseline_metrics["weight_loading"] * 100
            ttft_imp = (baseline_metrics["ttft"] - progressive_metrics["ttft"]) / baseline_metrics["ttft"] * 100
            
            print(f"Experiment {exp_num} ({mode}):")
            print(f"  Cold Start:      {baseline_metrics['cold_start']:.2f}s → {progressive_metrics['cold_start']:.2f}s ({cold_start_imp:+.1f}%)")
            print(f"  Weight Loading:  {baseline_metrics['weight_loading']:.2f}s → {progressive_metrics['weight_loading']:.2f}s ({weight_loading_imp:+.1f}%)")
            print(f"  TTFT:            {baseline_metrics['ttft']:.4f}s → {progressive_metrics['ttft']:.4f}s ({ttft_imp:+.1f}%)")
            print()
            
            exp_num += 1
    
    # 평균 계산
    avg_baseline_cold_start = sum(baseline_cold_starts) / len(baseline_cold_starts)
    avg_progressive_cold_start = sum(progressive_cold_starts) / len(progressive_cold_starts)
    avg_cold_start_improvement = sum(cold_start_improvements) / len(cold_start_improvements)
    
    avg_baseline_weight_loading = sum(baseline_weight_loadings) / len(baseline_weight_loadings)
    avg_progressive_weight_loading = sum(progressive_weight_loadings) / len(progressive_weight_loadings)
    avg_weight_loading_improvement = sum(weight_loading_improvements) / len(weight_loading_improvements)
    
    avg_baseline_ttft = sum(baseline_ttfts) / len(baseline_ttfts)
    avg_progressive_ttft = sum(progressive_ttfts) / len(progressive_ttfts)
    avg_ttft_improvement = sum(ttft_improvements) / len(ttft_improvements)
    
    # 표 출력
    print(f"{'='*80}")
    print(f"AVERAGE RESULTS (across {total_experiments} experiment(s) from all files)")
    print(f"{'='*80}")
    print(f"\n{'Metric':<25} {'Avg Baseline':<15} {'Avg Progressive':<18} {'Avg Improvement':<15}")
    print("-" * 75)
    print(f"{'Cold Start Time':<25} {avg_baseline_cold_start:<15.3f} {avg_progressive_cold_start:<18.3f} {avg_cold_start_improvement:>12.1f}%")
    print(f"{'Weight Loading Time':<25} {avg_baseline_weight_loading:<15.3f} {avg_progressive_weight_loading:<18.3f} {avg_weight_loading_improvement:>12.1f}%")
    print(f"{'TTFT':<25} {avg_baseline_ttft:<15.5f} {avg_progressive_ttft:<18.5f} {avg_ttft_improvement:>12.1f}%")
    
    # 표준편차 계산 
    if len(cold_start_improvements) > 1:
        import statistics
        std_cold_start = statistics.stdev(cold_start_improvements)
        std_weight_loading = statistics.stdev(weight_loading_improvements)
        std_ttft = statistics.stdev(ttft_improvements)
        
        print(f"\n{'Metric':<25} {'Std Dev':<15}")
        print("-" * 40)
        print(f"{'Cold Start Time':<25} {std_cold_start:<15.2f}%")
        print(f"{'Weight Loading Time':<25} {std_weight_loading:<15.2f}%")
        print(f"{'TTFT':<25} {std_ttft:<15.2f}%")
    
    print(f"\n Progressive model shows (AVERAGE across all files):")
    print(f"   - {avg_cold_start_improvement:.1f}% faster cold start")
    print(f"   - {avg_weight_loading_improvement:.1f}% faster weight loading")
    print(f"   - {avg_ttft_improvement:.1f}% faster TTFT")


def main():
    """메인 함수"""
    print("Experiment Results Summary")
    
    # 각 파일별로 분석
    all_data = {}
    
    for mode, filepath in JSON_FILES.items():
        if os.path.exists(filepath):
            analyze_file(filepath, mode)
            all_data[mode] = load_json_file(filepath)
        else:
            print(f"{filepath} not found. Skipping...")
    
    # 비교 요약
    if all_data:
        print_comparison_summary(all_data)
        calculate_improvements()

    print("Summary Complete")


if __name__ == "__main__":
    main()
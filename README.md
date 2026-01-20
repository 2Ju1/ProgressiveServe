# Progressive LLM Serving: System-Level Abstractions for Cold Start Reduction

> **Research Portfolio**  
> This repository presents my systems research on **LLM serving optimization**, focusing on *reducing cold-start latency through architectural abstractions*.  
> My work progresses from a **published domestic conference paper (KIISE 2025)** to an **ongoing system-architecture study targeting international venues**.

**Research focus**:
> *System-level abstractions that reduce LLM serving latency while preserving execution correctness and production constraints.*

---

## Research Overview

This repository contains **two related but distinct research efforts**:

- **KIISE 2025 (Published)**  
  *Progressive loading strategy for serverless LLM serving*

- **International Venue (In Progress)**  
  *Execution–memory decoupling as a serving architecture abstraction*

Together, they reflect a progression from **performance-driven optimization** to **architectural principles for ML systems**.

---

## Part 1. KIISE 2025 — Progressive Loading Strategy (Published)

### ProgressiveServe: Staged Model Loading

**Publication**:  
Korea Information Science Society (KIISE), December 2025

**Problem**:  
Serverless LLM cold start is dominated by model weight loading  
(13.5 GB → ~114s startup latency)

**Approach**:  
A **3-stage progressive loading strategy** that enables early execution with partial weights.

- Stage 1: Load selected layers → enable first response
- Stage 2–3: Incrementally recover remaining layers → restore full quality

**Key Results**:
- Cold start latency: **114s → 90s (21.1% ↓)**
- Final accuracy: **No degradation** vs full model
- Optional quality compensation via LoRA adapters

**Takeaway**:  
Progressive loading is effective, but **strategy-level optimization alone is insufficient** for production systems.

---

## Part 2. Ongoing Work — Execution–Memory Decoupling (International Venue Prep)

### Core Research Question

**Traditional assumption in DL systems**:
> Execution requires complete weight residency in memory.

**This work asks**:
> *Can logical execution be decoupled from physical memory state, 
> allowing inference to proceed while weights are still loading?*

Like how virtual memory decouples logical addresses from physical RAM, 
this approach separates execution flow from weight residency. 
Unlike VM's blocking page faults, our abstraction enables *asynchronous decoupling*—
execution continues with degraded output when weights are absent.

This reframes cold start optimization as a **systems abstraction problem**, 
not just a loading-speed issue.

---

### Proposed Abstraction: Alpha-Gated Execution

I introduce an abstraction where:
- The **execution graph remains fixed**
- Weights may be **absent or partially bound**
- Layer contribution is controlled dynamically via a lightweight gate

This enables:
- Deferred weight binding
- CUDA graph compatibility
- Runtime transition without graph reconstruction

---

### Prototype & Early Validation

**Implementation**:
- Integrated into **vLLM** (production LLM serving engine)
- Evaluated on **LLaMA-2-7B**, NVIDIA A6000

**Early Results**:
- Cold start latency: **29.6s → 19.2s (35.1% ↓)**
- Throughput impact: **<1%**
- CUDA graph topology: **unchanged across all loading stages**

These results suggest that **execution–memory decoupling is feasible in production-grade serving systems**.

---

### Current Focus

The ongoing work investigates:
- Eliminating abstraction overhead via **kernel-level fusion**
- Generalizing the abstraction beyond progressive loading
- Understanding theoretical and practical limits of decoupling

This study is being developed toward submission to an **international systems venue**.

---

## Repository Structure

```
progressive-serve/
├── README.md
├── 한국정보과학회_이주원.pdf       # KIISE 2025 paper
│
├── progressive_serve/
│   ├── __init__.py
│   ├── alpha_gated_layer.py            # Core abstraction
│   ├── progressive_llama_alpha.py      # vLLM integration
│   └── progressive_llama_for_causal_lm_alpha_v0.py
│
├── experiments/
│   ├── 01_baseline_comparison.py       # Main benchmark
│   ├── 01_summarize_json.py
│   ├── 02_baseline_comparison_13b.py
│   └── 03_test_cuda_graph.py      # CUDA graph verification
│
└── results/
    ├── 01_baseline_only.json
    ├── 01_progressive_only.json
    └── 01_both_baseline_first.json
```

---


## Research Status

- **KIISE 2025**: Published ✅
- **International Venue**: In progress  
  - Phase 1-2: Concept + prototype validated ✅
  - Phase 3: Kernel fusion optimization (ongoing)
  - Phase 4: Paper writing (planned)

## Contact

**Researcher**: Juwon Lee  
**Institution**: Ewha Womans University  
**Email**: 2276242@ewhain.net

**Research Interests**:  
-	LLM Inference Optimization & Serving Systems
-	Hardware-Software Co-design for AI Acceleration
-	Systems for Large-Scale AI Workloads


---

**Last Updated**: January 2026  
**Status**: KIISE published, international venue preparation in progress

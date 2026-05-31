# ComfyUI Hyperparameter Tuning & Metric Lab

This repository contains a server-side framework engineered for automated Grid Search, regression testing, and hyperparameter optimization within the ComfyUI ecosystem. It is designed to replace manual, interface-driven testing with structured execution matrices and objective, multi-metric evaluation vectors.

---

## Technical Architecture & Core Components

The framework operates via a decoupled server-side orchestration loop that bypasses the standard ComfyUI client interface. It flattens multi-dimensional parameter matrices, forces isolated micro-batch processing to manage VRAM overhead, and evaluates the output tensors natively.

### 1. `PromptBatchLoader` (Custom Node)
* **Source:** [`prompt_batch_loader.py`](./src/ComfyUI/custom_nodes/prompt_utils/prompt_batch_loader.py)
* **Function:** Serves as the matrix unrolling engine. It accepts a multi-dimensional JSON configuration matrix containing variant strings, steps, target CFG scales, and seed arrays. It flattens these permutations into synchronized index-linked arrays of length $N$ and programmatically overrides downstream batch allocation variables.

### 2. `MicroBatchKSampler` (Custom Node)
* **Source:** [`micro_batch_sampler.py`](./src/ComfyUI/custom_nodes/prompt_utils/micro_batch_sampler.py)
* **Function:** Manages compute isolation and mid-loop parameter injection. By enforcing a strict `max_micro_batch_size = 1` constraint, it sequentially iterates through the generation queue, dynamically reconfiguring `CFG`, `Steps`, and `Seed` parameters at runtime per individual frame. 
* **Edge-Case Handling:** Implements a localized token-slicing method (`slice_conditioning`) that isolates positive and negative text embeddings. If the parameter sweep depth outnumbers the textual variants, the loop retains the final valid token index to prevent index out-of-bounds errors and subsequent noise scheduler failures.
* **Data Integrity:** Outputs the final concatenated tensor explicitly wrapped as a single-element Python tuple `({"samples": merged_samples},)` to guarantee alignment with ComfyUI's down-stream unpacking expectations (`VAEDecode`).

### 3. `MultiMetricScorer` (Custom Node)
* **Source:** [`multi_metric_scorer.py`](./src/ComfyUI/custom_nodes/prompt_utils/multi_metric_scorer.py)
* **Function:** Performs automated, post-decode tensor analysis. It extracts pixel data directly from the VAE layer and routes it through two primary analytical pipelines:
  * **Semantic Alignment:** Evaluates prompt adherence using the `openai/clip-vit-large-patch14` embedding space.
  * **Structural Sharpness:** Computes a Laplacian variance kernel to detect edge contrast quality and capture pixel burnout or texture degradation.
  * **Output:** Generates a tournament-style aggregation matrix written directly to `ComfyUI/output/experiment_name/prompt_ranking.csv` on Google Drive.

---

## System Evolution & R&D Milestones

The infrastructure reached its current state through an iterative series of development phases designed to systematically eliminate confounding variables and evaluation errors[cite: 1]. Every phase was initiated as a direct logical improvement to address the core bottlenecks of the previous step.

### Phase 1: API-Driven Baseline & Metrics Setup
* **Context & Bottleneck:** Initial attempts relied on purely visual inspection of outputs from the ComfyUI Frontend API[cite: 1]. However, manual evaluation proved highly subjective and lacked scalability. To establish objective baselines, automated telemetry was introduced to calculate raw metrics from generated frames.
* **Model Constraint:** This baseline phase was conducted entirely using Stable Diffusion 1.5. Due to persistent limitations in structural and semantic fidelity, the project transitioned to SDXL.
* **Historical Documentation:** This phase is detailed in the archival report: [`Evaluation of Semantic Leakage and Style Integration in SD1.5`](./reports/jedi_van_gogh_1/report.md)

### Phase 2: Paired Evaluation Paradigm
* **Context & Bottleneck:** Having raw metrics for individual images was insufficient for tracking specific prompt or checkpoint updates without a controlled baseline.
* **Implementation:** Designed a structured architecture for automated **Paired Evaluation**[cite: 1]. Operating as a direct analogy to classic A/B testing, it ran dual-variant inferences under strictly identical environments to isolate single-variable impacts[cite: 1].

### Phase 3: Transition to SDXL & Matrix Expansion
* **Context & Bottleneck:** Moving to SDXL introduced massive composition shifts that random seeding obscured. 
* **Implementation:** Implemented cross-prompt validation (`run_same_seed_another_prompt`) by locking the initial latent noise distribution while mutating token strings[cite: 1]. To stabilize comparative analysis, the infrastructure began generating stitched grids (combining paired outputs sharing identical seeds into a single matrix).
* **Historical Documentation:** This transition and its associated telemetry are captured in: [`Performance Analysis: SDXL Transition & Semantic Optimization`](./reports/jedi_van_gogh_2/report.md).
* **Iterative Optimization Sweeps:** Following the transition, two critical optimization sweeps were executed to resolve prompt pollution and hardware limits:
  1. **Prompt Slicing & Iteration (`plain_cat_1_1`):** Analyzed token behavior on a mathematically isolated subject (a cosmic cat) to clear conflicting prompt signals. Iterations step-by-step isolated subject modifiers from background noise (`v1_1` to `v2_2`) and introduced prompt weights to adjust emotional emphasis (`v3_1`). *Summary:* [`Prompt Iteration Evaluation on SDXL Base 1.0`](./reports/plain_cat_1/report.md).
  2. **Parameter Sensitivity Sweep (`plain_cat_2`):** Run on an NVIDIA Tesla T4 backend, this matrix evaluated the interactions between CFG, Steps, Samplers, and Schedulers[cite: 1]. It identified a distinct "Goldilocks Zone" (Steps=30, CFG=8), beyond which non-linear degradation occurred, such as chromatic burn and texture breakdown. *Summary:* [`SDXL A/B Testing Matrix & Inference Optimization`](./reports/plain_cat_2/report.md)

### Phase 4: Spatial Conditioning via ControlNet (Canny)
* **Context & Bottleneck:** While prompt token adjustment and parameter tuning optimized semantic output, they could not resolve the fundamental structural limits inherent to SDXL's base composition engine when dealing with highly complex geometric boundaries.
* **Implementation:** Integrated dense spatial conditioning via ControlNet (Canny edge detection) to enforce invariant geometric boundaries across text mutations[cite: 1]. Testing focused on precise "Object Removal and Addition" tasks by calibrating `High threshold`, `Strength`, `End Percent`, and Gaussian Blur `Sigma`. This allowed the system to isolate and map the exact "point of no return"—the threshold of **Semantic Handover** where spatial conditioning loses control to text-prompt generation.
* **Historical Documentation:** Documented in: [`Structural Conditioning and Parameter Calibration Report`](./reports/ControlNet/report.md)

### Phase 5: Semantic Isolation & Regression Testing
* **Context & Bottleneck:** Processing complex prompts through ControlNet and dynamic weights revealed a telemetry flaw: passing complete, unified prompt strings into the CLIP validator caused massive signal cross-contamination between subject and style scores[cite: 1].
* **Implementation:** Refactored the data ingest to explicitly decouple subject nouns from stylistic modifiers before tensor scoring[cite: 1]. To ensure that this major telemetry rewrite did not alter historical calculation accuracy, a strict **Regression Test Subsystem** (`run_test_identity_check`) was built to verify metric stability down to floating-point precision[cite: 1].

### Phase 6: Statistical Trend Aggregation
* **Context & Bottleneck:** Evaluation matrices grew too massive for linear row-by-row comparisons. 
* **Implementation:** Replaced raw telemetry outputs with mathematical delta indicators[cite: 1]. The system now automatically aggregates paired execution lanes into dynamic indicators: `V1_Mean`, `V2_Mean`, `Delta_Mean`, and `Improved_Pct` to capture macro-level model performance shifts[cite: 1].

### Phase 7: Deterministic Initialization via Golden Seeds
* **Context & Bottleneck:** Random seeds frequently introduced structural outliers that skewed statistical averages, making poor prompt variants look artificially performant due to lucky initial noise.
* **Implementation:** Replaced arbitrary noise with a curated array of verified `Golden Seeds`—distributions proven to yield optimal compositional complexity—ensuring hyperparameter variations were evaluated against reliable baselines[cite: 1].

### Phase 8: Native Backend Consolidation (Current State)
* **Context & Bottleneck:** The system had grown into a fragmented array of external API calls, pipeline scripts, and file system watches, causing immense disk I/O bottlenecks and frequent memory leak collisions.
* **Implementation:** Fully consolidated all isolated procedural scripts, testing harnesses, and scoring engines into native ComfyUI Python extensions (`Custom Nodes`)[cite: 1]. This centralized the tensor data-flow directly within the execution graph[cite: 1].
* **Framework Telemetry Report:** The benchmarks, architecture, and performance results of this fully consolidated custom node structure are maintained in the report: [`report.md`](./reports/custom_node/report.md)

---

## Methodological Trade-offs & Current Limitations

### Design Choices
* **Atomic Slicing ($max\_micro\_batch\_size = 1$):** Prioritizes hardware stability and total parameter isolation over raw throughput[cite: 1]. It eliminates VRAM spikes at the cost of parallel processing efficiency, making it highly reliable for prolonged unmonitored sweeps on standard infrastructure[cite: 1].
* **Rigid Seed Rotation:** The system enforces a strict constraint where shifting hyperparameters are tested against an invariant seed within an execution slice, and the seed rotates exclusively between macro-iterations[cite: 1]: 
$$\text{Seed}_{I} = \text{Constant for all } H_{\kappa}, \quad \text{Seed}_{I} \neq \text{Seed}_{I+1}$$[cite: 1]
This isolates noise distribution from parameter behavior, though it requires broader matrices to average out seed-specific bias[cite: 1].

### Current Limitations
* **Linear Sigma Scaling Vulnerability:** When sweeping deeply varied `Steps` ranges (e.g., 15 vs 40 steps), native ComfyUI schedulers recalculate the entire noise-depletion grid[cite: 1]. Consequently, step 10 in a 15-step run represents a different noise level than step 10 in a 40-step run, subtly shifting the generation path[cite: 1].
* **Cold-Start Telemetry Latency:** The system does not currently account for CUDA warm-up times or weight-offloading delays during the initial execution slice, which slightly skews the execution time metric for the first frame of a batch[cite: 1].

---

## Getting Started

1. **Matrix Definition:** Configure your target variables within the execution dictionary inside the configuration module [`multi_prompt.json`](./config/multi_prompt.json)
2. **Execution:** Execute the headless orchestration layer:
   ```bash
   python orchestrator.py
3. **Data Retrieval:** Upon pipeline completion, inspect the analytical summary directly inside the output directory: ComfyUI/output/experiment_name/prompt_ranking.csv on Google Drive.

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

The infrastructure reached its current state through an iterative series of development phases designed to systematically eliminate confounding variables and evaluation errors:

* **Phase 1: API-Driven Baseline:** Initial implementation focused on triggering generation requests via the ComfyUI Frontend API and routing completed files to an external post-processing environment for metric calculation.
* **Phase 2: Paired Evaluation:** Automation of basic A/B testing paradigms, executing dual-variant runs under structurally identical conditions to isolate single-variable impacts.
* **Phase 3: Latent Stability Verification (`run_same_seed_another_prompt`):** Implemented cross-prompt validation by locking the initial latent noise distribution while mutating token strings, isolating how effectively different prompt structures guide identical base noise.
* **Phase 4: Structural Conditioning via ControlNet (Canny):** Integrated dense spatial conditioning using ControlNet (Canny edge detection) to enforce geometric boundaries across test variants. This allowed the system to benchmark prompt text mutations and parameter adjustments against an invariant, complex spatial composition (e.g., testing structural rendering performance on highly irregular object intersections).
* **Phase 5: Semantic Isolation & Regression Testing:** * *Telemetry Refactor:* Discovered that evaluating entire combined prompt strings through the CLIP validator caused significant signal pollution. The input logic was refactored to explicitly decouple subject nouns from stylistic modifiers before scoring.
  * *Regression Framework:* Built an automated test harness that reruns historical target images to verify that core metric calculations remain stable within floating-point precision during refactoring.
* **Phase 6: Statistical Trend Aggregation:** Replaced raw outputs with higher-level mathematical delta indicators, computing `V1_Mean`, `V2_Mean`, `Delta_Mean`, and `Improved_Pct` across paired execution lanes to capture macro-level model performance shifts.
* **Phase 7: Deterministic Initialization (`run_golden_seeds`):** Replaced arbitrary random noise seeds with a curated array of verified "Golden Seeds"—noise distributions proven to yield optimal compositional complexity—ensuring that hyperparameter behaviors are tested against structurally sound baselines.
* **Phase 8: Native Backend Consolidation (Current State):** Migrated decoupled procedural scripts and evaluation routines into native ComfyUI Python extensions, consolidating the data-flow within the tensor pipeline and eliminating intermediate disk I/O bottlenecks.

---

## Methodological Trade-offs & Current Limitations

### Design Choices
* **Atomic Slicing ($max\_micro\_batch\_size = 1$):** Prioritizes hardware stability and total parameter isolation over raw throughput. It eliminates VRAM spikes at the cost of parallel processing efficiency, making it highly reliable for prolonged unmonitored sweeps on standard infrastructure.
* **Rigid Seed Rotation:** The system enforces a strict constraint where shifting hyperparameters are tested against an invariant seed within an execution slice, and the seed rotates exclusively between macro-iterations:
$$\text{Seed}_{I} = \text{Constant for all } H_{\kappa}, \quad \text{Seed}_{I} \neq \text{Seed}_{I+1}$$
This isolates noise distribution from parameter behavior, though it requires broader matrices to average out seed-specific bias.

### Current Limitations
* **Linear Sigma Scaling Vulnerability:** When sweeping deeply varied `Steps` ranges (e.g., 15 vs 40 steps), native ComfyUI schedulers recalculate the entire noise-depletion grid. Consequently, step 10 in a 15-step run represents a different noise level than step 10 in a 40-step run, subtly shifting the generation path.
* **Cold-Start Telemetry Latency:** The system does not currently account for CUDA warm-up times or weight-offloading delays during the initial execution slice, which slightly skews the execution time metric for the first frame of a batch.

---

## Getting Started

1. **Matrix Definition:** Configure your target variables within the execution dictionary inside the configuration module.
2. **Execution:** Execute the headless orchestration layer:
   ```bash
   python orchestrator.py
2. **Data Retrieval:** Upon pipeline completion, inspect the analytical summary directly inside output/prompt_ranking.csv.

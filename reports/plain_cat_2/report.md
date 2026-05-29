# Title
SDXL A/B Testing Matrix & partially Inference Optimization
Stability AI / SDXL Base 1.0 — Parameter Sensitivity Analysis
Environment: ComfyUI / Python Backend on NVIDIA Tesla T4 (16GB VRAM)
Target Subject: Simple Cat
Objective: Identify the Pareto frontier for high-fidelity generative pipelines.
Experiment ID: plain_cat_2

## Summary
[//]: # (The experiment utilized a Fixed Seed A/B Matrix to isolate the impact of denoising steps on high-frequency details. )
[//]: # (Results indicate a non-linear relationship between step count and image coherence, with a significant "quality floor" identified at N < 30.)
This report analyzes the impact of Denoising Steps ($N$), Classifier-Free Guidance (CFG), Samplers, and Schedulers on image coherence and semantic alignment.
Key Finding: Increasing parameters beyond a "Goldilocks Zone" (Steps=30, CFG=8) results in non-linear degradation, characterized by chromatic burn, loss of textural realism, and diminishing semantic returns.

## Parameter Analysis: Denoising Steps ($N$)
**N=15: Sub-optimal Convergence**
- **Visual Artifacts**: High prevalence of "unpainted" latent regions and aliasing.
Characterized by a "coloring book" effect and flat, non-volumetric textures ($RGB \approx 0,0,0$ in shadows).
The model fails to complete the denoising trajectory, resulting in a "coloring book" effect where subject outlines are established but volumetric shading is absent.
- **Geometric Fidelity**: Significant aliasing ("zigzags") on high-contrast edges, specifically ocular contours.
- **Textural Rendering**: Whiskers lack volumetric depth and fail to respond to environmental lighting (flat white stalks). 
Deep shadows are rendered as crushed blacks ($RGB \approx 0,0,0$) rather than nuanced gradients.
- **Analogy**: Equivalent to a high-exposure "flash-blinded" photograph with clipped highlights and lost mid-tones.

**N=30: Optimal**
The convergence sweet spot for production-grade assets.
- **Observations**: Drastic improvement in ocular geometry and textural resolution. Most latent noise is resolved.
- **Subject Fidelity**: Significant jump in realism compared to $N=15$. 
This represents the baseline for production-grade assets where inference time and quality are balanced.

**N=60: Diminishing Returns & High-Fidelity Refinement**
Marginal gains in hair follicle separation and ocular specularity, but statistically irrelevant for non-macro shots.
- **Textural Density**: In "too close-up" scenarios, $N=60$ provides marginal gains in individual hair follicle separation and ocular specularity.
- **Perceptual Delta**: While the difference between $N=30$ and $N=60$ is negligible in wide/mid-shots, it becomes statistically relevant in macro-photography prompts where textural micro-details are the primary focus.

## Parameter Analysis: CFG Scale (Guidance Weight)
CFG dictates the trade-off between prompt adherence and image diversity.

**CFG=3: Chromatic Dimming & Semantic Weakness**
- **Visuals**: Overall luminance is reduced in both subject and background.
- **Texture**: Whiskers appear "greyed out" or physically thinner. Fur lacks vibrant saturation.
- **Metrics**: Lowest Sharpness ($61.43$) and slightly reduced CLIP alignment. The model lacks sufficient guidance to "push" the pixels into their intended semantic clusters.

**CFG=8: The Strategic Standard**
- **Metrics**: Peak CLIP_subject ($0.3346$).
- **Visuals**: Natural shadow-to-highlight gradients. Optimal balance of "fluffiness" and edge definition.

**CFG=18: Chromatic Burn & Structural Distortion ("The Feathering Effect")**
- **Visuals**: High contrast leads to "crushed" shadows and over-saturated highlights.
- **Artifacts**: The undercoat transitions from "fluffy" to "feathery"—becoming sharper, more stylized, and cartoonish.
- **The CFG Paradox**: Despite the visual "harshness," Sharpness spikes to $163.41$, while CLIP_subject drops to $0.3195$. This confirms that excessive guidance creates "burnt" pixels that deviate from the prompt's semantic intent, resulting in a loss of realism.

## Samplers & Schedulers: Convergence Efficiency
Testing across deterministic and stochastic pathways.
**Sampler Comparison**
- **Euler vs. DPM++ 2M SDE**: Euler exhibits slightly softer, "fluffier" hair edges. DPM++ 2M SDE provides a more grounded, textured feel.
- **Uni_PC**: Visually indistinguishable from Euler in this workflow, but metrics show higher raw Sharpness ($85.66$).

**Scheduler Comparison (Karras, Simple, Exponential)**
- **Visuals**: Zero perceptual delta across all three schedulers for this prompt.
- **Metrics**: CLIP and Sharpness scores remain stable within $\pm 3\%$ variance.
- **Conclusion**: For standard SDXL workloads, the choice of scheduler is secondary to the $N/CFG$ balance.

## Quantitative Matrix: Automated Metrics

| Configuration   | CLIP_subject | CLIP_style  | Sharpness  |
|-----------------|--------------|-------------|------------|
| Steps=15        | 0.29685      | 0.25000     | 195.235*   | 
| Steps=60        | 0.33610      | 0.24140     | 140.105    |
| **CFG=8 (Std)** | **0.33460**  | **0.27160** | **76.470** |
| CFG=3           | 0.32950      | 0.26390     | 61.430     | 
| CFG=18          | 0.31950      | 0.25040     | 163.410*   |

* High sharpness in N=15 and CFG=18 can possibly represent aliasing and "burnt" edges, not textural resolution.

## Performance Supplement (Tesla T4)
Approximate performance for SDXL on T4 hardware

| Step Count (N) | Approx. Latency (sec) | it/s       | Quality Category           |
|----------------|-----------------------|------------|----------------------------|
| 15             | ~16.5s                | ~0.9 - 1   | Draft (Unusable for Final) |
| 30             | ~32.0s                | ~0.8 - 1.5 | Standard (Optimal)         |
| 60             | ~63.0s                | ~0.8 - 1.5 | Premium (Macro/Print)      |

## RAM & Computational Overhead
- **VRAM Footprint**: Stable at ~12.5GB - 13.5GB for SDXL 1024x1024. Step count does not affect peak VRAM usage; it only scales linearly with time.
- **Sampler Interaction**: Using dpmpp_2m_sde with karras scheduler provides faster convergence than standard Euler, but SDE samplers are stochastic and may exhibit minor variations even with a fixed seed if the $N$ delta is extreme.

[//]: # (## Quantitative Analysis: Automated Metrics)

[//]: # (To validate visual observations, I calculated average scores for CLIP Semantic Alignment and Image Sharpness.)

[//]: # (The data reveals a clear trade-off between raw edge contrast and semantic coherence.)

[//]: # ()
[//]: # (| Metric       | N=15    | N=60    | Delta &#40;%&#41; |)

[//]: # (|--------------|---------|---------|-----------|)

[//]: # (| CLIP_subject | 0.29685 | 0.33610 | +13.2%    | )

[//]: # (| CLIP_style   | 0.25000 | 0.24140 | -3.4%     |)

[//]: # (| Sharpness    | 195.235 | 140.105 | -28.2%    |)

[//]: # (Lower Sharpness with higher Steps may lead to lower noise and soft, realistic transition.)

[//]: # (## The Sharpness Paradox)

[//]: # (A counter-intuitive finding is that N=15 shows significantly higher Sharpness scores &#40;$195.2$ vs $140.1$&#41;. In the context of under-converged diffusion models, this is a known artifact:)

[//]: # (- **High-Frequency Noise as "Sharpness"**: At 15 steps, the algorithm interprets unresolved latent noise and "zigzag" aliasing artifacts as high-contrast edges.)

[//]: # (- **Natural Smoothness**: At 60 steps, the model introduces realistic gradients and sub-pixel blending &#40;e.g., soft fur textures, eye moisture&#41;. )

[//]: # (This reduces the raw Laplacian variance &#40;Sharpness score&#41; but increases Perceptual Realism.)

[//]: # (- **Conclusion**: High sharpness at low step counts is a symptom of aliasing, not resolution.)

[//]: # (## Semantic Convergence &#40;CLIP Score&#41;)

[//]: # (The 13.2% increase in CLIP_subject similarity from 15 to 60 steps confirms that the model requires more iterations to fully "sculpt" the features of the cat from the noise. At $N=15$, the subject is semantically "blurry" to the CLIP encoder, even if the edges are sharp.)

[//]: # ()
[//]: # (## Qualitative vs. Quantitative Correlation)

[//]: # (The metrics align perfectly with the "Coloring Book" observation:)

[//]: # (- **Low CLIP_subject &#40;N=15&#41;**: Correlates with the "incomplete" look and lack of volumetric depth. The model hasn't finished "identifying" all features of the subject.)

[//]: # (- **High Sharpness &#40;N=15&#41;**: Correlates with the "harsh outlines" and "whiskers without volume" noted in the visual comparison.)

[//]: # (- **Clip_style Stability**: Confirms that the overall aesthetic &#40;composition, lighting&#41; is locked in early in the denoising process, while the subsequent steps are dedicated to textural refinement and semantic "cleaning.")

## Stability & Confidence Interval
- **Consistency**: Used three seeds for all runs.
- **The "2-of-3" Observation**: While 66% of generations were consistent, 33% (1 out of 3) resulted in compositional outliers (e.g., extreme close-ups).
- **Engineering Verdict**: The pipeline is highly sensitive to the initial latent noise. Future iterations should use **Prompt Weighting** or **ControlNet** to enforce compositional stability across all seeds.

## Tech Recap
[//]: # (- **Production Baseline**: Set $N_{steps} = 30$ as the default. The delta to $N=60$ does not justify a 100% increase in compute cost for most use cases.)
[//]: # (- **Aliasing Mitigation**: Avoid $N < 25$ for close-up portraits. The "zigzag" ocular artifacts indicate that the model has not yet resolved the high-frequency geometry of the iris and pupil.)
[//]: # (- **Future Testing**: Conduct a CFG Sweep &#40;$3.0 \rightarrow 12.0$&#41; at $N=30$ to determine the point of chromatic saturation and "burnt" pixel artifacts.)
1.	**Primary Config**: $N=30, CFG=8, \text{Sampler}=\text{dpmpp\_2m\_sde}, \text{Scheduler}=\text{Karras}$.
2.	**Avoid CFG Clipping**: Do not exceed CFG 12 for realistic subjects; the "feathery" artifacts at CFG 18 indicate a breakdown of natural textures.
3.	**The Sharpness Indicator**: Disregard Sharpness as a standalone quality metric. Target a "Natural Smoothness" range ($70-90$) for realistic animals.

## Appendix
- **Staps** images: steps_15_00001_.png - steps_60_00006_.png
- **CFG** images: cfg_3_00004_.png - crfg_18_00005_.png
- **Sampler** images: euler_00004_.png - uni_pc_00006_.png
- **Scheduler** images: simple_00004_.png - exponential_00006_.png
- **Tunnels**: https://drive.google.com/drive/folders/1ffHlxCxb2_rG8qiy7D0KWGCnAkmL3Ai3?usp=sharing
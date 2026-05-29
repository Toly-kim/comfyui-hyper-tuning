# Title 
## Evaluation of Semantic Leakage and Style Integration in SD1.5
Experiment ID: jedi_van_gogh_1

## Summary
This study evaluates the transition from a baseline prompt (v1) to an optimized, constraint-heavy prompt (v2) within the Stable Diffusion 1.5 architecture.
The goal was to resolve anatomical failures and "Style Leakage" where the subject and background diverged into different visual domains.
- v1 (Baseline): A concise, 38-token prompt introducing the core concept.
- v2 (Optimized): A 95-token "Style-First" prompt designed to force the Post-Impressionist aesthetic onto the foreground subjects, utilizing explicit anatomical and compositional constraints.

Core Objective: To achieve a cohesive, high-fidelity render of a Jedi Knight riding a unicorn in a Martian desert, unified under a Vincent Van Gogh aesthetic.

## Experimental setup
To isolate the impact of prompt modifiers, I utilized a Deterministic Seed-Locked Protocol:

- Model: v1-5-pruned-emaonly.ckpt (SD1.5)
- Resolution: 512x512
- Sampler/Scheduler: Euler / Normal (20 steps)
- Sample Size: 30 image pairs (Seed $n$ in v1 == Seed $n$ in v2)
- Golden Seed Selection: A subset of "high-potential" seeds was identified for deep-dive qualitative analysis to filter out "sterile" or artificially generated noise.

## Methodology: Qualitative Analysis
As this was the initial discovery phase, evaluation was performed via Human-in-the-Loop (HITL) qualitative assessment.
Images were scored based on subjective style consistency and anatomical validity.

No automated metrics (CLIP/LPIPS) were applied at this stage.

### Results & Qualitative Insights
#### Failure Analysis

| Feature              | Observation                                                    | Failure Rate (v1) |
|----------------------|----------------------------------------------------------------|-------------------|
| Anatomical Integrity | Severe artifacts (extra limbs, distorted muzzles)              | ~40%              |
| Foreground Style*    | Subjects defaulted to photorealism, ignoring painterly texture | ~70%              |
| Composition          | Significant cropping of heads and unicorn legs                 | ~20%              |

If FS = 70% is true, that possibly means *prompt engineering alone can only solve ~30%* of "architecture-level features")

#### Subject Fidelity
In the absence of automated CLIP scoring, qualitative analysis reveals a Semantic Tug-of-War.
The 'Jedi' token acts as a high-weight anchor for photorealism,
causing the model to prioritize cinematic textures over the requested Van Gogh style for the foreground character.

Anatomical Coherence: Approximately 40% of generations suffered from 'Liminal Distortion'
(merging of the Jedi’s limb with the unicorn’s torso).

#### Style Inconsistency & Token Conflict
A primary finding was the Semantic Conflict between tokens.
The term "Jedi Knight" acts as a high-weight anchor for photorealistic Sci-Fi imagery in the SD1.5 latent space.
- The Problem: The model effectively partitioned the image—rendering a Van Gogh background but a "movie-still" foreground.
- The v2 Fix: Moving style tokens (Post-Impressionism, impasto) to the front of the prompt helped suppress the photorealistic bias of the Jedi subject.

#### Structural & Perceptual Shift
The transition from v1 to v2 resulted in a Total Compositional Rebuild.
While v1 allowed the model high autonomy (leading to random cropping), v2 introduced strict spatial constraints
By adding "full body visible" and "anatomically correct",
the Structure Score (if we had measured it) would have shown massive variance from v1.

#### Seed Stability
High Intraclass Variance: Even with identical prompts, the 'Style-to-Subject Binding' varied wildly across the 30-seed sample.
This confirms that SD 1.5 requires a high 'Sampling Budget' to find a single technically perfect frame compared to more modern architectures

## Interpretation
In v1, the model treated "Van Gogh" as a background filter.
In v2, by describing the unicorn through the lens of the style ("unicorn is fully painted in brushwork"),
we  bridged the gap between the subject and the aesthetic.

However, I think, the limited parameters of SD1.5 (only 860M) mean that increasing prompt complexity often leads to "concept bleeding",
where the Martian desert starts looking like a typical Earth landscape.

## Next Steps
The limitations of "eye-balling" results became clear.
To scale this workflow, the following measures are required:
- Transition to SDXL: To leverage a larger latent space and better prompt following.
- Mathematical Validation: Implementing a pipeline to measure:
  - CLIP Subject/Style: To verify if v2 actually moved the needle.
  - LPIPS/SSIM: To quantify how much the composition "shook" between versions.
  - Bhattacharyya Distance: color palette integration.

## Appendix
v1 30 images from ComfyUI_00001_.png to ComfyUI_00035_.png 
v2 30 images: ComfyUI_00077_.png - ComfyUI_00106_.png

follow the link:
https://drive.google.com/drive/folders/1ffHlxCxb2_rG8qiy7D0KWGCnAkmL3Ai3?usp=sharing
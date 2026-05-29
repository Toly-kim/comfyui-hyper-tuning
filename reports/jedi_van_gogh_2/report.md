# Title 
## Performance Analysis: SDXL Transition & Semantic Optimization
Experiment ID: jedi_van_gogh_2

## Summary
The jedi_van_gogh_2 study evaluates the migration from SD1.5 to SDXL Base 1.0 to resolve systemic issues with anatomical validity, composition, and canvas framing.

v1 (Baseline): Ported SD1.5 prompt (~38 tokens).

v2 (Optimized): Enhanced for atmospheric depth and emotional resonance (~112 tokens).

Both prompts are in config.yaml (Ctrl-F them).

Core Objective: Eliminate legacy model constraints to generate a high-fidelity Jedi Knight riding a unicorn
through a Martian desert, utilizing a Post-Impressionist (Van Gogh) aesthetic.

## Experimental setup
I used a Deterministic Seed-Locked Protocol.

Model: sd_xl_base_1.0.safetensors

Resolution: 1024x1024

Sampler/Scheduler: DPM++ 2M SDE / Karras (40 steps)

Sample Size: 100 image pairs (Seed $n$ in v1 == Seed $n$ in v2)

## Methodology
Measured the Latent Drift between prompts using a multi-metric suite.

CLIP Subject/Style: Measures semantic alignment between text and pixels.

LPIPS (Perceptual): Quantifies "human-like" visual difference.

SSIM (Structural): Measures layout and geometric retention.

Laplacian Variance: Quantifies edge-frequency (Sharpness).

Bhattacharyya Distance: Measures the shift in the RGB color distribution.

## Results & Analysis

### Subject Fidelity & Anatomical Validity
Comparison: SDXL (v1) vs. Legacy SD1.5 (v2).

This was primarily a benchmark of Model Capacity. SDXL demonstrated a significant leap in spatial reasoning:

Anatomy: Unicorns were rendered with correct equine proportions. "Extra limb" artifacts dropped to 10%, a massive improvement over SD1.5.

Framing: "Head-cropping" issues were eliminated (0% vs. previous high failure rates).

Semantic Accuracy: ~40% of subjects correctly wielded "laser swords." 
However, the model showed a 50% "Unicorn-to-Horse" regression, failing to render the horn when environmental tokens became too dense.

Metrics (v1 vs v2 SDXL):

CLIP Subject v1: 0.3605

CLIP Subject v2: 0.3239

$\Delta$: -0.0366

Comment: Probably, this negative delta is actually a known trade-off.
As you increase prompt complexity (atmosphere/emotions), you increase Token Competition. 
The CLIP encoder's attention is spread thinner, 
leading to a slight drop in specific subject alignment scores even if the image "looks better".

### Style Fidelity (Van Gogh Post-Impressionism)
The Van Gogh aesthetic remained dominant, characterized by high-contrast yellow/blue palettes reminiscent of The Starry Night.

Metrics:
CLIP Style v1: 0.2924

CLIP Style v2: 0.3299

$\Delta$: +0.0375

Comment: This gain may be statistically significant. 
It confirms that atmospheric tokens ("dusty air," "texture") effectively pushed the image further into the aesthetic latent space without breaking the "Van Gogh" classifier.

Note the Primacy Effect. 
Van Gogh tokens placed at the start of the prompt effectively suppressed Sci-Fi elements.
If in CLIP-based models, early tokens hold higher positional weighting, preventing a "Style Conflict", this is understandable.

### Structural / perceptual shift
These metrics measure how much the image moved geometrically when I added the atmosphere tokens.

LPIPS: 0.5522 (High perceptual change)

SSIM: 0.3753 (Low structural retention)

Color Dist: 0.7821 (Massive palette shift)

Comment: An SSIM of 0.37 indicates a Total Compositional Rebuild.

This proves that atmospheric tokens aren't just "adding a filter"—they are fundamentally re-ordering the U-Net's noise-to-image denoising path.
I didn't just change the colors; I moved the Jedi. This is entirely a new image?

### Sharpness
sharpness_v1 = 545.02

sharpness_v1 = 554.26

$\Delta$: +9.24

Comment: This delta looks negligible.
It suggests that while the content changed, the VAE (Variational Autoencoder) decoder may have reached its limit of frequency reconstruction. 
The sharpness is "baked into" the model's output resolution rather than the prompt.

## Final Interpretation & Limitations
Prompt v1 (SDXL) solved the technical "broken limbs" problem, but the images felt sterile—typical "AI mush".

Prompt v2 introduced Human-Centric Quality: the Martian desert now feels tactile, hand-painted, and intentional.

### Limitations
CLIP Proxy Bias: CLIP scores are a "proxy," not an absolute truth.
A lower Subject Score in v2 doesn't mean the Jedi is "worse"—it means the scene is more complex.

Control Limitations: Vanilla SDXL (without ControlNet) still struggles with Foreground-Background Binding.
The model occasionally merges the unicorn's horn with the Jedi's sword.

Metric Isolation: Paired metrics only show the shift between prompts, not the absolute aesthetic value.

## Next Steps: Cinematic Intent
I am satisfied with the background atmosphere. 

The next phase will focus on Subject Legibility:

Semantic Decoupling: Experiment with "Break" keywords or regional prompting to separate the Jedi's anatomy from the desert's "dust" tokens.

Van Gogh Texture Recovery: Re-introduce specific "impasto" and "thick brushstroke" tokens to reclaim the hand-painted feel lost in the SDXL transition.

Anatomy Refinement: Use a low-strength Canny ControlNet to lock the unicorn's structure while iterating on style.

Try to work with prompt to keep an image engaged and emotional, get back Van Gogh style and slightly improving anatomy artifacts.
split foreground/background prompting.

[//]: # (ControlNet / inpaint)
[//]: # (LoRA)
[//]: # (custom ComfyUI node for evaluation or control)

## Appendix
metrics_log.csv - first run with random seeds
100 images from sdxl_00001_.png to sdxl_00100_.png follow the link:
https://drive.google.com/drive/folders/1ffHlxCxb2_rG8qiy7D0KWGCnAkmL3Ai3?usp=sharing

paired_results.csv - second run with the same seeds
sdxl_00104_.png - sdxl_00203_.png available on the same link

metrics_log.csv - calculate matrics clip_sub_v1, clip_sub_v2, clip_sty_v1, clip_sty_v2, lpips, ssim, sharpness_v1, sharpness_v2, color_dist
Image grids glued as one image from two above.
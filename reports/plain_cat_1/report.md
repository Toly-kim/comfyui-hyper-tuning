# Title
Prompt Iteration Evaluation on SDXL Base 1.0

## Summary
Experiment plain_cat_1_1 (Ctrl-F in config.yaml)
Focuing on a simple cosmic cat, because my previuos prompts had complex, different, opposite signals for SDXL Base 1.0.
No background noise. See how tokens like Photorealistic, Sharp focus work.

v1_1. Split prompt into subject & style for accurate calculation of clip similarity & style.
v1_2. Removed background from subject to style.
v2_1. Keeping the same cat, but replacing background. Testing a token rain-slicked neon.
v2_2. Removed background (a token rain-slicked neon) from subject to style.
v3_1. Added weights to emphasize a cat's emotions.

todo: Make a short summary for 5-6 lines. Put it in a nutshell, don't let you bog down in details.

## Experimental setup
model: sd_xl_base_1.0.safetensors

resolution: 1024x1024

[//]: # (sampler: euler)
sampler: dpmpp_2m_sde

[//]: # (scheduler: normal)
scheduler: karras

[//]: # (steps: 20)
steps: 40

paired-seed protocol

prompt_v1 vs prompt_v2 

number of pairs: from 10 to 100

## Methodology
First, generate 10-100 images. Use prompt v1_1 and random seeds.
Second, generate 10-100 images with prompt v2_1 and the same seeds. Same generation settings.
Metrics used to compare:
CLIP Subject, measures subject (of prompt) and image correspondence.
CLIP Style, measures style (of prompt) and image correspondence.
LPIPS
SSIM
sharpness
color_dist

LPIPS, SSIM, color_dist measure how much prompt v2_2 changed the image from prompt v1_1, not the whole quality

## Results
Let's dive into details.

### Subject fidelity
prompt v1_1 did what's expected - a simple cat with single color studio background.
prompt v2_1 also did what's expected - replaced the simple background with neon rooftop. 
I need to coompare numbers.

### Style fidelity

Что стало с Van Gogh / post-impressionism.

5.3 Structural / perceptual shift

Что показали LPIPS, SSIM, color distance.

5.4 Sharpness

Что стало с детализацией / crispness.

6. Interpretation

Один важный раздел. Здесь надо честно сказать:

prompt_v2 улучшил один аспект, но ухудшил другой.
это не “overall improvement”.
это trade-off.
prompt-only control на этой модели достиг потолка.
7. Limitations

Коротко:

CLIP — это proxy, не абсолютная истина.
paired metrics показывают shift between prompts, not full image quality.
vanilla SDXL без LoRA / ControlNet ограничивает control over foreground binding.
8. Next steps

Куда идти дальше:

ControlNet / inpaint
LoRA
split foreground/background prompting
custom ComfyUI node for evaluation or control
9. Appendix

Сюда:

paired image examples
CSV
seeds
code snippets
contact sheet
report figures
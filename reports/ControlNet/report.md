# SDXL + ControlNet Canny Inference Tuning
**Date**: April 17-22, 2026

**Stack**: ComfyUI, SDXL Base 1.0, ControlNet (xinsirControlnetCanny_v20).

**Goal**: Finding the balance between structural control (Canny) and prompt semantic flexibility during object transformation.

## Baseline Configuration
Fixed settings were used to ensure test consistency:
- **SDXL**: Steps: 20, CFG: 8.0, Sampler: Euler.
- **ControlNet**: xinsirControlnetCanny_v20.
- **Starting Point**: Low Threshold: 100, High Threshold: 200, Strength: 0.6, End Percent: 0.45.

## Phase 1: Structural Dominance Test
**"Chimpanzee on a Fish" Case**: High detail in the source image (impasto texture) was found to "lock" the generation process. 
Brushstroke lines are interpreted as hard boundaries, preserving clothing elements and human anatomy even when the prompt is radically changed.

## Phase 2: Object Removal and Addition
After switching to a "clean" reference (minimal noise lines), two scenarios were tested:
- **Removing the Horn**: Successfully achieved using $End\_Percent = 0.35$ and $High\_Threshold = 250$.
The model replaced the horn with a natural white "blaze" on the horse's forehead.
- **Adding a Sword**: The model exhibited "semantic laziness" — instead of drawing a sword in the empty fist, it anchored the prompt to the existing horn line, turning the horn into a sword hilt.
 
## Calibration Rules (End Percent)
- **> 0.5$**: **Structural Dictatorship**. Any discrepancy with the prompt results in artifacts (e.g., the horn becoming a "racket handle").
- **> $\approx 0.35$**: **Sweet Spot**. Large-scale structure is preserved, while small unwanted details are overwritten by the prompt's semantics.
- **$< 0.3$**: **Loss of Control**. The model begins to alter the pose and overall composition.

## Final Conclusion
For efficient inference: object removal is best handled via End Percent (0.35).
To add new elements (like a sword), it is more time-efficient to manually draw a guide line on the Canny map rather than trying to overpower the control with prompt weighting.

## Appendix
- **Chimpanzee on a Fish**: ComfyUI_00118_.png - ComfyUI_00123_.png, control_net_00001_.png - control_net_00010_.png
- **Removing the Horn**: control_net_00018_.png - control_net_00025_.png
- **Adding a Sword**: control_net_00026_.png - control_net_00033_.png
- **Links**: https://drive.google.com/drive/folders/1ffHlxCxb2_rG8qiy7D0KWGCnAkmL3Ai3?usp=sharing
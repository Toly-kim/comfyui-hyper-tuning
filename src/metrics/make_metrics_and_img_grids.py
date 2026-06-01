import os
import re
from metrics import metrics_service

def get_file_info(file_path):
    """
    Разбивает путь на префикс, номер и суффикс.
    Пример: 'C:/.../ComfyUI_00017_.png' -> ('ComfyUI_', 17, '_.png')
    """
    filename = os.path.basename(file_path)
    match = re.search(r'(.*?)(\d+)(.*)', filename)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        suffix = match.group(3)
        return prefix, number, suffix
    return "unknown_", 0, ".png"

def get_extended_metrics(img_path1, prompt1_sub, prompt1_sty, img_path2, prompt2_sub, prompt2_sty):
    s1 = metrics_service.calculate_sharpness(img_path1)
    s2 = metrics_service.calculate_sharpness(img_path2)
    ssim_val = metrics_service.compute_ssim(img_path1, img_path2)
    lpips_val = metrics_service.compute_lpips(img_path1, img_path2)
    color_val = metrics_service.color_hist_distance(img_path1, img_path2)

    # CLIP для V1 (старый результат)
    c_sub_v1 = metrics_service.clip_similarity(img_path1, prompt1_sub)
    c_sty_v1 = metrics_service.clip_similarity(img_path1, prompt1_sty)

    # CLIP для V2 (новый результат)
    c_sub_v2 = metrics_service.clip_similarity(img_path2, prompt2_sub)
    c_sty_v2 = metrics_service.clip_similarity(img_path2, prompt2_sty)

    return {
        "s1": s1, "s2": s2, "ssim": ssim_val, "lpips": lpips_val, "color": color_val,
        "c_sub_v1": c_sub_v1, "c_sty_v1": c_sty_v1,
        "c_sub_v2": c_sub_v2, "c_sty_v2": c_sty_v2
    }

def get_metrics4single_image(img_path1, prompt1_sub, prompt1_sty):
    sharpness = metrics_service.calculate_sharpness(img_path1)
    subject = metrics_service.clip_similarity(img_path1, prompt1_sub)
    style = metrics_service.clip_similarity(img_path1, prompt1_sty)

    return {
        "sharpness": sharpness, "subject": subject, "style": style
    }
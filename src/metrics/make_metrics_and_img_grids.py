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

# def get_metrics_for_pair(path1, path2, prompt):
#     return {
#         "s1": metrics_service.calculate_sharpness(path1),
#         "s2": metrics_service.calculate_sharpness(path2),
#         "ssim": metrics_service.compute_ssim(path1, path2),
#         "lpips": metrics_service.compute_lpips(path1, path2),
#         "clip": metrics_service.clip_similarity(path2, prompt),
#         "hist": metrics_service.color_hist_distance(path1, path2)
#     }

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

# def run_automated_pipeline(v1_paths, v2_paths, prompt, output_csv):
#     """
#     Основной цикл: расчеты -> склейка -> запись в CSV.
#     """
#     header = ["pair_index", "sharpness_v1", "sharpness_v2", "ssim", "lpips_dist",
#               "clip_score_v2", "hist_dist", "img_path", "notes"]
#
#     if not os.path.exists(config.OUTPUT_FOLDER):
#         os.makedirs(config.OUTPUT_FOLDER)
#
#     with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#
#         for i, (p1, p2) in enumerate(zip(v1_paths, v2_paths)):
#             _, num1, _ = get_file_info(p1)
#             _, num2, _ = get_file_info(p2)
#
#             print(f"🔄 Processing pair {i + 1}: Image {num1} vs {num2}")
#
#             try:
#                 # 1. Склеиваем картинки
#                 grid_path = create_grid_for_pair(p1, p2, i + 1, num1, num2)
#
#                 # 2. Считаем метрики
#                 m = get_metrics_for_pair(p1, p2, prompt)
#
#                 # 3. Пишем строку в файл
#                 writer.writerow([
#                     i + 1,
#                     round(m['s1'], 2),
#                     round(m['s2'], 2),
#                     round(m['ssim'], 4),
#                     round(m['lpips'], 4),
#                     round(m['clip'], 4),
#                     round(m['hist'], 4),
#                     grid_path,
#                     ""  # колонка для ваших заметок вручную
#                 ])
#
#                 # Важно для контроля процесса
#                 f.flush()
#
#             except Exception as e:
#                 print(f"❌ Error on pair {i + 1} ({num1} vs {num2}): {e}")
#
#     print(f"✅ Готово! Результаты в файле: {output_csv}")

# def run_automated_pipeline2(v1_paths, v2_paths, subject_prompt, style_prompt, output_csv):
#     """
#     Запуск расширенного анализа с детализацией по смыслу и стилю.
#     """
#     header = [
#         "pair_index", "seed", "v1_path", "v2_path",
#         "clip_subject_v1", "clip_subject_v2",
#         "clip_style_v1", "clip_style_v2",
#         "lpips", "ssim", "sharpness_v1", "sharpness_v2",
#         "color_dist", "notes"
#     ]
#
#     if not os.path.exists(config.OUTPUT_FOLDER):
#         os.makedirs(config.OUTPUT_FOLDER)
#
#     with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#
#         for i, (p1, p2) in enumerate(zip(v1_paths, v2_paths)):
#             _, seed_val, _ = get_file_info(p1)
#             print(f"🔬 Deep Analysis Pair {i+1} | Seed: {seed_val}")
#
#             try:
#                 # 1. Расчет расширенных метрик
#                 m = get_extended_metrics(p1, p2, subject_prompt, style_prompt)
#
#                 # 2. Запись строки согласно новому заголовку
#                 writer.writerow([
#                     i + 1,
#                     seed_val,
#                     p1,
#                     p2,
#                     round(m["c_sub_v1"], 4),
#                     round(m["c_sub_v2"], 4),
#                     round(m["c_sty_v1"], 4),
#                     round(m["c_sty_v2"], 4),
#                     round(m["lpips"], 4),
#                     round(m["ssim"], 4),
#                     round(m["s1"], 2),
#                     round(m["s2"], 2),
#                     round(m["color"], 4),
#                     ""  # notes
#                 ])
#                 f.flush()
#
#             except Exception as e:
#                 print(f"❌ Error at Seed {seed_val}: {e}")
#
#     print(f"📊 Анализ завершен. Файл {output_csv} готов.")
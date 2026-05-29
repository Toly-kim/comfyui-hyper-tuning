from click import prompt

from src.utils.paths import paths

# def get_run_params():
#     cfg = paths.get_cfg()
#
#     paths_cfg = cfg.get('paths', {})
#     runs_cfg = cfg.get('runs', {})
#     model_cfg = cfg.get('model', {})
#
#     active_exp_name = cfg.get('active_experiment')
#     experiments = cfg.get('experiments', {})
#     current_exp = experiments.get(active_exp_name)
#
#     if not current_exp:
#         print(f"⚠️ Warning: Experiment '{active_exp_name}' not found. Using fallback prompts.")
#         # Пытаемся взять из старой секции 'prompts' или отдаем пустые
#         prompts_cfg = cfg.get('prompts', {})
#         current_exp = {
#             "v1_1": prompts_cfg.get('v1_1', ""),
#             "v2_1": prompts_cfg.get('v2_1', "")
#         }
#
#     return {
#         # Пути (имена файлов или полные пути)
#         "csv_v1": paths_cfg.get('csv_file_name'),
#         "csv_paired": paths_cfg.get('paired_results_csv'),
#         "input_drive": paths_cfg.get('input_drive'),
#         "output_grid": paths_cfg.get("output_grid"),
#         "metrics_log_csv": paths_cfg.get("metrics_log_csv"),
#
#         # Настройки прогона
#         "count": runs_cfg.get('count', 0),
#         "wf_name": model_cfg.get('workflow_sdxl_json'),
#
#         # Промпты
#         "p1": current_exp.get('v1_1'),
#         "p2": current_exp.get('v2_1'),
#         "exp_name": active_exp_name
#     }

def get_run_params2():
    cfg = paths.get_cfg()

    paths_cfg = cfg.get('paths', {})
    runs_cfg = cfg.get('runs', {})
    model_cfg = cfg.get('model', {})
    moetrics_cfg = cfg.get('metrics', {})
    prompts_cfg = cfg.get('prompts', {})

    active_exp_name = cfg.get('active_experiment')
    experiments = cfg.get('experiments', {})
    current_exp = experiments.get(active_exp_name)

    weights_cfg = cfg.get('weights', {})

    custom_node_cfg = cfg.get('custom_node', {})

    # Fallback если эксперимент не найден
    if not current_exp:
        print(f"⚠️ Warning: Experiment '{active_exp_name}' not found and stopped")

    def process_prompt(version_data):
        if not isinstance(version_data, dict):
            # Если это строка, None или что-то еще — трактуем как строку
            val = version_data or ""
            return val, val, ""

        # subj = version_data.get('subject', "")
        # sty = version_data.get('style', "")
        # full = version_data.get('text', f"{subj}, {sty}".strip(". "))

        subj = version_data.get('subject', "").strip()
        sty = version_data.get('style', "").strip()

        if 'text' in version_data:
            full = version_data['text']
        else:
            clean_subj = subj.rstrip(".")
            full = ". ".join(filter(None, [clean_subj, sty]))

            if full and not full.endswith("."):
                full += "."

        return full, subj, sty

    p1_full, p1_sub, p1_sty = process_prompt(current_exp.get('v1', ""))
    p2_full, p2_sub, p2_sty = process_prompt(current_exp.get('v2', ""))

    return {
        "csv_v1": paths_cfg.get('csv_file_name'),
        "csv_paired": paths_cfg.get('paired_results_csv'),
        "input_drive": paths_cfg.get('input_drive'),
        "output_grid": paths_cfg.get("output_grid"),
        "metrics_log_csv": paths_cfg.get("metrics_log_csv"),
        "test_metrics_log_csv": paths_cfg.get("test_metrics_log_csv"),
        "summary_stats_csv": paths_cfg.get("summary_stats_csv"),
        "test_paired_results": paths_cfg.get("test_paired_results"),
        "all_prompts_csv": paths_cfg.get("all_prompts_csv"),
        "all_prompts_score": paths_cfg.get("all_prompts_score"),
        "select_best_output": paths_cfg.get("select_best_output"),
        "prompt_ranking": paths_cfg.get("prompt_ranking"),

        # Настройки
        "count": runs_cfg.get('count', 0),
        "wf_name": model_cfg.get('workflow_sdxl_json'),
        "wf_multi_prompt": model_cfg.get('wf_multi_prompt'),

        # Промпты для генерации (Full)
        "p1": p1_full,
        "p2": p2_full,

        # Метаданные для метрик (Sub/Sty)
        "p1_sub": p1_sub,
        "p1_sty": p1_sty,
        "p2_sub": p2_sub,
        "p2_sty": p2_sty,

        # Версии (для лога)
        "p1_ver": "v1",
        "p2_ver": "v2",
        "exp_name": active_exp_name,
        "current_exp": current_exp,

        # Negative prompt
        "negative": prompts_cfg.get('negative'),

        # Metrics
        "delta_mean_threshold": moetrics_cfg.get('delta_mean_threshold'),
        "improved_pct_threshold": moetrics_cfg.get('improved_pct_threshold'),

        # Filename prefix
        "file_name_pref": active_exp_name or "active_experiment",

        # weights, the node
        "weights": weights_cfg,

        # weights item by item
        "subject_clip": weights_cfg.get('subject_clip'),
        "style_clip": weights_cfg.get('style_clip'),
        "ssim": weights_cfg.get('ssim'),
        "lpips_inv": weights_cfg.get('lpips_inv'),
        "sharpness": weights_cfg.get('sharpness'),

        # Custom node
        "test_json": custom_node_cfg.get('test_json'),
        "test_wf": custom_node_cfg.get('test_wf'),
        "test_image": custom_node_cfg.get('test_image'),
        "test_img_description": custom_node_cfg.get('test_img_description')
    }
import csv
import os
from collections import defaultdict
from datetime import datetime, timezone

from scipy.stats import rankdata

from metrics.make_metrics_and_img_grids import get_extended_metrics
from metrics.make_metrics_and_img_grids import get_metrics4single_image
from utils.config_params import get_run_params2
from visual.create_img_grids import create_grid_for_pair


def get_seed_category(seed, test_suites):
    if seed in test_suites.get('best_seeds', []):
        return 'best'
    elif seed in test_suites.get('half_unicorn_seeds', []):
        return 'half_unicorn'
    elif seed in test_suites.get('two_knights_seeds', []):
        return 'two_knights'
    else:
        return 'other'

# Sends to Comfy, usually a first version of prompt
# Creates runs_log.csv
# run_id,seed,img_v1,img_v2,
# clip_sub_v1,clip_sub_v2,
# clip_sty_v1,clip_sty_v2,
# lpips,
# ssim,
# sharpness_v1,
# sharpness_v2,
# color_dist
def run_metrics():
    p = get_run_params2()
    base_drive_path = p['input_drive']
    input_csv = p['csv_paired']
    output_csv = p['metrics_log_csv']
    output_dir = p['output_grid']

    # Промпты для расчета CLIP (уже выбранные из активного эксперимента)
    prompt_v1_sub = p['p1_sub']
    prompt_v1_sty = p['p1_sty']
    prompt_v2_sub = p['p2_sub']
    prompt_v2_sty = p['p2_sty']

    experiment_name = p['exp_name']

    print(f"🚀 Запуск анализа эксперимента: {experiment_name}")
    print(f"📄 Reading pairs from: {input_csv}")
    print(f"📂 Looking for images on: {base_drive_path}")

    if not input_csv or not os.path.exists(input_csv):
        print(f"❌ Ошибка: Файл не найден: {input_csv}")
        return

    pairs_to_process = []
    with open(input_csv, mode='r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            pairs_to_process.append(row)

    if not pairs_to_process:
        print("⚠️ No data for analysis in file")
        return

    header = [
        "run_id", "seed", "img_v1", "img_v2",
        "clip_sub_v1", "clip_sub_v2", "clip_sty_v1", "clip_sty_v2",
        "lpips", "ssim", "sharpness_v1", "sharpness_v2", "color_dist"
    ]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for i, row in enumerate(pairs_to_process):
            seed = row['seed']

            filename1 = row['img_v1_filename'] # from paired_results.csv
            filename2 = row['img_v2_filename']
            path1 = os.path.join(base_drive_path, filename1)
            path2 = os.path.join(base_drive_path, filename2)

            if not os.path.exists(path1) or not os.path.exists(path2):
                print(f"⚠️ Missing run_id {row['run_id']}: files not found ({path1} or {path2})")
                continue

            print(f"🧪 [{i + 1}/{len(pairs_to_process)}] Анализ пары | Seed: {seed}")

            try:
                m = get_extended_metrics(
                    path1, prompt_v1_sub, prompt_v1_sty,
                    path2, prompt_v2_sub, prompt_v2_sty
                )
                pair_id_seed = f"{i+1}_{seed}"

                grid_path = create_grid_for_pair(
                    path1, path2,
                    pair_id_seed,
                    filename1,
                    filename2
                )

                writer.writerow([
                    row['run_id'], seed, path1, path2,
                    round(m.get("c_sub_v1", 0), 4), round(m.get("c_sub_v2", 0), 4),
                    round(m.get("c_sty_v1", 0), 4), round(m.get("c_sty_v2", 0), 4),
                    round(m.get("lpips", 0), 4), round(m.get("ssim", 0), 4),
                    round(m.get("s1", 0), 2), round(m.get("s2", 0), 2),
                    round(m.get("color", 0), 4)
                ])
                f_out.flush()

            except Exception as e:
                print(f"❌ Error when processing a pair of {seed}: {e}")

    print(f"🏁 Done! Metrics and Grids have been saved")
    print(f"📊 Итоговый лог: {output_csv}")

def run_test_identity_check():
    p = get_run_params2()
    base_drive_path = p['input_drive']
    input_csv = p['csv_paired']
    output_csv = p['test_metrics_log_csv']
    # не бум клеить
    # output_dir = p['output_grid']

    prompt_v1_sub = p['p1_sub']
    prompt_v1_sty = p['p1_sty']
    prompt_v2_sub = p['p2_sub']
    prompt_v2_sty = p['p2_sty']

    experiment_name = p['exp_name']

    print(f"🚀 Testing experiment: {experiment_name}")
    print(f"📂 Looking for images on: {base_drive_path}")

    if not input_csv or not os.path.exists(input_csv):
        print(f"❌ Ошибка: Файл не найден: {input_csv}")
        return

    pairs_to_process = []
    with open(input_csv, mode='r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            pairs_to_process.append(row)

    if not pairs_to_process:
        print("⚠️ No data for analysis in file")
        return

    header = [
        "run_id", "seed", "img_v1",
        "clip_sub_v1",
        "clip_sub_v1",
        "CLIP Subject Delta",
        "clip_sty_v1",
        "clip_sty_v1",
        "CLIP Style Delta",
        "lpips", "ssim",
        "sharpness_v1",
        "sharpness_v1",
        "Sharpness Delta",
        "color_dist"
    ]

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for i, row in enumerate(pairs_to_process):
            seed = row['seed']

            filename1 = row['img_v1_filename'] # from paired_results.csv
            filename2 = row['img_v2_filename']
            path1 = os.path.join(base_drive_path, filename1)
            path2 = os.path.join(base_drive_path, filename2)

            if not os.path.exists(path1):
                print(f"⚠️ Missing run_id {row['run_id']}: file not found ({path1})")
                continue

            print(f"🧪 [{i + 1}/{len(pairs_to_process)}] Анализ | Seed: {seed}")

            try:
                m = get_extended_metrics(
                    path1, prompt_v1_sub, prompt_v1_sty,
                    path1, prompt_v1_sub, prompt_v1_sty
                    # path2, prompt_v2_sub, prompt_v2_sty,
                    # path2, prompt_v2_sub, prompt_v2_sty
                )

                # header = [
                #     "run_id", "seed", "img_v1",
                #     "clip_sub_v1",
                #     "clip_sub_v1",
                #     "CLIP Subject Delta",
                #     "clip_sty_v1",
                #     "clip_sty_v1",
                #     "CLIP Style Delta",
                #     "lpips", "ssim",
                #     "sharpness_v1",
                #     "sharpness_v1",
                #     "Sharpness Delta",
                #     "color_dist"
                # ]

                writer.writerow([
                    row['run_id'], seed, path1,
                    round(m.get("c_sub_v1", 0), 4),
                    round(m.get("c_sub_v2", 0), 4),
                    round(m.get("c_sub_v2", 0), 4)-round(m.get("c_sub_v1", 0), 4),
                    round(m.get("c_sty_v1", 0), 4),
                    round(m.get("c_sty_v2", 0), 4),
                    round(m.get("c_sty_v2", 0), 4)-round(m.get("c_sty_v1", 0), 4),
                    round(m.get("lpips", 0), 4), round(m.get("ssim", 0), 4),
                    round(m.get("s1", 0), 2),
                    round(m.get("s2", 0), 2),
                    round(m.get("s2", 0), 2)-round(m.get("s1", 0), 2),
                    round(m.get("color", 0), 4)
                ])
                f_out.flush()

            except Exception as e:
                print(f"❌ Error when processing a {seed}: {e}")

    print(f"🏁 Done! Test metrics have been saved {output_csv}")

def compute_metrics():
    p = get_run_params2()
    weights = p['weights']
    base_drive_path = p['input_drive']
    input_csv = p['all_prompts_csv']
    output_csv = p['all_prompts_score']
    output_dir = p['output_grid']

    print(f"📄 Reading from: {input_csv}")

    if not input_csv or not os.path.exists(input_csv):
        print(f"❌File not found: {input_csv}")
        return

    pairs_to_process = []

    with open(input_csv, mode='r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            pairs_to_process.append(row)

    if not pairs_to_process:
        print("⚠️ No data for analysis in file")
        return

    all_rows = []

    for row in pairs_to_process:
        batch_id = row.get('batch_id', 'no_batch')
        seed = row['seed']
        prompt_version = row['prompt_version']
        test = row['test']
        filename = row['file_name']
        path = os.path.join(base_drive_path, filename)
        prompt_subject = row['prompt_subject']
        prompt_style = row['prompt_style']

        if not os.path.exists(path):
            print(f"⚠️ File {path} not found")
            continue

        try:
            m = get_metrics4single_image(path, prompt_subject, prompt_style)

            all_rows.append({
                "batch_id": batch_id,
                "seed": seed,
                "filename": filename,
                "prompt_version": prompt_version,
                "test": test,
                "subject": m.get("subject", 0),
                "style": m.get("style", 0),
                "sharpness": m.get("sharpness", 0)
            })

        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

    if not all_rows:
        print("⚠️ No valid rows after processing")
        return

    groups = defaultdict(list)
    print(f"Groups: {len(groups)} (batch, seed pairs)")

    for row in all_rows:
        group_key = (row["batch_id"], row["seed"])
        groups[row["seed"]].append(row)

    # for seed, rows in groups.items():
    for (batch_id, seed), rows in groups.items():
        subject_vals = [r["subject"] for r in rows]
        style_vals = [r["style"] for r in rows]
        sharp_vals = [r["sharpness"] for r in rows]

        subject_norm = normalize_metrics(subject_vals)
        style_norm = normalize_metrics(style_vals)
        sharpness_norm = normalize_metrics(sharp_vals)

        for i, r in enumerate(rows):
            r["subject_norm"] = subject_norm[i]
            r["style_norm"] = style_norm[i]
            r["sharpness_norm"] = sharpness_norm[i]

    header = [
    "batch_id",
        "seed",
        "file_name", "prompt_version", "test",
        "subject_clip_raw", "subject_clip_norm",
        "style_clip_raw", "style_clip_norm",
        "sharpness_raw", "sharpness_norm",
        "single_image_score"
    ]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for row in all_rows:
            # --- sharpness guard ---
            if row["subject_norm"] < 0.2:
                sharp_w = 0
            else:
                sharp_w = weights.get("sharpness", 0)

            score = (
                weights.get("subject_clip", 0) * row["subject_norm"] +
                weights.get("style_clip", 0) * row["style_norm"] +
                sharp_w * row["sharpness_norm"]
            )

            writer.writerow([
                row["batch_id"],
                row["seed"],
                row["filename"],
                row["prompt_version"],
                row["test"],
                round(row["subject"], 4),
                round(row["subject_norm"], 4),
                round(row["style"], 4),
                round(row["style_norm"], 4),
                round(row["sharpness"], 4),
                round(row["sharpness_norm"], 4),
                round(score, 4)
            ])

        f_out.flush()

    print(f"🏁 Done. Output: {output_csv}")

def compute_score(metrics: dict, weights: dict) -> float:
    # lpips = metrics.get("lpips")
    # lpips_inv = 1 - lpips if lpips is not None else 0

    sharpness = metrics.get("sharpness", 0)
    sharpness_norm = min(sharpness / 100.0, 1.0)

    score = (
            weights.get("subject_clip", 0) * metrics.get("subject", 0) +
            weights.get("style_clip", 0) * metrics.get("style", 0) +
        # weights.get("ssim", 0) * metrics.get("ssim", 0) +
        # weights.get("lpips_inv", 0) * lpips_inv +
            weights.get("sharpness", 0) * sharpness_norm
    )

    total_w = sum(weights.values())
    if total_w > 0:
        score /= total_w

    return score

def normalize_metrics(indicators: list) -> list:
    if not indicators:
        return []

    ranks = rankdata(indicators, method="average")
    n = len(indicators)

    return [(r - 1) / (n - 1) if n > 1 else 0.0 for r in ranks]

def select_best_output():
    p = get_run_params2()
    input_csv = p['all_prompts_score']
    output_csv = p['select_best_output']
    output_dir = p['output_grid']

    # suppose only top1
    top_k = 1

    grouped = defaultdict(list)

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                row["single_image_score"] = float(row.get("single_image_score", 0))
            except ValueError:
                row["single_image_score"] = 0.0

            batch_id = row.get("batch_id", "no_batch")
            group_key = (batch_id, row["seed"])

            grouped[row["seed"]].append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "batch_id",
            "seed",
            "rank",
            "file_name",
            "prompt_version",
            "test",
            "single_image_score",
            "subject_clip_norm",
            "style_clip_norm",
            "sharpness_norm",
            "timestamp"
        ])

        for (batch_id, seed), rows in grouped.items():
            # sort descending
            sorted_rows = sorted(
                rows,
                key=lambda x: x["single_image_score"],
                reverse=True
            )

            # take top_k
            top_rows = sorted_rows[:top_k]

            for rank, row in enumerate(top_rows, 1):
                writer.writerow([
                    batch_id,
                    seed,
                    rank,
                    row.get("file_name"),
                    row.get("prompt_version"),
                    row.get("test"),
                    row.get("single_image_score"),
                    row.get("subject_clip_norm"),
                    row.get("style_clip_norm"),
                    row.get("sharpness_norm"),
                    datetime.now(timezone.utc).isoformat()
                ])

        f.flush()

    print(f"🏁 Done. Best outputs saved to: {output_csv}")

def aggregate_best_by_prompt_version():
    p = get_run_params2()
    input_csv = p['select_best_output']
    output_csv = p['prompt_ranking']

    stats = defaultdict(lambda: {
        "wins": 0,
        "score_sum": 0.0
    })

    total_seeds = set()

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seed = row.get("seed")
            version = row.get("prompt_version")

            try:
                score = float(row.get("single_image_score", 0))
            except ValueError:
                score = 0.0

            total_seeds.add(seed)

            stats[version]["wins"] += 1
            stats[version]["score_sum"] += score

    total_n = len(total_seeds)

    # --- 2. WRITE ---
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "prompt_version",
            "wins",
            "win_rate",
            "avg_score",
            "final_score"
        ])

        results = []

        for version, data in stats.items():
            wins = data["wins"]
            avg_score = data["score_sum"] / wins if wins > 0 else 0
            win_rate = wins / total_n if total_n > 0 else 0

            # --- simple combined metric ---
            final_score = 0.6 * win_rate + 0.4 * avg_score

            results.append([
                version,
                wins,
                win_rate,
                avg_score,
                final_score
            ])

        # sort by final_score
        results.sort(key=lambda x: x[4], reverse=True)

        for row in results:
            writer.writerow(row)

        f.flush()

    print(f"🏁 Done. Aggregated results: {output_csv}")

# Main execution
if __name__ == "__main__":
    run_metrics() # paired_results.csv
import csv
import json
import os
import re
from datetime import datetime
from collections import defaultdict
import numpy as np
from PIL import Image
from . import metrics_calc

PROMPTS_SCORE_CSV = "all_prompts_score.csv"
RANKING_CSV = "prompt_ranking.csv"
BEST_OUTPUT_CSV = "select_best_output.csv"

COMFY_OUT = "output"

class MultiMetricScorer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "subject_list": ("LIST",),
                "style_list": ("LIST",),
                "version_list": ("LIST",),
                "test_list": ("LIST",),
                "experiment_name": ("STRING", {"default": "default_exp"}),
                "log_to_csv": (["enable", "disable"], {"default": "enable"}),
            },
            "optional": {
                "iteration_list": ("LIST",),
                "seed_list": ("LIST",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "score_batch"
    CATEGORY = "custom/metrics"
    OUTPUT_NODE = True

    # --- ENVIRONMENT & CONFIG SUBSYSTEM ---
    def resolve_output_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        comfy_root = os.path.dirname(parent_dir)
        output_dir = os.path.join(comfy_root, COMFY_OUT)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _load_weights(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        default_weights = {"subject": 0.5, "style": 0.5, "sharpness": 0.0}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("weights", default_weights)
            except Exception as e:
                print(f"⚠️ [METRICS] Failed to load config.json: {e}")
        return default_weights

    def sanitize_filename(self, name: str) -> str:
        return re.sub(r'[\\/*?:"<>| ]', "_", str(name)).strip("_")

    # --- STORAGE METRICS I/O SUBSYSTEM ---
    def _load_existing_csv(self, file_path: str) -> list:
        if not os.path.exists(file_path):
            return []
        records = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
        except Exception as e:
            print(f"⚠️ [METRICS] Error reading historical CSV: {e}")
        return records

    def _write_csv(self, file_path: str, records: list):
        if not records:
            return
        try:
            headers = [k for k in records[0].keys() if k != "image_tensor"]
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in records:
                    row_data = {k: v for k, v in r.items() if k != "image_tensor"}
                    writer.writerow(row_data)
        except Exception as e:
            print(f"❌ [METRICS] Failed to write CSV file {file_path}: {e}")

    def _save_image_to_disk(self, tensor, output_dir, filename):
        full_image_path = os.path.join(output_dir, filename)
        try:
            img_np = tensor.cpu().numpy()
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np).convert("RGB").save(full_image_path, quality=95)
        except Exception as e:
            print(f"⚠️ [METRICS] Failed to save image {filename}: {e}")

    def _purge_duplicate_iterations(self, historical_records, experiment_name, active_iterations):
        filtered_historical = []
        for r in historical_records:
            try:
                hist_iter = int(r.get("iteration", 1))
                if r.get("experiment") == experiment_name and hist_iter in active_iterations:
                    continue
            except (ValueError, TypeError):
                pass
            filtered_historical.append(r)
        return filtered_historical

    def _calculate_raw_metrics(self, i, image_tensor, subject_list, style_list, version_list, test_list,
                               clean_iterations, clean_seeds, experiment_name):
        subject = subject_list[i] if i < len(subject_list) else ""
        style = style_list[i] if i < len(style_list) else ""
        version = version_list[i] if i < len(version_list) else f"v_{i}"
        test_type = test_list[i] if i < len(test_list) else "unknown"

        iteration_number = clean_iterations[i] if i < len(clean_iterations) else 1
        seed = clean_seeds[i] if i < len(clean_seeds) else metrics_calc.generate_seed()

        raw_sharpness = metrics_calc.calc_sharpness(image_tensor)
        raw_clip_subject = metrics_calc.calc_clip_similarity(image_tensor, str(subject).strip())
        raw_clip_style = metrics_calc.calc_clip_similarity(image_tensor, str(style).strip())

        return {
            "experiment": experiment_name,
            "iteration": iteration_number,
            "version": version,
            "test": test_type,
            "seed": seed,
            "filename": f"run_{iteration_number}_{self.sanitize_filename(version)}_{self.sanitize_filename(test_type)}_{seed}.png",
            "image_tensor": image_tensor,
            "raw_sharpness": raw_sharpness,
            "norm_sharpness": 0.0,
            "raw_subject": raw_clip_subject,
            "norm_subject": 0.0,
            "raw_style": raw_clip_style,
            "norm_style": 0.0,
            "score": 0.0
        }

    def _apply_global_normalization(self, records, weights):
        sharpness_list = [float(r["raw_sharpness"]) for r in records]
        subject_list_clip = [float(r["raw_subject"]) for r in records]
        style_list_clip = [float(r["raw_style"]) for r in records]

        norm_sharpness_list = metrics_calc.normalize_metrics(sharpness_list)
        norm_subject_list = metrics_calc.normalize_metrics(subject_list_clip)
        norm_style_list = metrics_calc.normalize_metrics(style_list_clip)

        for i in range(len(records)):
            record = records[i]
            record["norm_sharpness"] = round(norm_sharpness_list[i], 4)
            record["norm_subject"] = round(norm_subject_list[i], 4)
            record["norm_style"] = round(norm_style_list[i], 4)

            weighted_score = (
                    (record["norm_subject"] * weights.get("subject", 0.6)) +
                    (record["norm_style"] * weights.get("style", 0.2)) +
                    (record["norm_sharpness"] * weights.get("sharpness", 0.1))
            )
            record["score"] = round(float(weighted_score), 4)

    # --- PIPELINE ROUTING ENTRYPOINT ---
    def score_batch(self, images, subject_list, style_list, version_list, test_list, experiment_name, log_to_csv,
                    iteration_list=None, seed_list=None):

        comfy_ui_output = self.resolve_output_path()
        if "{" in experiment_name or len(experiment_name) > 100:
            experiment_name = f"fallback_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir = os.path.join(comfy_ui_output, experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        weights = self._load_weights()
        csv_file_path = os.path.join(output_dir, PROMPTS_SCORE_CSV)
        historical_records = self._load_existing_csv(csv_file_path)

        clean_iterations = [int(x) for x in iteration_list] if iteration_list else []
        clean_seeds = [int(x) for x in seed_list] if seed_list else []

        incoming_raw_results = []
        batch_size = images.shape[0]
        print(f"\n📊 [PIPELINE] Processing macro-batch size: {batch_size} incoming images...")

        # Process each item in the batch sequentially
        for i in range(batch_size):
            record = self._calculate_raw_metrics(
                i, images[i], subject_list, style_list, version_list, test_list,
                clean_iterations, clean_seeds, experiment_name
            )
            incoming_raw_results.append(record)

            print(f"📈 [METRIC REPORT] Frame {i+1} ({record['version']}) -> "
                  f"CLIP Subj: {round(record['raw_subject'], 4)} | "
                  f"CLIP Style: {round(record['raw_style'], 4)} | "
                  f"Sharpness: {round(record['raw_sharpness'], 2)}")

            self._save_image_to_disk(record["image_tensor"], output_dir, record["filename"])

        # Merge history with active step frames safely
        active_iterations = set(r["iteration"] for r in incoming_raw_results)
        filtered_historical = self._purge_duplicate_iterations(historical_records, experiment_name, active_iterations)
        combined_records = filtered_historical + incoming_raw_results

        # Calculate cross-matrix normalization weights
        self._apply_global_normalization(combined_records, weights)

        # Atomic CSV logging output routing
        if log_to_csv == "enable":
            self._write_csv(csv_file_path, combined_records)

            # Restrict ranking operations strictly to the scope of the active experiment
            active_experiment_records = [r for r in combined_records if r.get("experiment") == experiment_name]

            best_outputs = self._pipeline_select_best_output(active_experiment_records)
            self._write_csv(os.path.join(output_dir, BEST_OUTPUT_CSV), best_outputs)

            prompt_ranking = self._pipeline_aggregate_rankings(best_outputs)
            self._write_csv(os.path.join(output_dir, RANKING_CSV), prompt_ranking)

        print(f"✅ [PIPELINE] Completed matrix normalization. Active rows logged: {len(combined_records)}\n")
        return (0.0,)

    # --- STATISTICAL ANALYSIS AGGREGATORS ---
    def _pipeline_select_best_output(self, records: list) -> list:
        grouped = {}
        for r in records:
            iter_val = int(r["iteration"]) if isinstance(r["iteration"], int) else int(float(r["iteration"]))
            key = (iter_val, r["version"])
            if key not in grouped or float(r["score"]) > float(grouped[key]["score"]):
                grouped[key] = r
        return sorted(list(grouped.values()), key=lambda x: (int(x["iteration"]), x["version"]))

    def _pipeline_aggregate_rankings(self, best_records: list) -> list:
        iteration_battles = defaultdict(list)
        version_metadata = {}

        for r in best_records:
            iter_idx = int(r["iteration"])
            iteration_battles[iter_idx].append(r)
            version_metadata[r["version"]] = r.get("test", "unknown")

        total_iterations = len(iteration_battles)
        stats = defaultdict(lambda: {"wins": 0, "score_sum": 0.0})

        # Calculate wins based on head-to-head tournament matchups per iteration split
        for iter_idx, records in iteration_battles.items():
            if not records:
                continue

            max_score = max(float(r["score"]) for r in records)
            for r in records:
                v = r["version"]
                score = float(r["score"])

                stats[v]["score_sum"] += score
                if score == max_score and max_score > 0.0:
                    stats[v]["wins"] += 1

        # Build clean tournament output dictionary rows
        aggregation = []
        for v, data in stats.items():
            wins = data["wins"]
            score_sum = data["score_sum"]

            avg_score = score_sum / total_iterations if total_iterations > 0 else 0.0
            win_rate = wins / total_iterations if total_iterations > 0 else 0.0
            final_score = round((0.6 * win_rate) + (0.4 * avg_score), 4)

            aggregation.append({
                "version": v,
                "test": version_metadata.get(v, "unknown"),
                "wins": wins,
                "win_rate": round(win_rate, 4),
                "avg_score": round(avg_score, 4),
                "final_score": final_score,
                "rank": 0
            })

        return self._compute_dense_ranks(aggregation)

    def _compute_dense_ranks(self, aggregation: list) -> list:
        scores_array = [x["final_score"] for x in aggregation]
        final_ranks = metrics_calc.rank_metrics(scores_array)

        for i in range(len(aggregation)):
            aggregation[i]["rank"] = final_ranks[i]

        return sorted(aggregation, key=lambda x: x["rank"])

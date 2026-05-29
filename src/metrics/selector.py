import csv
import os
from copy import deepcopy
from datetime import datetime, timezone

from core.call_api import queue_prompt, wait_for_filename
from src.utils.config_params import get_run_params2
from src.utils.paths import paths
from src.utils.utils import generate_seed
from workflow_api import wf_transformer
from workflow_api.wf_transformer import find_nodes

def prepare_prompt_versions(current_exp: dict) -> dict:
    prompt_versions = {}

    for version, node in current_exp.items():
        if not isinstance(node, dict):
            continue
        if not version.startswith("v"):
            continue

        test_name = node.get("test", "")
        subject = node.get("subject", "")
        style = node.get("style", "")

        if not subject.strip():
            continue

        prompt_text = f"{subject}. {style}".strip()

        prompt_versions[version] = {
            "prompt": prompt_text,
            "test": test_name,
            "subject": subject,
            "style": style,
        }

    return prompt_versions

def run_single_seed_for_all_prompts(seed_value: int, batch_id: str):
    cfg = paths.get_cfg()
    p = get_run_params2()

    all_prompts_csv = p["all_prompts_csv"]
    negative = p["negative"]
    wf_name = p["wf_name"]
    file_name_pref = p["file_name_pref"]
    current_exp = p["current_exp"]

    prompt_versions = prepare_prompt_versions(current_exp)

    if not prompt_versions:
        print("❌ No valid prompt versions found")
        return

    total = len(prompt_versions)

    base_workflow = wf_transformer.load_workflow_by_filename(wf_name)
    ks_id, pos_id, neg_id = find_nodes(base_workflow)

    file_exists = os.path.exists(all_prompts_csv)

    with open(all_prompts_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "batch_id",
                "seed",
                "prompt_version",
                "test",
                "file_name",
                "timestamp",
                "prompt",
                "prompt_subject",
                "prompt_style",
            ])

        for i, (version, data) in enumerate(prompt_versions.items(), 1):
            prompt_text = data["prompt"]
            test_name = data["test"]
            prompt_subject = data["subject"]
            prompt_style = data["style"]

            wf = deepcopy(base_workflow)

            wf[ks_id]["inputs"]["seed"] = seed_value
            wf[pos_id]["inputs"]["text"] = prompt_text
            wf[neg_id]["inputs"]["text"] = negative

            unique_prefix = f"{file_name_pref}_{batch_id}_{seed_value}_{version}"
            wf[neg_id]["inputs"]["filename_prefix"] = unique_prefix

            prompt_id = queue_prompt(wf, cfg)

            if not prompt_id:
                print(f"❌ [{i}/{total}] {version} API Error")
                continue

            print(f"⏳ [{i}/{total}] {version} | seed={seed_value} | batch={batch_id}")
            real_filename = wait_for_filename(prompt_id, cfg)

            if not real_filename:
                real_filename = "TIMEOUT_ERROR"
                print(f"⚠️ [{i}/{total}] {version} Timeout")
                continue

            writer.writerow([
                batch_id,
                seed_value,
                version,
                test_name,
                real_filename,
                datetime.now(timezone.utc).isoformat(),
                prompt_text,
                prompt_subject,
                prompt_style,
            ])
            f.flush()

            print(f"✨ saved: {real_filename}")
    print(f"🏁 Seed done: {seed_value}")

def run_multi_seed_for_all_prompts():
    # cfg = paths.get_cfg()
    p = get_run_params2()
    n_seeds = p['count']

    for i in range(1, n_seeds + 1):
        seed_value = generate_seed()
        print(f"\n=== Seed run {i}/{n_seeds} | seed={seed_value} ===")
        run_single_seed_for_all_prompts(seed_value)
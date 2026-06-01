import csv
import os
import time
from copy import deepcopy
from datetime import datetime, timezone

from core.call_api import queue_prompt, wait_for_filename
from src.utils.config_params import get_run_params2
from src.utils.paths import paths
from src.utils.utils import generate_seed
from workflow_api import wf_transformer
from workflow_api.wf_transformer import find_nodes

def run_random_seeds():
    cfg = paths.get_cfg()
    p = get_run_params2()
    csv_file_path = p['csv_v1']
    runs_count = p['count']
    prompt_v1 = p['p1']
    negative_prompt = p['negative']

    wf_name = p['wf_name']
    base_workflow = wf_transformer.load_workflow_by_filename(wf_name)

    ks_id, pos_id, neg_id = find_nodes(base_workflow)

    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "seed", "prompt_version", "filename", "timestamp"])

        for i in range(1, runs_count + 1):
            wf = deepcopy(base_workflow)
            seed = generate_seed()

            wf[ks_id]["inputs"]["seed"] = seed
            wf[pos_id]["inputs"]["text"] = prompt_v1
            wf[neg_id]["inputs"]["text"] = negative_prompt

            prompt_id = queue_prompt(wf, cfg)

            if not prompt_id:
                print(f"❌ [{i}/{runs_count}] Ошибка очереди. Пропуск...")
                continue

            print(f"⏳ [{i}/{runs_count}] Seed: {seed} | В очереди...")

            real_filename = wait_for_filename(prompt_id, cfg)

            if real_filename:
                print(f"✨ [{i}/{runs_count}] Готово -> {real_filename}")
            else:
                real_filename = "TIMEOUT_ERROR"
                print(f"⚠️ [{i}/{runs_count}] Превышено время ожидания для ID: {prompt_id}")

            # Логирование
            writer.writerow([
                i,
                seed,
                "V1_SDXL",
                real_filename,
                datetime.now(timezone.utc).isoformat()
            ])
            f.flush()

    print(f"🏁 Генерация завершена. Лог сохранен в {csv_file_path}")

# sends to Comfy another version of prompt
# Creates paired_results.csv
# run_id,seed,img_v1_filename,prompt_version,img_v2_filename,timestamp
def run_same_seed_another_prompt():
    cfg = paths.get_cfg()
    p = get_run_params2()
    csv_v1_path = p['csv_v1']
    paired_results_csv = p['csv_paired']
    runs_count = p['count']
    # prompt_v1 = p['p1']
    prompt_v2 = p['p2']
    negative = p['negative']
    wf_name = p['wf_name']
    file_name_pref = p['file_name_pref']

    print(f"📖 Reading seeds from {csv_v1_path}...")
    valid_data_v1 = [] # Будем хранить кортежи (seed, filename)

    if not os.path.exists(csv_v1_path):
        print(f"❌ File not found: {csv_v1_path}")
        return

    with open(csv_v1_path, mode='r', encoding='utf-8') as scsv:
        reader = csv.DictReader(scsv)
        for row in reader:
            s_raw = row.get('seed')
            img_v1 = row.get('filename')
            if s_raw and s_raw.isdigit():
                valid_data_v1.append((int(s_raw), img_v1))

    valid_data_v1 = valid_data_v1[:runs_count]
    total = len(valid_data_v1)

    base_workflow = wf_transformer.load_workflow_by_filename(wf_name)
    ks_id, pos_id, neg_id = find_nodes(base_workflow)

    with open(paired_results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "seed", "img_v1_filename", "prompt_version", "img_v2_filename", "timestamp"])

        for i, (seed_value, img_v1_filename) in enumerate(valid_data_v1, 1):
            wf = deepcopy(base_workflow)

            wf[ks_id]["inputs"]["seed"] = seed_value
            wf[pos_id]["inputs"]["text"] = prompt_v2
            wf[neg_id]["inputs"]["text"] = negative
            wf[neg_id]["inputs"]["filename_prefix"] = file_name_pref

            prompt_id = queue_prompt(wf, cfg)

# It seems that we don't need to write hits if in paired run
            if not prompt_id:
                print(f"❌ [{i}/{total}] API Error. Missed...")
                continue

            print(f"⏳ [{i}/{total}] Ожидание V2 (Seed: {seed_value})...")
            real_filename_v2 = wait_for_filename(prompt_id, cfg)

            if not real_filename_v2:
                real_filename_v2 = "TIMEOUT_ERROR"
                print(f"⚠️ [{i}/{total}] Timeout")

            writer.writerow([
                i,
                seed_value,
                img_v1_filename,
                "V2", # todo: remove this hardcode. though it's usually v2, v3 and so on
                real_filename_v2,
                datetime.now(timezone.utc).isoformat()
            ])
            f.flush()
            print(f"✨ [{i}/{total}] Склеено в лог: V1:{img_v1_filename} | V2:{real_filename_v2}")

    print(f"🏁 Все готово. Мастер-лог: {paired_results_csv}")

def run_golden_seeds():
    cfg = paths.get_cfg()
    base_workflow = wf_transformer.load_workflow_by_filename(cfg['model']['workflow_sdxl_json'])

    golden_seeds = cfg['test_suites']['best_seeds']
    prompts = cfg['prompts']

    print(f"🚀 Запуск уточнения для {len(golden_seeds)} избранных сидов...")

    for i, seed_value in enumerate(golden_seeds, 1):
        wf = deepcopy(base_workflow)

        for node_id, node in wf.items():
            if node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = seed_value

            if node.get("class_type") == "CLIPTextEncode":
                if "unicorn" in str(node["inputs"].get("text", "")).lower() or i == 1:
                    # node["inputs"]["text"] = config.NEUTRAL_PROMPT
                    node["inputs"]["text"] = prompts['base_test_prompt']

        print(f"[{i}/{len(golden_seeds)}] Отправка сида {seed_value} с промптом V3...")
        prompt_id = queue_prompt(wf, cfg)

        if prompt_id:
            print(f"✅ Успешно! ID: {prompt_id}")
        else:
            print(f"❌ Ошибка для сида {seed_value}")

        time.sleep(1)
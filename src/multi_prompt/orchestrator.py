import json
import re
from copy import deepcopy

from core.call_api import queue_prompt
from src.utils.config_params import get_run_params2
from src.utils.paths import paths
from src.utils.utils import generate_seed
from workflow_api import wf_transformer


def prepare_prompt_versions_matrix(current_exp: dict, total_runs: int) -> list:
    sorted_versions = sorted([k for k in current_exp.keys() if k.startswith("v")],
                             key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    mega_variants = []
    for run_idx in range(1, total_runs + 1):
        session_seed = generate_seed()

        for version in sorted_versions:
            node = current_exp[version]
            if not isinstance(node, dict):
                continue
            subject = node.get("subject", "").strip()
            if not subject:
                continue

            # Извлекаем индивидуальные параметры инференса для данной версии.
            # Если они не заданы в словаре версии, берем стандартные безопасные дефолты.
            version_cfg = float(node.get("cfg", 7.0))
            version_steps = int(node.get("steps", 20))

            mega_variants.append({
                "version": version,
                "test": node.get("test", "Sampler_Optimization"),
                "subject": subject,
                "style": node.get("style", ""),
                "iteration": int(run_idx),
                "seed": session_seed,
                "cfg": version_cfg,  # Инжектируем в JSON для PromptBatchLoader
                "steps": version_steps  # Инжектируем в JSON для PromptBatchLoader
            })
    return mega_variants


def _extract_generation_params(p: dict) -> dict:
    gen_cfg = p.get("generation_params", {})
    return {
        #todo: Indtoduce constants
        "ckpt_name": gen_cfg.get("ckpt_name", "sd_xl_base_1.0.safetensors"),
        "steps": gen_cfg.get("steps", 20),
        "cfg_scale": gen_cfg.get("cfg", 7.0),
        "negative_prompt": gen_cfg.get("negative_prompt", "low quality, blurry, distorted, extra limbs"),
        "sampler_name": gen_cfg.get("sampler_name", "euler"),
        "scheduler": gen_cfg.get("scheduler", "normal"),
        "exp_name": p.get("exp_name", "sampler_grid_search")
    }


def _inject_workflow_inputs(wf: dict, mega_variants: list, params: dict) -> dict:
    """Мутирует копию графа воркфлоу, динамически определяя узлы по их class_type."""
    graph = deepcopy(wf)

    ks_id = None
    loader_id = None
    neg_id = None
    pos_id = None
    scorer_id = None
    save_id = None

    # Динамический поиск узлов по типам классов
    for node_id, node_data in graph.items():
        cls_type = node_data.get("class_type")
        if cls_type in ["MicroBatchKSampler", "KSampler"]:
            ks_id = node_id
        elif cls_type == "PromptBatchLoader":
            loader_id = node_id
        elif cls_type == "CLIPTextEncode" and "text" in node_data.get("inputs", {}):
            neg_id = node_id
        elif cls_type == "BatchCLIPTextEncode":
            pos_id = node_id
        elif cls_type == "MultiMetricScorer":
            scorer_id = node_id
        elif cls_type == "SaveImage":
            save_id = node_id

    # Валидация
    missing_nodes = []
    if not ks_id: missing_nodes.append("MicroBatchKSampler/KSampler")
    if not loader_id: missing_nodes.append("PromptBatchLoader")
    if not neg_id: missing_nodes.append("CLIPTextEncode (Negative)")

    if missing_nodes:
        raise ValueError(f"🚨 [ORCHESTRATOR] Critical Error: Missing components in JSON: {missing_nodes}")

    # Переключаем класс узла сэмплера на наш кастомный, если там был базовый
    graph[ks_id]["class_type"] = "MicroBatchKSampler"

    # 1. Загрузка матрицы в PromptBatchLoader
    graph[loader_id]["inputs"]["variants_json"] = json.dumps(mega_variants)

    # 2. Конфигурация MicroBatchKSampler
    graph[ks_id]["inputs"]["steps"] = params["steps"]
    graph[ks_id]["inputs"]["cfg"] = params["cfg_scale"]
    graph[ks_id]["inputs"]["sampler_name"] = params["sampler_name"]
    graph[ks_id]["inputs"]["scheduler"] = params["scheduler"]
    graph[neg_id]["inputs"]["text"] = params["negative_prompt"]

    # КРИТИЧЕСКИЙ ФИКС ДЛЯ СВИПА: Размер микро-батча строго 1
    graph[ks_id]["inputs"]["max_micro_batch_size"] = 1

    # Коммутация LIST-интерфейсов из PromptBatchLoader (Индексы из RETURN_TYPES лоадера)
    # 5: batch_size, 6: iteration_list, 7: seed_list, 8: cfg_list, 9: steps_list
    graph[ks_id]["inputs"]["seed_list"] = [loader_id, 7]
    graph[ks_id]["inputs"]["cfg_list"] = [loader_id, 8]
    graph[ks_id]["inputs"]["steps_list"] = [loader_id, 9]

    # Автоматическая привязка размера латентного батча к выходу лоадера (индекс 5 - batch_size)
    for node_id, node_data in graph.items():
        if node_data.get("class_type") == "EmptyLatentImage":
            node_data["inputs"]["batch_size"] = [loader_id, 5]

    # 3. Настройка параметров и линковки в MultiMetricScorer
    if scorer_id:
        graph[scorer_id]["inputs"]["subject_list"] = [loader_id, 1]
        graph[scorer_id]["inputs"]["style_list"] = [loader_id, 2]
        graph[scorer_id]["inputs"]["version_list"] = [loader_id, 3]
        graph[scorer_id]["inputs"]["test_list"] = [loader_id, 4]
        graph[scorer_id]["inputs"]["iteration_list"] = [loader_id, 6]
        graph[scorer_id]["inputs"]["seed_list"] = [loader_id, 7]

        graph[scorer_id]["inputs"]["experiment_name"] = params["exp_name"]
        graph[scorer_id]["inputs"]["log_to_csv"] = "enable"
        if "run_count" in graph[scorer_id]["inputs"]:
            del graph[scorer_id]["inputs"]["run_count"]

    # 4. Настройка префикса для вашего нового узла сохранения картинок 80
    if save_id:
        graph[save_id]["inputs"]["filename_prefix"] = f"{params['exp_name']}_tuning"

    return graph

def run_debug_batch():
    cfg = paths.get_cfg()
    p = get_run_params2()
    current_exp_dict = p.get("current_exp", {})
    total_runs = p.get("count", 1)
    gen_params = _extract_generation_params(p)
    json_file_name = p["wf_multi_prompt"]

    if not current_exp_dict:
        print("⚠️ [ORCHESTRATOR] Error: No current experiment")
        return None

    mega_variants = prepare_prompt_versions_matrix(current_exp_dict, total_runs)
    print(f"📦 Matrix built from current_exp_dict. Total variations to run: {len(mega_variants)}")

    base_workflow = wf_transformer.load_workflow_by_filename(json_file_name)
    executable_workflow = _inject_workflow_inputs(base_workflow, mega_variants, gen_params)

    print(f"📡 Sending matrix of {len(mega_variants)} elements to ComfyUI using template file multi_prompt.json...")
    try:
        prompt_id = queue_prompt(executable_workflow, cfg)
        print(f"🚀 Dynamic execution successfully triggered! Prompt ID: {prompt_id}")
        return prompt_id
    except Exception as e:
        print(f"❌ Error while queuing template batch to ComfyUI: {e}")
        return None

if __name__ == "__main__":
    run_debug_batch()

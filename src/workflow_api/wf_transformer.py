import json
from src.utils.paths import paths

def load_workflow(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_workflow_by_filename(filename: str) -> dict:
    return load_workflow(paths.get_workflow_path(filename))

# def load_and_update_workflow(
#     workflow_path: str,
#     new_prompt: str,
#     *,
#     new_seed: int | None = None
# ) -> Dict[str, Any]:
#
#     if not os.path.exists(workflow_path):
#         raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
#
#     with open(workflow_path, "r", encoding="utf-8") as f:
#         workflow: Dict[str, Any] = json.load(f)
#
#     if not isinstance(workflow, dict):
#         raise ValueError("Workflow must be a dict of nodes")
#
#     # --- 1. Найти KSampler ---
#     ksampler_node = None
#     for node in workflow.values():
#         if node.get("class_type") == "KSampler":
#             ksampler_node = node
#             break
#
#     if ksampler_node is None:
#         raise ValueError("KSampler node not found")
#
#     inputs = ksampler_node.get("inputs", {})
#
#     if "positive" not in inputs or "negative" not in inputs:
#         raise ValueError("KSampler missing positive or negative connections")
#
#     # --- 2. Извлечь node_id из связей ---
#     positive_node_id = inputs["positive"][0]
#     negative_node_id = inputs["negative"][0]
#
#     # --- 3. Проверить существование нод ---
#     if positive_node_id not in workflow:
#         raise ValueError(f"Positive CLIP node {positive_node_id} not found")
#
#     if negative_node_id not in workflow:
#         raise ValueError(f"Negative CLIP node {negative_node_id} not found")
#
#     positive_node = workflow[positive_node_id]
#     negative_node = workflow[negative_node_id]
#
#     # --- 4. Валидация типов ---
#     if positive_node.get("class_type") != "CLIPTextEncode":
#         raise ValueError("Positive node is not CLIPTextEncode")
#
#     if negative_node.get("class_type") != "CLIPTextEncode":
#         raise ValueError("Negative node is not CLIPTextEncode")
#
#     # --- 5. Обновление промптов ---
#     positive_node["inputs"]["text"] = new_prompt
#     negative_node["inputs"]["text"] = config.NEGATIVE_PROMPT_TEXT
#
#     # --- 6. Обновление seed (опционально) ---
#     if new_seed is not None:
#         if "seed" in inputs:
#             inputs["seed"] = int(new_seed)
#
#     return workflow

# def find_nodes(workflow: dict):
#     ks = None
#     pos = None
#     neg = None
#     for node_id, node in workflow.items():
#         ct = node.get("class_type")
#         if ct == "KSampler":
#             ks = node
#         elif ct == "CLIPTextEncode":
#             for parent in workflow.values():
#                 inputs = parent.get("inputs", {})
#                 if "positive" in inputs and inputs["positive"][0] == node_id:
#                     pos = node
#                 if "negative" in inputs and inputs["negative"][0] == node_id:
#                     neg = node
#     if not (ks and pos and neg):
#         raise RuntimeError("Required nodes not found")
#     return ks, pos, neg

def find_nodes(workflow: dict):
    ks_id = None

    # 1. Ищем ID KSampler'а
    for node_id, node in workflow.items():
        if "KSampler" in node.get("class_type", ""):
            ks_id = node_id
            break

    if not ks_id:
        raise RuntimeError("KSampler node not found")

    # 2. Получаем ID связанных нод промптов из входов KSampler'а
    ks_node = workflow[ks_id]
    inputs = ks_node.get("inputs", {})

    # ComfyUI хранит линки как [ID, OutputIndex], берем ID
    pos_id = str(inputs.get("positive", [None])[0])
    neg_id = str(inputs.get("negative", [None])[0])

    # Проверка на существование
    if pos_id not in workflow or neg_id not in workflow:
        raise RuntimeError(f"Linked nodes {pos_id} or {neg_id} not found in workflow")

    return ks_id, pos_id, neg_id

def find_custom_nodes(wf):
    ks_id = None
    pos_id = None  # В новой схеме это может быть BatchCLIPTextEncode
    neg_id = None
    loader_id = None

    for node_id, node in wf.items():
        class_type = node.get("class_type")

        if class_type == "KSampler":
            ks_id = node_id

        # Находим наш новый загрузчик батча
        elif class_type == "PromptBatchLoader":
            loader_id = node_id

        # Находим кодировщик батча (или обычный, если используем его)
        # В вашей новой схеме это BatchCLIPTextEncode
        elif class_type == "BatchCLIPTextEncode":
            pos_id = node_id

        # Отрицательный промпт обычно остается стандартным CLIPTextEncode
        elif class_type == "CLIPTextEncode":
            # Простая эвристика: если узел не pos_id, значит это neg_id
            # Или проверьте наличие слова "negative" в тексте, если есть default
            if node_id != pos_id:
                neg_id = node_id

    # Проверка на наличие критических узлов
    if not all([ks_id, neg_id, loader_id]):
        missing = [k for k, v in {"KSampler": ks_id, "Neg": neg_id, "Loader": loader_id}.items() if v is None]
        raise ValueError(f"Could not find all required nodes in workflow: missing {missing}")

    return ks_id, pos_id, neg_id, loader_id

# def find_nodes(workflow: dict):
#     ks_node = None
#
#     for node_id, node in workflow.items():
#         if "KSampler" in node.get("class_type", ""):
#             ks_node = node
#             break  # первый найденный
#
#     if not ks_node:
#         raise RuntimeError("KSampler node not found")
#
#     inputs = ks_node.get("inputs", {})
#     pos_id = inputs.get("positive", [None])[0]
#     neg_id = inputs.get("negative", [None])[0]
#
#     # сами ноды по ID
#     pos_node = workflow.get(str(pos_id))
#     neg_node = workflow.get(str(neg_id))
#
#     if not (pos_node and neg_node):
#         raise RuntimeError("Linked CLIPTextEncode nodes not found in workflow")
#
#     return ks_node, pos_node, neg_node

# def get_target_node_ids(base_workflow):
#     ks_node_ref, pos_node_ref, neg_node_ref = find_nodes(base_workflow)
#
#     try:
#         ks_id = next(k for k, v in base_workflow.items() if v is ks_node_ref)
#         pos_id = next(k for k, v in base_workflow.items() if v is pos_node_ref)
#         neg_id = next(k for k, v in base_workflow.items() if v is neg_node_ref)
#         return ks_id, pos_id, neg_id
#     except StopIteration:
#         raise ValueError("❌ Couldn't find ID of target nodes in a workflow")

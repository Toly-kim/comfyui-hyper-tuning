import time
from copy import deepcopy

from core.call_api import queue_prompt, get_history_id
from utils.config_params import get_run_params2
from utils.paths import paths
from workflow_api import wf_transformer

TIMEOUT = 60

def update_json(prompt: dict, image_name: str, description: str) -> dict:
    json_prompt = deepcopy(prompt)

    # unwrap if already wrapped
    if "prompt" in json_prompt and isinstance(json_prompt["prompt"], dict):
        nodes = json_prompt["prompt"]
    else:
        nodes = json_prompt

    for node in nodes.values():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type == "LoadImage":
            inputs["image"] = image_name

        elif class_type == "CLIPScorer":
            inputs["text"] = description
            inputs["nonce"] = str(time.time_ns())

    # json_prompt["extra_data"] = {"force": True}
    return json_prompt

def run_scorer() -> float:
    cfg = paths.get_cfg()
    p = get_run_params2()

    wf_name = p['test_json']
    base_workflow = wf_transformer.load_workflow_by_filename(wf_name)

    test_image = p['test_image']
    description_ = p['test_img_description']

    updated_prompt = update_json(base_workflow, test_image, description_)

    prompt_id = queue_prompt(updated_prompt, cfg)
    print("PROMPT_ID:", prompt_id)
    if not prompt_id:
        raise RuntimeError("Failed to queue prompt")

    start = time.time()
    while True:
        if time.time() - start > TIMEOUT:
            raise TimeoutError("ComfyUI did not respond in time")

        data = get_history_id(cfg, prompt_id)

        print("DATA:", data)

        if not data:
            time.sleep(0.5)
            continue

        entry = data.get(prompt_id)
        if not entry:
            time.sleep(0.5)
            continue

        if entry.get("status", {}).get("completed"):
            outputs = entry.get("outputs", {})

            for node_output in outputs.values():
                score_list = node_output.get("score")
                if score_list:
                    return score_list[0]

            raise RuntimeError("Execution completed but no score was found in outputs")

        time.sleep(0.5)

# def run(image_name: str, text: str, timeout: int = 60) -> float:
#     # 1. Build prompt graph
#     prompt = {
#         "prompt": {
#             "1": {
#                 "class_type": "LoadImage",
#                 "inputs": {
#                     "image": image_name
#                 }
#             },
#             "2": {
#                 "class_type": "CLIPScorer",
#                 "inputs": {
#                     "image": ["1", 0],
#                     "text": text
#                 }
#             }
#         }
#     }
#
#     # 2. Send prompt
#     r = requests.post(f"{COMFY_URL}/prompt", json=prompt)
#     r.raise_for_status()
#     prompt_id = r.json()["prompt_id"]
#
#     # 3. Poll history
#     start = time.time()
#     while True:
#         if time.time() - start > timeout:
#             raise TimeoutError("ComfyUI did not respond in time")
#
#         h = requests.get(f"{COMFY_URL}/history/{prompt_id}")
#         h.raise_for_status()
#         data = h.json()
#
#         if prompt_id in data:
#             entry = data[prompt_id]
#
#             # check completion
#             if entry["status"]["completed"]:
#                 outputs = entry.get("outputs", {})
#
#                 # node id "2" = CLIPScorer
#                 if "2" in outputs and "score" in outputs["2"]:
#                     return outputs["2"]["score"][0]
#
#                 raise ValueError("No score in outputs")
#
#         time.sleep(0.5)


if __name__ == "__main__":
    score = run_scorer()
    print("CLIP score:", score)
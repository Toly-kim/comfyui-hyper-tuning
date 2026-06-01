import json
import time
import requests
from typing import Optional

TIMEOUT = 60

def queue_prompt(workflow_data: dict, cfg) -> Optional[str]:
    srv_cfg = cfg['server']
    endpoint_ = srv_cfg['endpoint']

    return send2srv(workflow_data, cfg, endpoint_)

def send2srv(workflow_data: dict, cfg, endpoint_) -> Optional[str]:
    srv_cfg = cfg['server']
    srv_runs = cfg['runs']
    url_ = srv_cfg['url']

    print(f"-> Отправка запроса на {url_}...")

    print("--- Исходящий JSON ---")
    print(json.dumps({"prompt": workflow_data}, indent=2))
    print("----------------------")

    full_url = f"{url_.rstrip('/')}{endpoint_}"
    headers = srv_cfg['headers']
    # timeout = srv_runs['timeout']
    json_text = {"prompt": workflow_data}
    print(full_url)

    return _request_srv(full_url, headers, TIMEOUT, json_text)

import requests
from typing import Optional, Dict, Any


def get_history_id(cfg, history_id: str) -> Optional[Dict[str, Any]]:
    srv_cfg = cfg['server']
    srv_runs = cfg['runs']
    # timeout = srv_runs['timeout']

    url = srv_cfg['url']
    endpoint = srv_cfg['history_endpoint']

    if not history_id:
        raise ValueError("history_id is empty")

    url_endpoint = f"{url.rstrip('/')}/{endpoint.strip('/')}"
    full_url = f"{url_endpoint}/{history_id}"

    try:
        response = requests.get(full_url, TIMEOUT)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"GET /history failed: {e}")
        return None
    except ValueError:
        print("Invalid JSON in response")
        return None

def _request_srv(full_url: str, headers: dict, timeout: float, json_data: dict) -> Optional[str]:
    try:
        response = requests.post(
            url=full_url,
            json=json_data,
            headers=headers,
            timeout=timeout
        )

        response.raise_for_status()

        data = response.json()
        prompt_id = data.get("prompt_id")

        if not prompt_id:
            raise KeyError("prompt_id not found in response")

        print(f"<- Задача успешно поставлена в очередь. ID: {prompt_id}")
        return prompt_id

    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"Ошибка: {e}")
        return None

def wait_for_filename(prompt_id, cfg, timeout=300):
    srv_cfg = cfg['server']
    history_url = f"{srv_cfg['url'].rstrip('/')}/history/{prompt_id}"

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(history_url, TIMEOUT)
            if response.status_code == 200:
                history = response.json()
                # Если prompt_id появился в истории — значит, работа готова
                if prompt_id in history:
                    outputs = history[prompt_id].get('outputs', {})
                    # Ищем узел, который сохранил изображение (обычно класс SaveImage)
                    for node_id, node_data in outputs.items():
                        if 'images' in node_data:
                            return node_data['images'][0]['filename']
        except Exception as e:
            print(f"Network error: {e}")
            # Небольшая пауза при ошибке сети, чтобы не вылететь
            time.sleep(2)

        time.sleep(3)  # Пауза между проверками (polling interval)

    return None  # Если вышли по таймауту
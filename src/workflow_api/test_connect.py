from comfyuiclient import ComfyUIClient

COMFYUI_SERVER_URL = "https://petite-spoons-fetch.loca.lt"
PROMPT_URL = f"{COMFYUI_SERVER_URL}/prompt"

MODEL_TO_USE = "v1-5-pruned-emaonly.ckpt" # ### Run ComfyUI with localtunnel
WORKFLOW_JSON_PATH = "knight_workflow.json"
NEW_PROMPT_TEXT = "A valiant Jedi Knight, riding a unicorn, dramatic sunset, oil painting. Highly detailed."

CUSTOM_HEADERS = {
    'User-Agent': 'ComfyUI_API_Client/1.0',
    'Content-Type': 'application/json',
    'Bypass-Tunnel-Reminder': 'true'
}

def test():
    try:
        client = ComfyUIClient(COMFYUI_SERVER_URL)
        print(client.get_system_stats())
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test()
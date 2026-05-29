# @title #Run_everyday_comfy {"display-mode":"code"}
# ==============================================================================
# 1. MOUNT GOOGLE DRIVE (Interactive)
# ==============================================================================
import subprocess

from google.colab import drive
import os
import sys

if not os.path.exists('/content/drive'):
    print("🔄 Mounting Google Drive...")
    drive.mount('/content/drive')

# ==============================================================================
# 2. SILENT DEPENDENCY INSTALLATION ENGINE
# ==============================================================================
print("📦 Installing required environment dependencies...")

dependencies = [
    ["/usr/bin/python3", "-m", "pip", "install", "av"],
    ["/usr/bin/python3", "-m", "pip", "install", "torchsde"],
    ["/usr/bin/python3", "-m", "pip", "install", "spandrel"],
    ["/usr/bin/python3", "-m", "pip", "install", "-r", "/content/drive/MyDrive/ComfyUI/requirements.txt"],
    ["/usr/bin/python3", "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"],
    ["/usr/bin/python3", "-m", "pip", "install", "clip"],
    ["/usr/bin/python3", "-m", "pip", "install", "pillow", "numpy", "torch", "opencv-python", "transformers"]
]

# Install localtunnel globally via npm
subprocess.run(["npm", "install", "-g", "localtunnel"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Run pip installations sequentially, hiding the messy install logs
for cmd in dependencies:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Warning running install step: {cmd}. Error: {e}")

print("✅ Environment successfully configured.")

# ==============================================================================
# 3. BACKGROUND LOCALTUNNEL TUNNELING
# ==============================================================================
import threading
import time
import socket
import urllib.request

def iframe_thread(port):
    while True:
        time.sleep(0.5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        if result == 0:
            break
        sock.close()

    print("\n🚀 ComfyUI backend initialized! Generating localtunnel gateway...\n")
    try:
        endpoint_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
        print(f"🔑 Your localtunnel Tunnel Password/IP is: {endpoint_ip}")
        print("👉 Click the link generated below, paste this IP, and click 'Submit'.\n")
    except Exception:
        print("⚠️ Could not retrieve external IP automatically. Use your Colab runtime instance IP.")

    p = subprocess.Popen(["lt", "--port", str(port)], stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line.decode(), end='')

# Start localtunnel monitoring in a separate parallel thread
threading.Thread(target=iframe_thread, daemon=True, args=(8188,)).start()
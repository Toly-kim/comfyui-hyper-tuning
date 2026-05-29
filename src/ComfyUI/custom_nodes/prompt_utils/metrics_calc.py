import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata
import random as py_random

_clip_proc = None
_clip_model = None

def init_clip(target_device="cuda"):
    """
    Инициализирует процессор и модель CLIP на целевом устройстве,
    если они ещё не были загружены.
    """
    global _clip_proc, _clip_model
    if _clip_proc is None or _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        model_name = "openai/clip-vit-large-patch14"
        _clip_proc = CLIPProcessor.from_pretrained(model_name)
        # Загружаем модель строго на тот девайс, где крутится текущий поток
        _clip_model = CLIPModel.from_pretrained(model_name).to(target_device)


def _ensure_3d_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Защитная функция: гарантирует, что на входе именно 3D тензор [H, W, C].
    Если прилетел 4D батч [1, H, W, C], безопасно убирает батч-директиву.
    """
    if len(image_tensor.shape) == 4:
        if image_tensor.shape[0] == 1:
            return image_tensor.squeeze(0)
        else:
            raise ValueError(f"Expected a single image tensor, but received batch dimension: {image_tensor.shape}")
    return image_tensor


def calc_sharpness(image_tensor):
    """
    Вычисляет резкость одиночного изображения (тензора) с помощью Лапласиана.
    Вход: image_tensor размерностью [H, W, C] или [1, H, W, C] (значения 0.0 - 1.0)
    Выход: float (значение резкости)
    """
    try:
        # Защита от кривой размерности upstream-узлов
        image_tensor = _ensure_3d_tensor(image_tensor)

        # 1. Переводим тензор PyTorch в NumPy массив [0, 255]
        img_np = image_tensor.cpu().numpy()
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)

        # 2. Конвертируем в градации серого для работы оператора Лапласа
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 3. Считаем дисперсию Лапласиана
        sharpness_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(sharpness_value)
    except Exception as e:
        print(f"⚠️ [METRICS_CALC] Error computing sharpness: {e}")
        return 0.0

def calc_clip_similarity(image_tensor, text: str) -> float:
    """
    Считает косинусное сходство между текстовым описанием и изображением через CLIP.
    Принимает тензор ComfyUI формата [H, W, C] или [1, H, W, C], значения [0.0, 1.0]
    """
    if not text or not text.strip():
        return 0.0

    try:
        # Защита от кривой размерности upstream-узлов
        image_tensor = _ensure_3d_tensor(image_tensor)

        # Динамически определяем девайс на основе входного тензора
        current_device = image_tensor.device
        init_clip(target_device=current_device)

        if _clip_proc is None or _clip_model is None:
            raise RuntimeError("CLIP model failed to initialize")

        # Переводим тензор ComfyUI в PIL Image в памяти
        img_np = image_tensor.cpu().numpy()
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np).convert("RGB")

        # Токенизируем данные
        inputs = _clip_proc(
            text=[text],
            images=img,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(current_device)

        with torch.no_grad():
            outputs = _clip_model(**inputs)
            # Принудительно отвязываем эмбеддинги от контекста и уводим на CPU
            img_feat = outputs.image_embeds.detach().cpu()
            txt_feat = outputs.text_embeds.detach().cpu()

        # Нормализация (Cosine Similarity) на стороне CPU для очистки VRAM
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

        similarity = (img_feat * txt_feat).sum(dim=-1)
        return float(similarity.item())

    except Exception as e:
        print(f"⚠️ [METRICS_CALC] Error computing CLIP similarity: {e}")
        return 0.0

# Используется min-max normalization:
# (x - min) / (max - min)
def normalize_metrics(values_list):
    if not values_list:
        return []

    arr = np.array(values_list, dtype=np.float32)
    min_val = arr.min()
    max_val = arr.max()

    diff = max_val - min_val
    if diff < 1e-8:
        # Если все элементы идентичны, возвращаем средний уровень заполнения
        return list(np.ones_like(arr, dtype=np.float32) * 0.5)

    normalized = (arr - min_val) / diff
    return [float(x) for x in normalized]

def normalize_metrics2(indicators: list) -> list:
    if not indicators:
        return []

    ranks = rankdata(indicators, method="average")
    n = len(indicators)

    return [(r - 1) / (n - 1) if n > 1 else 0.0 for r in ranks]

def rank_metrics(values_list):
    """
    Ранжирование элементов. Чем больше значение скора, тем выше ранг (1 = лучший).
    """
    if not values_list:
        return []

    arr = np.array(values_list, dtype=np.float32)
    # rankdata выдает ранги по возрастанию. Инвертируем знак, чтобы большие скоры получали 1-й ранг.
    ranks = rankdata(-arr, method="min")
    return [int(r) for r in ranks]


# Оставляем пустую заглушку для обратной совместимости, если внешние ноды обращаются к ней напрямую
def generate_seed() -> int:
    return py_random.randint(1, 2 ** 63 - 1)
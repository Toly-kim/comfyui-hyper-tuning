import cv2
import lpips
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPProcessor, CLIPModel

# Настройка устройства (как статическое поле в Java)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_proc = None
_lpips_model = None

_lpips_tr = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def init_clip():
    global _clip_model, _clip_proc
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()

def clip_similarity(image_path: str, text: str) -> float:
    init_clip()
    if _clip_proc is None or _clip_model is None:
        raise RuntimeError("CLIP model failed to initialize")
    
    img = Image.open(image_path).convert("RGB")
    inputs = _clip_proc(
        text=[text],
        images=img,
        return_tensors="pt",
        padding=True,
        truncation=True,      #
        max_length=77         #
    ).to(DEVICE)

    # inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _clip_model(**inputs)
        img_feat = outputs.image_embeds
        # img_feat = outputs.get_image_features
        txt_feat = outputs.text_embeds

    # Нормализация (Cosine Similarity)
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)
    # img_feat /= img_feat.norm(dim=-1, keepdim=True)
    # txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    return float((img_feat @ txt_feat.T).item())

def init_lpips():
    global _lpips_model
    if _lpips_model is None:
        # spatial=False возвращает одно число (дистанцию)
        _lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
        _lpips_model.eval()

def calculate_sharpness(image_path: str) -> float:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return 0.0

    gauss_img = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.Laplacian(gauss_img, cv2.CV_64F).var()

def calculate_brightness_contrast(image_path: str):
    """Оценка освещенности и контраста"""
    image = cv2.imread(image_path)
    if image is None: return 0, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray), np.std(gray)

def compute_lpips(path_a: str, path_b: str) -> float:
    """How big the prompt change is perceptually
    0.1–0.2 → похожие
    0.3–0.4 → заметно разные
    LPIPS > 0.4 при одинаковых seed → pipeline не детерминирован или seed не применяется корректно.
    Или → SDXL + prompt complexity создаёт high entropy генерацию
    0.5 → сильно разные
    """
    init_lpips()

    a = _lpips_tr(Image.open(path_a).convert("RGB")).unsqueeze(0).to(DEVICE)
    b = _lpips_tr(Image.open(path_b).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        d = _lpips_model(a,b)
    # return float(d.item())
    return float(d.squeeze().cpu().item())

def compute_ssim(path_a: str, path_b: str) -> float:
    """Важно: SSIM чувствителен к шуму, чтение в Grayscale

    SSIM > 0.8: Структура сохранена. ИИ просто поменял освещение, текстуру или добавил мелкий шум.
    SSIM 0.4–0.6: "Композиционный дрейф". Объекты на тех же местах, но их контуры и позы изменились.
    SSIM < 0.3: Полная перестановка. Модель сгенерировала принципиально другую композицию.
    """

    A = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if A is None or B is None:
        return 0.0

    if A.shape != B.shape:
        target_size = (min(A.shape[1], B.shape[1]), min(A.shape[0], B.shape[0]))
        A = cv2.resize(A, target_size, interpolation=cv2.INTER_AREA)
        B = cv2.resize(B, target_size, interpolation=cv2.INTER_AREA)

    # return float(ssim(A, B))
    return float(ssim(A, B, data_range=255))

def color_hist_distance(path_a: str, path_b: str, bins=32) -> float: # bins=16
    """Bhattacharyya returns 0.0 if identical. 1.0, if different"""
    #color_dist ≈ 0.78 - цветовая палитра между v1 и v2 сильно различается
    a = cv2.imread(path_a)
    b = cv2.imread(path_b)
    if a is None or b is None:
        return 1.0

    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

    # a = cv2.resize(a, (256, 256)) # 256 → для color_hist, LPIPS (стабильность)?
    # b = cv2.resize(b, (256, 256)) # 512 → для SSIM, style?
    a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA)

    a_h = cv2.calcHist([a], [0,1,2], None, [bins]*3, [0,256]*3)
    b_h = cv2.calcHist([b], [0,1,2], None, [bins]*3, [0,256]*3)

    cv2.normalize(a_h, a_h, alpha=1, norm_type=cv2.NORM_L1)
    cv2.normalize(b_h, b_h, alpha=1, norm_type=cv2.NORM_L1)

    return float(cv2.compareHist(a_h, b_h, cv2.HISTCMP_BHATTACHARYYA))
import torch
import numpy as np
from pathlib import Path
from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import read_image


def initialize_matcher(
    superglue_weights: str = 'indoor',
    keypoint_thresh: float = 0.005,
    nms_radius: int = 4,
    max_keypoints: int = 1024,
    sinkhorn_iterations: int = 20,
    match_thresh: float = 0.2,
    device: torch.device | None = None
) -> Matching:
    """
    Crea y devuelve un objeto `Matching` (SuperPoint + SuperGlue) configurado.

    Parámetros:
    - superglue_weights: {'indoor', 'outdoor'} (pesos de SuperGlue a usar).
    - keypoint_thresh: umbral de confianza para SuperPoint.
    - nms_radius: radio de NMS para SuperPoint.
    - max_keypoints: número máximo de keypoints que SuperPoint extrae.
    - sinkhorn_iterations: número de iteraciones de Sinkhorn en SuperGlue.
    - match_thresh: umbral mínimo de confianza para aceptar un match.
    - device: dispositivo en que correr (p.ej. `torch.device('cuda')` o `torch.device('cpu')`). 
              Si es None, se elige automáticamente CUDA si está disponible.

    Devuelve:
    - Un objeto `Matching` listo para hacer inferencia.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'superpoint': {
            'nms_radius':         nms_radius,
            'keypoint_threshold': keypoint_thresh,
            'max_keypoints':      max_keypoints
        },
        'superglue': {
            'weights':             superglue_weights,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold':     match_thresh,
        }
    }
    matcher = Matching(config).eval().to(device)
    return matcher


def match_two_images(
    img_path0: str,
    img_path1: str,
    matcher: Matching,
    device: torch.device | None = None,
    resize: tuple[int, int] = (640, 480),
    resize_float: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dadas dos rutas de imagen y un matcher ya inicializado, devuelve dos arrays:
      - mkpts0: array (N×2) con coordenadas [x, y] de keypoints en imagen 0.
      - mkpts1: array (N×2) con coordenadas [x, y] de keypoints en imagen 1 
                que coinciden con mkpts0.

    Obs.: Las coordenadas están en el espacio de la imagen redimensionada a `resize` (por defecto 640×480).

    Parámetros:
    - img_path0, img_path1: rutas (str) a cada imagen.
    - matcher: objeto `Matching` inicializado con `initialize_matcher`.
    - device: dispositivo en que correr la inferencia. Si es None, se usa el mismo que el `matcher`.
    - resize: tupla (ancho, alto) para redimensionar antes de la inferencia.
              Para no redimensionar, pasar algo como (-1, -1).
    - resize_float: si True, convierte primero a float y luego redimensiona (mismo comportamiento que en el ejemplo oficial).

    Retorna:
    - mkpts0: array (N, 2) con coordenadas [x, y] de keypoints emparejados en la imagen 0.
    - mkpts1: array (N, 2) con coordenadas [x, y] de sus correspondientes en la imagen 1.
    """
    if device is None:
        device = next(matcher.parameters()).device

    # Leemos y preprocesamos ambas imágenes usando la utilidad de SuperGlue
    # read_image devuelve: (image_bgr, inp_tensor, scales)
    image0, inp0, scales0 = read_image(Path(img_path0), device, resize, 0, resize_float)
    image1, inp1, scales1 = read_image(Path(img_path1), device, resize, 0, resize_float)

    if image0 is None or image1 is None:
        raise ValueError(f"No se pudo leer alguna de las imágenes: {img_path0}, {img_path1}")

    # Armamos el diccionario de entrada esperado por el modelo
    input_dict = {'image0': inp0, 'image1': inp1}

    # Inference (SuperPoint + SuperGlue)
    with torch.no_grad():
        pred = matcher(input_dict)

    # Convertimos cada tensor a numpy (solo el primer batch)
    pred_np = {k: v[0].cpu().numpy() for k, v in pred.items()}

    # Extraemos los keypoints detectados y la correspondencia
    kpts0      = pred_np['keypoints0']        # shape (K0, 2)
    kpts1      = pred_np['keypoints1']        # shape (K1, 2)
    matches0   = pred_np['matches0']          # shape (K0,), valor -1 si no hay match
    match_conf = pred_np['matching_scores0']   # shape (K0,)

    # Filtramos solo aquellos keypoints de imagen0 que tienen match en imagen1
    valid_mask = matches0 > -1
    mkpts0 = kpts0[valid_mask]
    mkpts1 = kpts1[matches0[valid_mask]]

    return mkpts0, mkpts1

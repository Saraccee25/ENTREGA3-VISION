# -*- coding: utf-8 -*-
# Reto 4 - Transformer para Restauracion de Imagenes SAR
# Modelo: SwinIR (Swin Transformer for Image Restoration)
# Sin reentrenamiento - se usa modelo preentrenado de HuggingFace
# Ejecutar desde cualquier PC: python reto4.py

# ==============================================================================
# 1. IMPORTS Y DEPENDENCIAS
#    Instalar si es necesario:
#    pip install torch torchvision timm requests pillow
#    pip install scikit-image matplotlib opencv-python numpy
# ==============================================================================
import os
import sys
import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# scikit-image para metricas
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

# ==============================================================================
# 2. RUTAS  — automaticas desde donde este el script
# ==============================================================================
BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = BASE_DIR / 'img_reto4'
OUTPUT_DIR = BASE_DIR / 'outputs_reto4'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'axes.titlepad':    10,
    'figure.titlesize': 14,
    'figure.dpi':       110,
})

# ==============================================================================
# 3. CARGA DE IMAGENES
#    Acepta .tiff, .tif, .png, .jpg de la carpeta img_reto4
# ==============================================================================
def load_image(path):
    """Carga imagen como array float32 [0,1], RGB o gris->RGB"""
    img = np.array(Image.open(str(path)).convert('RGB')).astype(np.float32)
    # normalizar segun el maximo real del archivo
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)

exts = {'.tiff', '.tif', '.TIF', '.TIFF', '.png', '.PNG', '.jpg', '.JPG', '.jpeg'}
img_paths = sorted([p for p in INPUT_DIR.iterdir() if p.suffix in exts])

if not img_paths:
    raise FileNotFoundError(f'No se encontraron imagenes en {INPUT_DIR}')

print(f'Imagenes encontradas: {len(img_paths)}')
images_orig = {}
for p in img_paths:
    img = load_image(p)
    images_orig[p.stem] = img
    print(f'  {p.name}: {img.shape}, min={img.min():.3f}, max={img.max():.3f}, mean={img.mean():.3f}')

# ==============================================================================
# 4. DETECCION DE DEFECTOS EN LAS IMAGENES
#    Antes de restaurar se analizan los defectos visibles:
#    - Ruido speckle (varianza local alta)
#    - Bordes degradados (zona oscura en los bordes)
#    - Regiones sin informacion (valores cercanos a 0)
# ==============================================================================
def detect_defects(img, name):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # Varianza local como proxy del ruido speckle
    mean_local = cv2.blur(gray.astype(np.float32), (7, 7))
    var_local   = cv2.blur((gray.astype(np.float32) - mean_local)**2, (7, 7))
    noise_level = float(np.mean(var_local))
    # Proporcion de pixeles muy oscuros (sin informacion)
    dark_ratio  = float(np.mean(gray < 10))
    # ENL de la imagen original
    mu, sig = np.mean(img), np.std(img)
    enl = (mu / sig)**2 if sig > 0 else 0.0
    print(f'  [{name}] noise_var={noise_level:.1f} | dark_ratio={dark_ratio:.3f} | ENL={enl:.2f}')
    return noise_level, dark_ratio, enl

print('\nAnalisis de defectos:')
defects = {name: detect_defects(img, name) for name, img in images_orig.items()}

# ==============================================================================
# 5. VISUALIZACION INICIAL — imagenes originales con histogramas
# ==============================================================================
n_imgs = len(images_orig)
fig, axes = plt.subplots(2, n_imgs, figsize=(5 * n_imgs, 10))
fig.suptitle('Imagenes Originales — Analisis Previo a la Restauracion',
             fontsize=14, fontweight='bold', y=1.01)

for col, (name, img) in enumerate(images_orig.items()):
    gray = img.mean(axis=2)
    axes[0, col].imshow(gray, cmap='gray', vmin=0, vmax=1)
    axes[0, col].set_title(name, fontsize=10, pad=8)
    axes[0, col].axis('off')
    axes[1, col].hist(gray.flatten(), bins=80, color='steelblue', alpha=0.8)
    axes[1, col].set_xlabel('Intensidad'); axes[1, col].set_ylabel('Frecuencia')
    axes[1, col].set_title(f'Histograma — {name}', fontsize=9)
    axes[1, col].grid(alpha=.3)

plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)
plt.savefig(OUTPUT_DIR / 'imagenes_originales.png', dpi=130, bbox_inches='tight')
plt.show()

# ==============================================================================
# 6. SWINIR — DESCARGA Y CARGA DEL MODELO PREENTRENADO
#
#    SwinIR (Swin Transformer for Image Restoration) es un modelo Transformer
#    que usa ventanas desplazadas (shifted windows) para capturar dependencias
#    de largo alcance en la imagen.
#
#    Se usa el modelo preentrenado para denoising de imagen real (nivel sigma=15)
#    publicado por los autores originales en GitHub.
#    No requiere reentrenamiento.
# ==============================================================================

# --- 6a. Descargar codigo de SwinIR si no existe ---
SWINIR_DIR = BASE_DIR / 'swinir_model'
SWINIR_DIR.mkdir(exist_ok=True)

NETWORK_FILE = SWINIR_DIR / 'network_swinir.py'
WEIGHTS_FILE = SWINIR_DIR / 'swinir_denoising_color.pth'

NETWORK_URL = ('https://raw.githubusercontent.com/JingyunLiang/SwinIR/'
               'main/models/network_swinir.py')
WEIGHTS_URL = ('https://github.com/JingyunLiang/SwinIR/releases/download/'
               'v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth')

if not NETWORK_FILE.exists():
    print('Descargando arquitectura SwinIR...')
    urllib.request.urlretrieve(NETWORK_URL, NETWORK_FILE)
    print('  OK')

if not WEIGHTS_FILE.exists():
    print('Descargando pesos preentrenados (~50 MB)...')
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
    print('  OK')

# --- 6b. Importar SwinIR dinamicamente ---
sys.path.insert(0, str(SWINIR_DIR))
from network_swinir import SwinIR  # noqa: E402

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDispositivo: {device}')

# --- 6c. Instanciar modelo con los hiperparametros del checkpoint descargado ---
model_swinir = SwinIR(
    upscale=1,
    in_chans=3,
    img_size=128,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='',
    resi_connection='1conv'
)

checkpoint = torch.load(str(WEIGHTS_FILE), map_location=device)
# los pesos pueden estar bajo 'params' o directamente
state_dict = checkpoint.get('params', checkpoint)
model_swinir.load_state_dict(state_dict, strict=True)
model_swinir.eval().to(device)
print('Modelo SwinIR cargado correctamente.')

# ==============================================================================
# 7. APLICAR SWINIR A CADA IMAGEN
#    SwinIR requiere que el tamano sea multiplo de window_size (8).
#    Se procesa en tiles de 512x512 si la imagen es grande.
# ==============================================================================
WINDOW = 8

def pad_to_multiple(img_tensor, multiple=8):
    """Agrega padding para que H y W sean multiplos de 'multiple'"""
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(
            img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return img_tensor, h, w

def restore_image(img_np, model, dev, tile=512, tile_overlap=32):
    """
    Aplica SwinIR a una imagen numpy [H,W,3] float32 [0,1].
    Usa procesamiento por tiles para manejar imagenes grandes sin colapsar RAM.
    """
    h, w, c = img_np.shape
    # si la imagen cabe en un tile, procesar directamente
    if h <= tile and w <= tile:
        t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
        t, oh, ow = pad_to_multiple(t, WINDOW)
        with torch.no_grad():
            out = model(t)
        out = out[:, :, :oh, :ow]
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    # procesamiento por tiles
    result    = np.zeros_like(img_np)
    weight_map = np.zeros((h, w, 1), dtype=np.float32)

    stride = tile - tile_overlap
    ys = list(range(0, h - tile, stride)) + [h - tile]
    xs = list(range(0, w - tile, stride)) + [w - tile]

    for y in ys:
        for x in xs:
            patch = img_np[y:y+tile, x:x+tile, :]
            t = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
            t, ph, pw = pad_to_multiple(t, WINDOW)
            with torch.no_grad():
                out_p = model(t)
            out_p = out_p[:, :, :ph, :pw].squeeze(0).permute(1, 2, 0).cpu().numpy()
            result[y:y+tile, x:x+tile]    += out_p
            weight_map[y:y+tile, x:x+tile] += 1.0

    return np.clip(result / weight_map, 0, 1)

print('\nAplicando SwinIR a las imagenes...')
images_restored = {}
for name, img in images_orig.items():
    print(f'  Procesando {name}...', end=' ', flush=True)
    restored = restore_image(img, model_swinir, device)
    images_restored[name] = restored.astype(np.float32)
    print('listo')

# ==============================================================================
# 8. METRICAS CUANTITATIVAS
#    Se compara la imagen original (ruidosa) con la restaurada.
#    Como no hay ground truth separado para estas imagenes,
#    se usan las metricas de auto-evaluacion mas el ENL.
# ==============================================================================
def compute_enl(img):
    mu, sig = np.mean(img), np.std(img)
    return (mu / sig)**2 if sig > 0 else 0.0

def compute_metrics_pair(orig, restored, name=''):
    o_gray = orig.mean(axis=2)
    r_gray = restored.mean(axis=2)
    s = ssim(o_gray, r_gray, data_range=1.0)
    p = psnr(o_gray, r_gray, data_range=1.0)
    enl_orig = compute_enl(o_gray)
    enl_rest = compute_enl(r_gray)
    if name:
        print(f'  {name}: SSIM={s:.4f} | PSNR={p:.2f}dB | ENL_orig={enl_orig:.2f} | ENL_rest={enl_rest:.2f}')
    return {'SSIM': s, 'PSNR': p, 'ENL_orig': enl_orig, 'ENL_rest': enl_rest}

print('\nMetricas (Original vs Restaurada):')
all_metrics = {}
for name in images_orig:
    all_metrics[name] = compute_metrics_pair(
        images_orig[name], images_restored[name], name)

print('\n' + '='*65)
print(f'{"Imagen":<20} {"SSIM":>8} {"PSNR":>10} {"ENL orig":>10} {"ENL rest":>10}')
print('-'*65)
for name, m in all_metrics.items():
    print(f'{name:<20} {m["SSIM"]:>8.4f} {m["PSNR"]:>10.2f} '
          f'{m["ENL_orig"]:>10.2f} {m["ENL_rest"]:>10.2f}')
print('='*65)

# ==============================================================================
# 9. VISUALIZACIONES
# ==============================================================================
colores = ['tomato', 'steelblue', 'darkorchid', 'seagreen',
           'goldenrod', 'coral', 'teal', 'slateblue']

# ── 9a. Comparacion visual: Original vs Restaurada ───────────────────────────
fig = plt.figure(figsize=(14, 5 * n_imgs))
fig.suptitle('Comparacion Visual — Original vs SwinIR Restaurada',
             fontsize=14, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(n_imgs, 2, figure=fig, hspace=0.35, wspace=0.2)

for row, (name, orig) in enumerate(images_orig.items()):
    rest  = images_restored[name]
    color = colores[row % len(colores)]
    for col, (img, tit) in enumerate([(orig, f'Original\n{name}'),
                                       (rest, f'SwinIR Restaurada\n{name}')]):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img.mean(axis=2), cmap='gray', vmin=0, vmax=1)
        ax.set_title(tit, fontsize=10, pad=8)
        ax.axis('off')
        border_c = 'tomato' if col == 0 else 'steelblue'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_c); spine.set_linewidth(2.5); spine.set_visible(True)

plt.savefig(OUTPUT_DIR / 'comparacion_visual.png', dpi=130, bbox_inches='tight')
plt.show()

# ── 9b. Zoom en region central (bordes y detalles) ───────────────────────────
fig = plt.figure(figsize=(14, 5 * n_imgs))
fig.suptitle('Zoom — Analisis de Bordes y Detalles (region central)',
             fontsize=14, fontweight='bold', y=1.01)
gs2 = gridspec.GridSpec(n_imgs, 2, figure=fig, hspace=0.35, wspace=0.2)

for row, (name, orig) in enumerate(images_orig.items()):
    rest = images_restored[name]
    h, w = orig.shape[:2]
    cy, cx = h // 2, w // 2
    half   = min(h, w) // 4
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    for col, (img, tit) in enumerate([(orig, f'Original — {name}'),
                                       (rest, f'Restaurada — {name}')]):
        ax = fig.add_subplot(gs2[row, col])
        ax.imshow(img[y1:y2, x1:x2].mean(axis=2), cmap='gray', vmin=0, vmax=1)
        ax.set_title(tit, fontsize=10, pad=8)
        ax.axis('off')
        border_c = 'tomato' if col == 0 else 'steelblue'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_c); spine.set_linewidth(2.5); spine.set_visible(True)

plt.savefig(OUTPUT_DIR / 'zoom_detalles.png', dpi=130, bbox_inches='tight')
plt.show()

# ── 9c. Mapa de diferencias |restaurada - original| ──────────────────────────
fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
fig.suptitle('Mapa de Cambios — |Restaurada − Original|',
             fontsize=13, fontweight='bold', y=1.02)

if n_imgs == 1:
    axes = [axes]

for ax, (name, orig) in zip(axes, images_orig.items()):
    rest = images_restored[name]
    diff = np.abs(rest.mean(axis=2) - orig.mean(axis=2))
    im   = ax.imshow(diff, cmap='hot', vmin=0, vmax=0.2)
    ax.set_title(f'{name}\ncambio medio = {diff.mean():.4f}', fontsize=10, pad=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout(pad=2.5, w_pad=3.0)
plt.savefig(OUTPUT_DIR / 'mapa_cambios.png', dpi=130, bbox_inches='tight')
plt.show()

# ── 9d. Deteccion de bordes Canny: original vs restaurada ────────────────────
def edges(img_np, lo=30, hi=100):
    gray8 = (img_np.mean(axis=2) * 255).astype(np.uint8)
    return cv2.Canny(gray8, lo, hi)

fig = plt.figure(figsize=(14, 5 * n_imgs))
fig.suptitle('Deteccion de Bordes (Canny) — Original vs Restaurada',
             fontsize=13, fontweight='bold', y=1.01)
gs3 = gridspec.GridSpec(n_imgs, 2, figure=fig, hspace=0.35, wspace=0.2)

for row, (name, orig) in enumerate(images_orig.items()):
    rest = images_restored[name]
    for col, (img, tit) in enumerate([(orig,  f'Bordes Original — {name}'),
                                       (rest,  f'Bordes Restaurada — {name}')]):
        ax = fig.add_subplot(gs3[row, col])
        ax.imshow(edges(img), cmap='gray')
        ax.set_title(tit, fontsize=10, pad=8)
        ax.axis('off')
        border_c = 'tomato' if col == 0 else 'steelblue'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_c); spine.set_linewidth(2.5); spine.set_visible(True)

plt.savefig(OUTPUT_DIR / 'analisis_bordes.png', dpi=130, bbox_inches='tight')
plt.show()

# ── 9e. Comparacion de ENL (original vs restaurada) ──────────────────────────
nombres   = list(all_metrics.keys())
enl_orig  = [all_metrics[n]['ENL_orig'] for n in nombres]
enl_rest  = [all_metrics[n]['ENL_rest'] for n in nombres]

x = np.arange(len(nombres))
w = 0.35
fig, ax = plt.subplots(figsize=(max(9, 3*n_imgs), 5))
b1 = ax.bar(x - w/2, enl_orig, w, label='Original (ruidosa)',
            color='tomato',    edgecolor='white')
b2 = ax.bar(x + w/2, enl_rest, w, label='SwinIR Restaurada',
            color='steelblue', edgecolor='white')

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(nombres, fontsize=10)
ax.set_ylabel('ENL (mas alto = menos ruido residual)')
ax.set_title('Comparacion de ENL — Original vs SwinIR',
             fontweight='bold', pad=12)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(pad=2.5)
plt.savefig(OUTPUT_DIR / 'comparacion_enl.png', dpi=130, bbox_inches='tight')
plt.show()

# ── 9f. Tabla resumen de metricas ────────────────────────────────────────────
ssim_o = [ssim(images_orig[n].mean(2),
               images_orig[n].mean(2), data_range=1.0) for n in nombres]
# SSIM antes: auto-referencia = 1.0; lo que importa es SSIM orig vs rest
ssim_vals = [all_metrics[n]['SSIM']     for n in nombres]
psnr_vals = [all_metrics[n]['PSNR']     for n in nombres]

x2 = np.arange(len(nombres))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Metricas SSIM y PSNR — Original vs Restaurada',
             fontsize=13, fontweight='bold')

axes[0].bar(x2, ssim_vals, color='darkorchid', edgecolor='white')
for i, v in enumerate(ssim_vals):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_xticks(x2); axes[0].set_xticklabels(nombres, fontsize=9)
axes[0].set_ylabel('SSIM'); axes[0].set_title('SSIM Original vs Restaurada')
axes[0].set_ylim(0, 1.1); axes[0].grid(axis='y', alpha=0.3)
axes[0].spines[['top','right']].set_visible(False)

axes[1].bar(x2, psnr_vals, color='seagreen', edgecolor='white')
for i, v in enumerate(psnr_vals):
    axes[1].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
axes[1].set_xticks(x2); axes[1].set_xticklabels(nombres, fontsize=9)
axes[1].set_ylabel('PSNR (dB)'); axes[1].set_title('PSNR Original vs Restaurada')
axes[1].grid(axis='y', alpha=0.3)
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout(pad=2.5, w_pad=3.5)
plt.savefig(OUTPUT_DIR / 'comparacion_ssim_psnr.png', dpi=130, bbox_inches='tight')
plt.show()

# ── 9g. Guardar imagenes restauradas ─────────────────────────────────────────
for name, rest in images_restored.items():
    out_path = OUTPUT_DIR / f'{name}_swinir.png'
    Image.fromarray((rest * 255).astype(np.uint8)).save(str(out_path))
    print(f'Guardada: {out_path.name}')

print(f'\nTodo guardado en: {OUTPUT_DIR}')
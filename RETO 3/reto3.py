import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from glob import glob

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPooling2D,
                                     UpSampling2D, BatchNormalization,
                                     Activation, Input, Concatenate, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ==============================================================================
# 2. RUTAS Y CONFIGURACION
# ==============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR  = os.path.join(BASE_DIR, 'Noisy')
GT_DIR     = os.path.join(BASE_DIR, 'GTruth')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE   = 128    
N_IMAGES   = 500    
BATCH_SIZE = 8
random.seed(42)
np.random.seed(42)

plt.rcParams.update({
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'axes.titlepad':    10,
    'figure.titlesize': 14,
    'figure.dpi':       110,
})

# ==============================================================================
# 3. CARGA DEL DATASET
# ==============================================================================
def load_tiff(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(path))
        except Exception:
            return None

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / img.max() if img.max() > 0 else img
    if img.ndim == 3:
        img = img[..., 0]
    img = cv2.resize(img, (img_size, img_size))
    return np.clip(img, 0.0, 1.0)

def load_dataset(noisy_dir, gt_dir, img_size=128, n_images=500):
    noisy_files = sorted(
        glob(os.path.join(noisy_dir, '*.tif')) +
        glob(os.path.join(noisy_dir, '*.tiff')) +
        glob(os.path.join(noisy_dir, '*.TIF')) +
        glob(os.path.join(noisy_dir, '*.TIFF'))
    )
    gt_files = sorted(
        glob(os.path.join(gt_dir, '*.tif')) +
        glob(os.path.join(gt_dir, '*.tiff')) +
        glob(os.path.join(gt_dir, '*.TIF')) +
        glob(os.path.join(gt_dir, '*.TIFF'))
    )

    if not noisy_files:
        raise FileNotFoundError(f'No se encontraron archivos .tiff en {noisy_dir}')
    if not gt_files:
        raise FileNotFoundError(f'No se encontraron archivos .tiff en {gt_dir}')

    print(f'Archivos encontrados: {len(noisy_files)} noisy | {len(gt_files)} gt')

    noisy_files = noisy_files[:n_images]
    gt_files    = gt_files[:n_images]

    n_pairs = min(len(noisy_files), len(gt_files))
    print(f'Cargando {n_pairs} pares...')

    X_noisy, Y_clean = [], []
    errores = 0

    for i, (nf, gf) in enumerate(zip(noisy_files[:n_pairs], gt_files[:n_pairs])):
        if i % 100 == 0:
            print(f'  Progreso: {i}/{n_pairs}')

        n_img = load_tiff(nf, img_size)
        g_img = load_tiff(gf, img_size)

        if n_img is None or g_img is None:
            errores += 1
            continue

        X_noisy.append(n_img[..., np.newaxis])
        Y_clean.append(g_img[..., np.newaxis])

    if errores > 0:
        print(f'  Advertencia: {errores} pares no se pudieron cargar')

    X = np.array(X_noisy, dtype=np.float32)
    Y = np.array(Y_clean, dtype=np.float32)
    print(f'\nDataset final: {len(X)} pares | Tamano: {img_size}x{img_size}')
    return X, Y

print('Cargando dataset SAR...')
X_noisy, Y_clean = load_dataset(NOISY_DIR, GT_DIR,
                                 img_size=IMG_SIZE, n_images=N_IMAGES)


n      = len(X_noisy)
n_test = max(2, int(n * 0.15))
n_val  = max(2, int(n * 0.15))
idx    = list(range(n))
random.shuffle(idx)

test_idx  = idx[:n_test]
val_idx   = idx[n_test:n_test + n_val]
train_idx = idx[n_test + n_val:]

X_train, Y_train = X_noisy[train_idx], Y_clean[train_idx]
X_val,   Y_val   = X_noisy[val_idx],   Y_clean[val_idx]
X_test,  Y_test  = X_noisy[test_idx],  Y_clean[test_idx]

print(f'Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')

# ==============================================================================
# 4. VISUALIZACION DEL DATASET
# ==============================================================================
n_show = min(4, len(X_noisy))
fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 9))
fig.suptitle('Dataset SAR — Pares de Imagenes (Ruidosa vs Ground Truth)',
             fontsize=14, fontweight='bold', y=1.01)

for i in range(n_show):
    axes[0, i].imshow(X_noisy[i, ..., 0], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'Ruidosa #{i}', fontsize=11, pad=8)
    axes[0, i].axis('off')
    axes[1, i].imshow(Y_clean[i, ..., 0], cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title(f'Ground Truth #{i}', fontsize=11, pad=8)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Imagen Ruidosa (SAR)', fontsize=11, labelpad=10)
axes[1, 0].set_ylabel('Ground Truth (limpia)', fontsize=11, labelpad=10)
plt.tight_layout(pad=2.5, h_pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_pares.png'), dpi=130, bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Distribucion de Intensidades — Ruidosa vs Ground Truth',
             fontsize=13, fontweight='bold')

axes[0].hist(X_noisy.flatten(), bins=100, color='tomato',
             alpha=0.75, label='Ruidosa (SAR)')
axes[0].hist(Y_clean.flatten(), bins=100, color='steelblue',
             alpha=0.75, label='Ground Truth')
axes[0].set_xlabel('Intensidad'); axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Histograma Global'); axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].hist(X_noisy[0, ..., 0].flatten(), bins=80, color='tomato',
             alpha=0.75, label='Ruidosa')
axes[1].hist(Y_clean[0, ..., 0].flatten(), bins=80, color='steelblue',
             alpha=0.75, label='Ground Truth')
axes[1].set_xlabel('Intensidad'); axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Histograma Par #0'); axes[1].legend(); axes[1].grid(alpha=.3)

plt.tight_layout(pad=2.5, w_pad=3.5)
plt.savefig(os.path.join(OUTPUT_DIR, 'histogramas_dataset.png'), dpi=130, bbox_inches='tight')
plt.show()

# ==============================================================================
# 5. FUNCIONES DE PERDIDA Y METRICAS
# ==============================================================================
def combined_loss(y_true, y_pred, alpha=0.8):
    """80% MSE + 20% (1 - SSIM)
    El MSE penaliza errores de pixeles, el SSIM penaliza perdida estructural."""
    mse      = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return alpha * mse + (1.0 - alpha) * (1.0 - ssim_val)

def compute_enl(img):
    """ENL = (media / desviacion)^2
    Mide homogeneidad en zonas uniformes. Mas alto = menos ruido residual."""
    mu, sig = np.mean(img), np.std(img)
    return (mu / sig) ** 2 if sig > 0 else 0.0

def full_metrics(y_true, y_pred, name=''):
    ssim_vals, psnr_vals, enl_vals = [], [], []
    for yt, yp in zip(y_true, y_pred):
        yt2 = yt[..., 0]; yp2 = yp[..., 0]
        ssim_vals.append(ssim(yt2, yp2, data_range=1.0))
        psnr_vals.append(psnr(yt2, yp2, data_range=1.0))
        enl_vals.append(compute_enl(yp2))
    res = {'SSIM': np.mean(ssim_vals),
           'PSNR': np.mean(psnr_vals),
           'ENL':  np.mean(enl_vals)}
    if name:
        print(f'\n{name}:')
        for k, v in res.items():
            print(f'  {k}: {v:.4f}')
    return res

# ==============================================================================
# 6. AUTOENCODER BASE
# ==============================================================================
def build_autoencoder_base(input_shape=(128, 128, 1)):
    inp = Input(shape=input_shape, name='input_sar')

    
    x = Conv2D(32,  (3,3), padding='same')(inp)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(32,  (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)            

    x = Conv2D(64,  (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(64,  (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)        

    x = Conv2D(128, (3,3), padding='same')(x)     
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)


    x = UpSampling2D((2,2))(x)                  
    x = Conv2D(64,  (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(64,  (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)

    x = UpSampling2D((2,2))(x)                   
    x = Conv2D(32,  (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(32,  (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)

    out = Conv2D(1, (1,1), activation='sigmoid', name='output')(x)
    return Model(inp, out, name='Autoencoder_Base')

model_ae = build_autoencoder_base(input_shape=(IMG_SIZE, IMG_SIZE, 1))
model_ae.summary()
model_ae.compile(optimizer=Adam(1e-3), loss=combined_loss, metrics=['mae'])

cb_ae = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss'),
]

print('\n--- Entrenando Autoencoder Base (max 50 epocas) ---')
H_ae = model_ae.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=cb_ae,
)

# ==============================================================================
# 7. U-NET (Arquitectura no secuencial con skip connections)
# ==============================================================================
def conv_block(x, filters, dropout_rate=0.0):
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def build_unet(input_shape=(128, 128, 1), filters_base=32):
    inp = Input(shape=input_shape, name='input_sar')

    c1 = conv_block(inp, filters_base)            
    p1 = MaxPooling2D((2,2))(c1)

    c2 = conv_block(p1,  filters_base * 2)         
    p2 = MaxPooling2D((2,2))(c2)

    c3 = conv_block(p2,  filters_base * 4)         
    p3 = MaxPooling2D((2,2))(c3)

    c4 = conv_block(p3,  filters_base * 8)          
    p4 = MaxPooling2D((2,2))(c4)

  
    bn = conv_block(p4, filters_base * 16,
                    dropout_rate=0.3)                


    u6 = Conv2DTranspose(filters_base * 8, (2,2), strides=(2,2), padding='same')(bn)
    u6 = Concatenate()([u6, c4])
    c6 = conv_block(u6, filters_base * 8)

    u7 = Conv2DTranspose(filters_base * 4, (2,2), strides=(2,2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = conv_block(u7, filters_base * 4)

    u8 = Conv2DTranspose(filters_base * 2, (2,2), strides=(2,2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = conv_block(u8, filters_base * 2)

    u9 = Conv2DTranspose(filters_base,     (2,2), strides=(2,2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = conv_block(u9, filters_base)

    out = Conv2D(1, (1,1), activation='sigmoid', name='output')(c9)
    return Model(inp, out, name='UNet_Despeckle')

model_unet = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1), filters_base=32)
model_unet.summary()
model_unet.compile(optimizer=Adam(5e-4), loss=combined_loss, metrics=['mae'])

cb_unet = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, monitor='val_loss'),
]

print('\n--- Entrenando U-Net (max 60 epocas) ---')
H_unet = model_unet.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=60,
    batch_size=BATCH_SIZE,
    callbacks=cb_unet,
)

# ==============================================================================
# 8. PREDICCIONES
# ==============================================================================
print('\nGenerando predicciones...')
pred_ae   = np.clip(model_ae.predict(X_test,   batch_size=BATCH_SIZE), 0.0, 1.0)
pred_unet = np.clip(model_unet.predict(X_test, batch_size=BATCH_SIZE), 0.0, 1.0)

# ==============================================================================
# 9. METRICAS CUANTITATIVAS
# ==============================================================================
metrics_noisy = full_metrics(Y_test, X_test,   'Imagen Ruidosa (sin filtrar)')
metrics_ae    = full_metrics(Y_test, pred_ae,   'Autoencoder Base')
metrics_unet  = full_metrics(Y_test, pred_unet, 'U-Net')

print('\n' + '='*60)
print('RESUMEN DE METRICAS')
print('='*60)
print(f'{"Modelo":<25} {"SSIM":>8} {"PSNR (dB)":>12} {"ENL":>10}')
print('-'*60)
for name, m in [('Ruidosa (sin filtrar)', metrics_noisy),
                ('Autoencoder Base',      metrics_ae),
                ('U-Net',                metrics_unet)]:
    print(f'{name:<25} {m["SSIM"]:>8.4f} {m["PSNR"]:>12.4f} {m["ENL"]:>10.4f}')
print('='*60)

# ==============================================================================
# 10. VISUALIZACIONES
# ==============================================================================
idx_show = 0
colores  = ['tomato', 'steelblue', 'darkorchid', 'seagreen']
titulos  = ['Imagen Ruidosa\n(entrada SAR)',
            'Autoencoder Base\n(filtrada)',
            'U-Net\n(filtrada)',
            'Ground Truth\n(referencia)']


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Curvas de Entrenamiento — Autoencoder Base vs U-Net',
             fontsize=14, fontweight='bold')

for row, (H, name, color) in enumerate([
    (H_ae,   'Autoencoder Base', 'steelblue'),
    (H_unet, 'U-Net',            'darkorchid'),
]):
    axes[row, 0].plot(H.history['loss'],     color=color, lw=2, label='Train')
    axes[row, 0].plot(H.history['val_loss'], color=color, lw=2, ls='--', label='Validacion')
    axes[row, 0].set_title(f'{name} — Loss (perdida)')
    axes[row, 0].set_xlabel('Epoca'); axes[row, 0].set_ylabel('Loss')
    axes[row, 0].legend(); axes[row, 0].grid(alpha=.35)

    axes[row, 1].plot(H.history['mae'],     color='tomato', lw=2, label='Train')
    axes[row, 1].plot(H.history['val_mae'], color='tomato', lw=2, ls='--', label='Validacion')
    axes[row, 1].set_title(f'{name} — MAE (error absoluto medio)')
    axes[row, 1].set_xlabel('Epoca'); axes[row, 1].set_ylabel('MAE')
    axes[row, 1].legend(); axes[row, 1].grid(alpha=.35)

plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, 'curvas_entrenamiento.png'), dpi=130, bbox_inches='tight')
plt.show()


n_show = min(3, len(X_test))

fig = plt.figure(figsize=(18, 6 * n_show))
fig.suptitle('Comparacion Visual — Ruidosa · Autoencoder · U-Net · Ground Truth',
             fontsize=14, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(n_show, 4, figure=fig, hspace=0.45, wspace=0.3)

for row in range(n_show):
    imgs = [X_test[row, ..., 0], pred_ae[row, ..., 0],
            pred_unet[row, ..., 0], Y_test[row, ..., 0]]
    for col, (img, tit, col_c) in enumerate(zip(imgs, titulos, colores)):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(tit, fontsize=10, pad=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(col_c)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

plt.savefig(os.path.join(OUTPUT_DIR, 'comparacion_visual.png'), dpi=130, bbox_inches='tight')
plt.show()


cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
half   = IMG_SIZE // 4
y1, y2 = cy - half, cy + half
x1, x2 = cx - half, cx + half

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Zoom — Analisis de Bordes y Detalles (region central)',
             fontsize=13, fontweight='bold', y=1.02)

for ax, img, tit, col_c in zip(axes,
    [X_test[idx_show, ..., 0], pred_ae[idx_show, ..., 0],
     pred_unet[idx_show, ..., 0], Y_test[idx_show, ..., 0]],
    ['Ruidosa', 'Autoencoder Base', 'U-Net', 'Ground Truth'],
    colores
):
    ax.imshow(img[y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
    ax.set_title(tit, fontsize=11, pad=8, color=col_c, fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(col_c)
        spine.set_linewidth(2.5)
        spine.set_visible(True)

plt.tight_layout(pad=2.5, w_pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, 'zoom_detalles.png'), dpi=130, bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Mapa de Error Absoluto — |Prediccion - Ground Truth|',
             fontsize=13, fontweight='bold')

for ax, pred, name, cmap in zip(
    axes,
    [pred_ae[idx_show, ..., 0], pred_unet[idx_show, ..., 0]],
    ['Autoencoder Base', 'U-Net'],
    ['Blues', 'Purples']
):
    diff = np.abs(pred - Y_test[idx_show, ..., 0])
    im   = ax.imshow(diff, cmap=cmap, vmin=0, vmax=0.3)
    ax.set_title(f'{name}\nerror promedio = {diff.mean():.4f}', fontsize=11, pad=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout(pad=2.5, w_pad=3.5)
plt.savefig(os.path.join(OUTPUT_DIR, 'mapa_errores.png'), dpi=130, bbox_inches='tight')
plt.show()

psnr_max = max(metrics_noisy['PSNR'], metrics_ae['PSNR'], metrics_unet['PSNR']) + 1
enl_max  = max(metrics_noisy['ENL'],  metrics_ae['ENL'],  metrics_unet['ENL'])  + 1

modelos   = ['Ruidosa\n(sin filtrar)', 'Autoencoder\nBase', 'U-Net']
colores_b = ['tomato', 'steelblue', 'darkorchid']
raw_vals  = [
    [metrics_noisy['SSIM'], metrics_noisy['PSNR'], metrics_noisy['ENL']],
    [metrics_ae['SSIM'],    metrics_ae['PSNR'],    metrics_ae['ENL']],
    [metrics_unet['SSIM'],  metrics_unet['PSNR'],  metrics_unet['ENL']],
]
norm_vals = [[v[0], v[1] / psnr_max, v[2] / enl_max] for v in raw_vals]

x = np.arange(3)
w = 0.25
fig, ax = plt.subplots(figsize=(12, 5))

for i, (modelo, col_b, nv, rv) in enumerate(zip(modelos, colores_b, norm_vals, raw_vals)):
    bars = ax.bar(x + (i - 1) * w, nv, w, label=modelo,
                  color=col_b, edgecolor='white')
    for bar, vr in zip(bars, rv):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{vr:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['SSIM\n(mas alto = mejor)',
                    'PSNR\n(normalizado, mas alto = mejor)',
                    'ENL\n(normalizado, mas alto = mejor)'], fontsize=10)
ax.set_ylabel('Valor (normalizado para comparar)')
ax.set_title('Comparacion de Metricas SAR — SSIM · PSNR · ENL',
             fontweight='bold', pad=12)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(pad=2.5)
plt.savefig(os.path.join(OUTPUT_DIR, 'comparacion_metricas.png'), dpi=130, bbox_inches='tight')
plt.show()


def detect_edges(img_norm, low=50, high=150):
    """Canny con umbrales absolutos sobre imagen uint8"""
    return cv2.Canny((img_norm * 255).astype(np.uint8), low, high)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Deteccion de Bordes (Canny) — Preservacion de Detalles',
             fontsize=13, fontweight='bold', y=1.02)

for ax, img, tit, col_c in zip(axes,
    [X_test[idx_show, ..., 0], pred_ae[idx_show, ..., 0],
     pred_unet[idx_show, ..., 0], Y_test[idx_show, ..., 0]],
    ['Ruidosa', 'Autoencoder Base', 'U-Net', 'Ground Truth'],
    colores
):
    ax.imshow(detect_edges(img), cmap='gray')
    ax.set_title(tit, fontsize=11, pad=8, color=col_c, fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(col_c)
        spine.set_linewidth(2.5)
        spine.set_visible(True)

plt.tight_layout(pad=2.5, w_pad=3.0)
plt.savefig(os.path.join(OUTPUT_DIR, 'analisis_bordes.png'), dpi=130, bbox_inches='tight')
plt.show()

# ==============================================================================
# 11. GUARDAR MODELOS
# ==============================================================================
model_ae.save(os.path.join(OUTPUT_DIR, 'modelo_autoencoder_base.keras'))
model_unet.save(os.path.join(OUTPUT_DIR, 'modelo_unet_despeckle.keras'))
print('\nModelos guardados en:', OUTPUT_DIR)
print('Graficas guardadas en:', OUTPUT_DIR)
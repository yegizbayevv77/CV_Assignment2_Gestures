"""Gradio demo: hand gesture recognition (peace / stop / thumbs_up).

Pipeline: image -> MediaPipe hand crop -> CNN prediction -> Grad-CAM explanation.
Run: python app.py   (then open the printed local address)
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224
CLASS_NAMES = ['peace', 'stop', 'thumbs_up']
MODEL_PATH = 'a2_crop_model.keras'   # saved by the notebook
HAND_MODEL_PATH = 'hand_landmarker.task'  # downloaded by the README step 2

crop_model = keras.models.load_model(MODEL_PATH)
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=1, min_hand_detection_confidence=0.5))


def crop_hand(img_rgb, pad=0.3, size=IMG_SIZE):
    """Square crop around the detected hand, or None if no hand is found."""
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if not res.hand_landmarks:
        return None
    lm = res.hand_landmarks[0]
    h, w = img_rgb.shape[:2]
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    s = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * pad)
    x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    x1, y1 = int(min(cx + s / 2, w)), int(min(cy + s / 2, h))
    return cv2.resize(img_rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def gradcam(model, img, class_idx=None):
    """Grad-CAM heatmap (IMG_SIZE x IMG_SIZE, values 0..1) for the last conv layer."""
    conv = [l for l in model.layers if isinstance(l, layers.Conv2D)][-1]
    grad_model = keras.Model(model.input, [conv.output, model.output])
    x = tf.cast(img[None], tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0].numpy()
    return cam, class_idx


def predict(img):
    if img is None:
        return None, None, 'Upload an image or use the webcam.'
    t0 = time.time()
    img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    crop = crop_hand(img)
    note = 'Hand found'
    if crop is None:
        crop = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        note = 'No hand found: the whole image was used'
    probs = crop_model(crop[None].astype('float32'), training=False).numpy()[0]
    cam, k = gradcam(crop_model, crop.astype('float32'))
    heat = plt.cm.jet(cam)[..., :3]
    overlay = np.clip(0.6 * crop / 255.0 + 0.4 * heat, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    label = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    ms = (time.time() - t0) * 1000
    info = f'{note}. Prediction: {CLASS_NAMES[k]}. Time: {ms:.0f} ms (MediaPipe + CNN + Grad-CAM).'
    return label, overlay, info


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=['upload', 'webcam'], type='numpy', label='Hand image'),
    outputs=[gr.Label(num_top_classes=3, label='Prediction'),
             gr.Image(label='Grad-CAM (which regions matter)'),
             gr.Textbox(label='Info')],
    title='Hand gesture recognition: peace / stop / thumbs_up',
    description='Assignment 2: MediaPipe hand crop + CNN + Grad-CAM explanation.',
)

if __name__ == '__main__':
    demo.launch()

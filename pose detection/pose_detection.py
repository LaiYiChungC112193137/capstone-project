# movenet_rigorous.py
import os
import math
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# === USER: set your image path here ===
IMAGE_PATH = "/content/test.jpg"
OUT_PATH = "movenet_result.png"
CONF_THRESHOLD = 0.3

# Model selection: lightning or thunder TF Hub URL
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

def load_movenet(model_url=MODEL_URL):
    model = hub.load(model_url)
    signature = model.signatures['serving_default']
    # infer expected input size from signature if available
    inp = signature.structured_input_signature[1]
    key = list(inp.keys())[0]
    shape = inp[key].shape.as_list()  # e.g., [1,256,256,3]
    if shape[1] is None or shape[2] is None:
        target_size = (256, 256)
    else:
        target_size = (shape[2], shape[1])  # (width, height)
    return model, signature, target_size

def letterbox_image(img, target_size, fill_color=(0,0,0)):
    # img is PIL Image in RGB
    src_w, src_h = img.size
    tgt_w, tgt_h = target_size
    scale = min(tgt_w / src_w, tgt_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    padded = Image.new("RGB", (tgt_w, tgt_h), fill_color)
    padded.paste(resized, (pad_left, pad_top))
    meta = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "src_size": (src_w, src_h),
        "resized_size": (new_w, new_h)
    }
    return padded, meta

def preprocess_image(path, target_size):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect orientation
    if img.mode == "RGBA":
        # drop alpha by compositing on white
        bg = Image.new("RGB", img.size, (255,255,255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode == "L":
        img = img.convert("RGB")
    else:
        img = img.convert("RGB")
    img_letterbox, meta = letterbox_image(img, target_size)
    arr = np.array(img_letterbox).astype(np.int32)
    input_tensor = tf.expand_dims(arr, axis=0)
    return img, input_tensor, meta

def postprocess_keypoints(keypoints_with_scores, meta):
    # keypoints_with_scores shape [1,1,17,3] with (y,x,score) normalized to [0,1] relative to model input
    kps = keypoints_with_scores[0,0,:,:]  # (17,3)
    scale = meta["scale"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]
    src_w, src_h = meta["src_size"]
    tgt_w = int(round(src_w * scale)) + 2*pad_left
    tgt_h = int(round(src_h * scale)) + 2*pad_top
    mapped = []
    for (y, x, s) in kps:
        # coordinates relative to model input -> remove padding -> divide by scale -> to original image coords
        x_model = x * tgt_w
        y_model = y * tgt_h
        x_unpad = (x_model - pad_left) / scale
        y_unpad = (y_model - pad_top) / scale
        # clamp to image bounds
        x_clamped = float(np.clip(x_unpad, 0.0, src_w - 1.0))
        y_clamped = float(np.clip(y_unpad, 0.0, src_h - 1.0))
        mapped.append((x_clamped, y_clamped, float(s)))
    return np.array(mapped)  # (17,3)

def draw_keypoints(orig_img, mapped_kps, threshold=CONF_THRESHOLD):
    img = np.array(orig_img).copy()
    h, w = img.shape[:2]
    for (x, y, s) in mapped_kps:
        if s < threshold:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 4, (0,255,255), -1)
    # optional: draw skeleton edges (same mapping as TF tutorial)
    edges = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]
    for (i,j) in edges:
        xi, yi, si = mapped_kps[i]
        xj, yj, sj = mapped_kps[j]
        if si < threshold or sj < threshold:
            continue
        cv2.line(img, (int(round(xi)), int(round(yi))), (int(round(xj)), int(round(yj))), (0,128,255), 2)
    return img

def run(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    model, signature, target_size = load_movenet()
    orig_img, input_tensor, meta = preprocess_image(image_path, target_size)
    outputs = signature(tf.constant(input_tensor))
    # output key name may be 'output_0'
    out_key = list(outputs.keys())[0]
    kps = outputs[out_key].numpy()
    mapped = postprocess_keypoints(kps, meta)
    vis = draw_keypoints(orig_img, mapped)
    cv2.imwrite(OUT_PATH, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {OUT_PATH}")

if __name__ == "__main__":
    run(IMAGE_PATH)

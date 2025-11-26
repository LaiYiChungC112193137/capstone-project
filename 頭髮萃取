# hair_extract_and_show.py
# Requirements:
# pip install transformers torch torchvision pillow opencv-python matplotlib

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt

# --- Paths (as requested) ---
INPUT_PATH = "/content/OIP.png"
OUTPUT_PATH = "/content/hair_output.png"
# ---------------------------

# Model and hair class (change model if you prefer another face-parsing checkpoint)
MODEL_NAME = "jonathandinu/face-parsing"
HAIR_CLASS_ID = 13  # common CelebAMask-HQ hair index; adjust if your model uses a different mapping

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(name):
    processor = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).to(device)
    return processor, model

def predict_parsing(img_pil, processor, model):
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    up = torch.nn.functional.interpolate(logits, size=img_pil.size[::-1], mode="bilinear", align_corners=False)
    seg = up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return seg

def make_trimap(mask, fg_erode=10, bg_dilate=12):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_fg = cv2.erode(mask, kernel, iterations=fg_erode)
    sure_bg = cv2.dilate(mask, kernel, iterations=bg_dilate)
    sure_bg = cv2.bitwise_not(sure_bg)
    unknown = cv2.subtract(255 - sure_bg, sure_fg)
    trimap = np.zeros_like(mask)
    trimap[sure_bg==255] = 0
    trimap[unknown==255] = 128
    trimap[sure_fg==255] = 255
    return trimap, sure_fg, sure_bg, unknown

def alpha_from_trimap_force_unknown_transparent(trimap):
    alpha = np.zeros_like(trimap, dtype=np.uint8)
    alpha[trimap==255] = 255
    # Slightly smooth foreground edges while keeping unknown fully transparent
    alpha = cv2.GaussianBlur(alpha, (5,5), 0)
    _, alpha = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    return alpha

def extract_hair_and_save(input_path, output_path):
    img_pil = Image.open(input_path).convert("RGB")
    processor, model = load_model(MODEL_NAME)
    seg = predict_parsing(img_pil, processor, model)
    hair_mask = (seg == HAIR_CLASS_ID).astype(np.uint8) * 255

    trimap, sure_fg, sure_bg, unknown = make_trimap(hair_mask, fg_erode=10, bg_dilate=12)
    alpha = alpha_from_trimap_force_unknown_transparent(trimap)

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    b,g,r = cv2.split(img_cv)
    rgba = cv2.merge([b,g,r, alpha])
    out = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
    out.save(output_path, format="PNG")
    print("Saved hair image to:", output_path)
    return img_pil, out

def show_side_by_side(input_img_pil, output_img_pil):
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    axes[0].imshow(input_img_pil)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Display output with transparency over a subtle background for clarity
    bg = Image.new("RGBA", output_img_pil.size, (255,255,255,255))
    composed = Image.alpha_composite(bg, output_img_pil.convert("RGBA"))
    axes[1].imshow(composed)
    axes[1].set_title("Extracted Hair (transparent regions shown over white)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_img, output_img = extract_hair_and_save(INPUT_PATH, OUTPUT_PATH)
    show_side_by_side(input_img, output_img)

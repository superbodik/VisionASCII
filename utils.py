import os
import datetime
from PIL import Image
import cv2

def ensure_dirs():
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/ascii", exist_ok=True)

def save_frame_and_ascii(frame, ascii_str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = f"dataset/images/{timestamp}.png"
    ascii_path = f"dataset/ascii/{timestamp}.txt"

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img.save(image_path)

    with open(ascii_path, "w", encoding="utf-8") as f:
        f.write(ascii_str)

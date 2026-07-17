from PIL import Image
import numpy as np
from pathlib import Path


def load_image(image_path):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path)
    if img.format not in ['JPEG', 'PNG']:
        raise ValueError(f"Unsupported format: {img.format}. Only JPEG and PNG are supported.")
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img)


def save_image(image_array, output_path):
    img = Image.fromarray(image_array.astype(np.uint8))
    ext = Path(output_path).suffix.lower()
    kwargs = {"quality": 95} if ext in (".jpg", ".jpeg") else {}
    img.save(output_path, **kwargs)


def validate_image_format(file_path):
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    ext = Path(file_path).suffix.lower()
    return ext in valid_extensions


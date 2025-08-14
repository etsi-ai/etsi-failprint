import pandas as pd
from PIL import Image, ImageStat
import numpy as np

def extract_image_features(image_path: str) -> dict:
    """Calculates basic statistical features for a single image."""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale for brightness/contrast calculation
            grayscale_img = img.convert("L")
            stats = ImageStat.Stat(grayscale_img)

            width, height = img.size

            return {
                "brightness": np.mean(stats.mean),
                "contrast": np.mean(stats.stddev),
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height > 0 else 0,
            }
    except Exception as e:
        print(f"Warning: Could not process image {image_path}. Error: {e}")
        return {
            "brightness": None, "contrast": None, "width": None,
            "height": None, "aspect_ratio": None
        }

def build_cv_feature_df(image_paths: list) -> pd.DataFrame:
    """
    Creates a DataFrame of CV features from a list of image paths.
    """
    feature_list = [extract_image_features(path) for path in image_paths]
    return pd.DataFrame(feature_list)
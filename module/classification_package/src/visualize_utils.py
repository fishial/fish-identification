import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom
from matplotlib import cm
import torch

def save_attention_overlay(image_pil, attention_map, save_path, title=None):
    """
    Scales an attention map to the size of a PIL image, overlays it, and saves the result as a PNG.
    If a title is provided, it is added as a watermark.

    :param image_pil: A PIL.Image object (in RGB format).
    :param attention_map: A torch.Tensor of shape [1, H', W'] or [1, 1, H', W'].
    :param save_path: The path to save the output PNG file.
    :param title: An optional title to add to the image.
    :return: The resized attention map as a numpy.ndarray with shape H x W.
    """
    # Squeeze dimensions if the tensor is 4D or 3D
    if attention_map.ndim == 4:
        attention_map = attention_map[0, 0]
    elif attention_map.ndim == 3:
        attention_map = attention_map[0]

    # Convert PIL image to a float32 numpy array (RGB)
    image_np = np.array(image_pil).astype(np.float32) / 255.0  # Shape: [H, W, 3]
    h, w = image_np.shape[:2]

    # Normalize the attention map
    attention = attention_map.cpu().detach().numpy()
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-5)
    attention = np.clip(attention, 0, 1)

    # Resize the attention map to the image's dimensions using bicubic interpolation
    attention_resized = zoom(attention, (h / attention.shape[0], w / attention.shape[1]), order=1)

    # Generate a heatmap from the attention map
    heatmap = cm.get_cmap('jet')(attention_resized)[..., :3]  # Get RGB channels

    # Blend the heatmap with the original image
    overlay = 0.5 * image_np + 0.5 * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    # Convert the result back to a PIL image
    overlay_pil = Image.fromarray(overlay)

    # Add the title if one is provided
    if title:
        draw = ImageDraw.Draw(overlay_pil)
        try:
            # Try to use a common system font
            font = ImageFont.truetype("arial.ttf", size=max(h // 30, 12))
        except IOError:
            # Fallback to the default PIL font if Arial is not found
            font = ImageFont.load_default()

        text_position = (10, 10)
        text_color = (255, 255, 255)  # White
        outline_color = (0, 0, 0)     # Black

        # Add a black outline for better readability
        draw.text((text_position[0] - 1, text_position[1]), title, font=font, fill=outline_color)
        draw.text((text_position[0] + 1, text_position[1]), title, font=font, fill=outline_color)
        draw.text((text_position[0], text_position[1] - 1), title, font=font, fill=outline_color)
        draw.text((text_position[0], text_position[1] + 1), title, font=font, fill=outline_color)

        # Draw the main text
        draw.text(text_position, title, font=font, fill=text_color)

    # Save the final image
    # Ensure the directory for the output file exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    overlay_pil.save(save_path)

    return attention_resized
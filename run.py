import sys
import numpy as np
import cv2
import argparse
import os
from pathlib import Path
import pdb

import numpy as np
from PIL import Image
import pdb
import torch
import pdb

# ============================================================================
# CUSTOM LLAVA_API IMPLEMENTATION (Enhanced version from pal environment)
# ============================================================================

## Imports for custom llava_api
import time, base64, requests, json, datetime
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# WEB SERVER IMPORTS
# ============================================================================
try:
    from flask import Flask, request, render_template_string, send_from_directory, jsonify
    from werkzeug.utils import secure_filename
    import io
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

# Try to import Colab-specific modules
COLAB_MODULE_AVAILABLE = False
try:
    import google.colab
    COLAB_MODULE_AVAILABLE = True
    IN_COLAB = True
    print("Successfully imported google.colab")
except ImportError:
    IN_COLAB = False
    print("google.colab not available")

# Additional Colab detection
try:
    import os
    # Check for Colab environment variables or paths
    colab_indicators = [
        'COLAB_GPU' in os.environ,
        'COLAB_TPU_ADDR' in os.environ,
        '/content/' in os.getcwd(),
        os.path.exists('/content'),
        'COLAB_RELEASE_TAG' in os.environ
    ]

    if any(colab_indicators):
        IN_COLAB = True
        print("Detected Colab via environment variables")
    else:
        print("No Colab indicators found")
except Exception as e:
    print(f"Error in Colab detection: {e}")
    pass

# Try to import ngrok for tunneling (optional)
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

# Cloudflare tunnel imports (much simpler!)
try:
    import atexit, requests, subprocess, time, re, os
    from random import randint
    from threading import Timer
    from queue import Queue
    CLOUDFLARE_AVAILABLE = True
except ImportError:
    CLOUDFLARE_AVAILABLE = False

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import torchvision.transforms as T

def cloudflared_tunnel(port, metrics_port, output_queue):
    """Create a Cloudflare tunnel for the given port"""
    atexit.register(lambda p: p.terminate(), subprocess.Popen(['cloudflared', 'tunnel', '--url', f'http://127.0.0.1:{port}', '--metrics', f'127.0.0.1:{metrics_port}'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
    attempts, tunnel_url = 0, None
    while attempts < 10 and not tunnel_url:
        attempts += 1
        time.sleep(3)
        try:
            tunnel_url = re.search("(?P<url>https?:\/\/[^\s]+.trycloudflare.com)", requests.get(f'http://127.0.0.1:{metrics_port}/metrics').text).group("url")
        except:
            pass
    if not tunnel_url:
        raise Exception("Can't connect to Cloudflare Edge")
    output_queue.put(tunnel_url)

# Custom helper functions
def readImg(p):
    return Image.open(p)

def toImg(t):
    return T.ToPILImage()(t)

def invtrans(mask, image, method=Image.BICUBIC):
    return mask.resize(image.size, method)

def merge(mask, image, grap_scale=200):
    gray = np.ones((image.size[1], image.size[0], 3))*grap_scale
    image_np = np.array(image).astype(np.float32)[..., :3]
    mask_np = np.array(mask).astype(np.float32)
    mask_np = mask_np / 255.0
    blended_np = image_np * mask_np[:, :, None]  + (1 - mask_np[:, :, None]) * gray
    blended_image = Image.fromarray((blended_np).astype(np.uint8))
    return blended_image

def normalize(mat, method="max"):
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    else:
        raise NotImplementedError

def enhance(mat, coe=10):
    mat = mat - mat.mean()
    mat = mat / mat.std()
    mat = mat * coe
    mat = torch.sigmoid(mat)
    mat = mat.clamp(0,1)
    return mat

def revise_mask(patch_mask, kernel_size=3, enhance_coe=10):
    patch_mask = normalize(patch_mask, "min")
    patch_mask = enhance(patch_mask, coe=enhance_coe)

    assert kernel_size % 2 == 1
    padding_size = int((kernel_size - 1) / 2)
    conv = torch.nn.Conv2d(1,1,kernel_size=kernel_size, padding=padding_size, padding_mode="replicate", stride=1, bias=False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size**2
    conv.to(patch_mask.device)

    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = patch_mask
    return mask

def blend_mask(image_path_or_pil_image, mask, enhance_coe, kernel_size, interpolate_method, grayscale):
    mask = revise_mask(mask.float(), kernel_size=kernel_size, enhance_coe=enhance_coe)
    mask = mask.detach().cpu()
    mask = toImg(mask.reshape(1,24,24))

    if isinstance(image_path_or_pil_image, str):
        image = readImg(image_path_or_pil_image)
    elif isinstance(image_path_or_pil_image, Image.Image):
        image = image_path_or_pil_image
    else:
        raise NotImplementedError

    # Resize mask to match image
    mask = invtrans(mask, image, method=interpolate_method)
    # Convert mask to normalized grayscale uint8
    mask_np = np.array(mask.convert("L")).astype(np.float32)
    mask_norm = cv2.normalize(mask_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply Jet colormap to mask
    heatmap_bgr = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
    # Prepare original image as BGR NumPy array
    if isinstance(image_path_or_pil_image, str):
        orig_bgr = cv2.imread(image_path_or_pil_image)
    else:
        orig_np = np.array(image.convert("RGB"))
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    # Determine blending alpha (use grayscale parameter if valid, else default 0.5)
    alpha = grayscale if (isinstance(grayscale, (int, float)) and 0 < grayscale <= 1) else 0.5
    # Overlay the heatmap onto the original image
    overlay_bgr = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    merged_image = Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    return merged_image, mask

# ============================================================================
# CUSTOM LLAVA_API IMPLEMENTATION (Enhanced version from pal environment)
# ============================================================================

# Import the base functions we need for our custom implementation
from apiprompting.api_llava.functions import getmask, get_model
from apiprompting.api_llava.hook import hook_logger

# Override get_model with a cached, quantization-aware loader for Colab/T4/L4
# Uses apillava.model.builder directly to enable 4-bit/8-bit when needed.
_MODEL_CACHE = {}

def _detect_quantization_preference():
    try:
        q = os.getenv("LLAVA_QUANT", "").strip().lower()
        if q in ("4bit", "8bit", "fp16"):
            return q
        # Heuristic: T4 (16GB) → 4bit by default; otherwise fp16
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            if total_gb <= 17.0:
                return "4bit"
    except Exception:
        pass
    return "fp16"

def get_model(model_name):
    quant = _detect_quantization_preference()
    cache_key = (model_name, quant)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Lazy import to avoid hard dependency at module import time
    from apillava.model.builder import load_pretrained_model
    from apillava.mm_utils import get_model_name_from_path

    model_path = f"liuhaotian/{model_name}"
    model_base = None
    inner_name = get_model_name_from_path(model_path)

    load_kwargs = {"device_map": "auto", "device": "cuda"}
    if quant == "4bit":
        load_kwargs.update({"load_4bit": True})
    elif quant == "8bit":
        load_kwargs.update({"load_8bit": True})
    # else default fp16 via builder

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=inner_name,
        **load_kwargs,
    )

    _MODEL_CACHE[cache_key] = (tokenizer, model, image_processor, context_len, inner_name)
    return _MODEL_CACHE[cache_key]

def custom_llava_api(images, queries, model_name, batch_size=1, layer_index=20, enhance_coe=10, kernel_size=3, interpolate_method_name="LANCZOS", grayscale=0):
    """
    Generates image masks and blends them using the specified model and parameters.

    Parameters:
    images (list): list of images. Each item can be a path to image (str) or a PIL.Image.
    queries (list): list of queries. Each item is a str.
    batch_size (int): Batch size for processing images. Only support 1.
    model_name (str): Name of the model to load the pretrained model. One of "llava-v1.5-7b" and "llava-v1.5-13b".
    layer_index (int): Index of the layer in the model to hook. Default is 20.
    enhance_coe (int): Enhancement coefficient for mask blending. Default is 10.
    kernel_size (int): Kernel size for mask blending. Should be odd numbers. Default is 3.
    interpolate_method_name (str): Name of the interpolation method for image processing. Can be any interpolation method supported by PIL.Image.resize. Default is "LANCZOS".
    grayscale (float): Whether to convert the image to grayscale. Default is 0.

    Returns:
    tuple: A tuple containing four lists:
        - masked_images: A list of the masked images. Each item is a PIL.Image.
        - attention_maps: A list of the attention maps as torch tensors with shape (1, 1, 24, 24).
        - mota_masks: A list of processed masks.
        - text_answers: A list of text answers from the LLaVA model to the queries.
    """

    tokenizer, model, image_processor, context_len, inner_model_name = get_model(model_name)
    hl = hook_logger(model, model.device, layer_index=layer_index)

    interpolate_method = getattr(Image, interpolate_method_name)
    masked_images = []
    attention_maps = []
    mota_masks = []
    text_answers = []

    for image_path_or_pil_image, question in zip(images, queries):
        with torch.no_grad():
            mask_args = type('Args', (), {
                    "hl":      hl,
                    "model_name": model_name,
                    "model": model,
                    "tokenizer": tokenizer,
                    "image_processor": image_processor,
                    "context_len": context_len,
                    "query": question,
                    "conv_mode": None,
                    "image_file": image_path_or_pil_image,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 20,
                })()
            mask, text_output = getmask(mask_args)
            # Convert mask of shape (24, 24) to (1, 1, 24, 24)
            attention_map = mask.clone().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 24, 24)
            attention_maps.append(attention_map)  # Store the reshaped attention map
            merged_image, mota_mask = blend_mask(image_path_or_pil_image, mask, enhance_coe, kernel_size, interpolate_method, grayscale)
            masked_images.append(merged_image)
            mota_masks.append(mota_mask)
            text_answers.append(text_output)  # Store the text answer

    return masked_images, attention_maps, mota_masks, text_answers

# ============================================================================
# END CUSTOM LLAVA_API IMPLEMENTATION
# ============================================================================

# ============================================================================
# HTML TEMPLATE FOR WEB INTERFACE
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaVA Attention Warping Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #818cf8;
            --secondary: #ec4899;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1e293b;
            --gray-50: #f8fafc;
            --gray-100: #f1f5f9;
            --gray-200: #e2e8f0;
            --gray-300: #cbd5e1;
            --gray-400: #94a3b8;
            --gray-500: #64748b;
            --gray-600: #475569;
            --gray-700: #334155;
            --gray-800: #1e293b;
            --gray-900: #0f172a;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem 1rem;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: var(--shadow-xl);
            overflow: hidden;
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            padding: 3rem 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 8s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.1) rotate(180deg); }
        }

        h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }

        .subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }

        .main-content {
            padding: 2.5rem;
        }

        .upload-zone {
            border: 3px dashed var(--gray-300);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            background: var(--gray-50);
            margin-bottom: 2rem;
        }

        .upload-zone:hover {
            border-color: var(--primary);
            background: linear-gradient(135deg, rgba(99,102,241,0.05) 0%, rgba(236,72,153,0.05) 100%);
            transform: translateY(-2px);
        }

        .upload-zone.dragover {
            border-color: var(--primary);
            background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(236,72,153,0.1) 100%);
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            display: inline-block;
            animation: float 3s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        .upload-text {
            color: var(--gray-600);
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }

        .upload-subtext {
            color: var(--gray-400);
            font-size: 0.9rem;
        }

        #image {
            display: none;
        }

        .preview-container {
            margin-top: 1.5rem;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow-md);
            display: none;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .preview-container img {
            width: 100%;
            display: block;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: var(--gray-700);
            font-size: 0.95rem;
            letter-spacing: 0.01em;
        }

        textarea {
            width: 100%;
            padding: 1rem;
            border: 2px solid var(--gray-200);
            border-radius: 12px;
            font-size: 1rem;
            font-family: inherit;
            transition: all 0.3s ease;
            resize: vertical;
            background: white;
        }

        textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99,102,241,0.1);
        }

        .parameter-section {
            background: var(--gray-50);
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
        }

        .parameter-section h3 {
            color: var(--gray-800);
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .parameter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
        }

        select, input[type="number"] {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 2px solid var(--gray-200);
            border-radius: 10px;
            font-size: 0.95rem;
            font-family: inherit;
            transition: all 0.3s ease;
            background: white;
        }

        select:focus, input[type="number"]:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99,102,241,0.1);
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem;
            background: white;
            border-radius: 10px;
            border: 2px solid var(--gray-200);
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .checkbox-group:hover {
            border-color: var(--primary-light);
            background: var(--gray-50);
        }

        .checkbox-group input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
            accent-color: var(--primary);
        }

        .checkbox-group label {
            margin: 0;
            cursor: pointer;
            flex: 1;
        }

        .btn {
            width: 100%;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            position: relative;
            overflow: hidden;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            box-shadow: var(--shadow-lg);
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl);
        }

        .btn-primary:active:not(:disabled) {
            transform: translateY(0);
        }

        .btn-primary:disabled {
            background: var(--gray-300);
            cursor: not-allowed;
            box-shadow: none;
        }

        .loading {
            text-align: center;
            padding: 3rem 2rem;
            display: none;
            animation: fadeIn 0.3s ease-in;
        }

        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid var(--gray-200);
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1.5rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            color: var(--gray-600);
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }

        .loading-subtext {
            color: var(--gray-400);
            font-size: 0.9rem;
        }

        .error {
            background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.1) 100%);
            border-left: 4px solid var(--danger);
            color: var(--danger);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin-top: 1rem;
            display: none;
            animation: shake 0.5s ease-in-out;
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-10px); }
            75% { transform: translateX(10px); }
        }

        .results {
            margin-top: 3rem;
            display: none;
            animation: fadeIn 0.5s ease-in;
        }

        .result-section {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--gray-200);
        }

        .result-section h3 {
            color: var(--gray-800);
            font-size: 1.4rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .text-answer {
            background: linear-gradient(135deg, rgba(99,102,241,0.05) 0%, rgba(236,72,153,0.05) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid var(--primary);
            line-height: 1.8;
        }

        .text-answer strong {
            color: var(--primary);
            display: block;
            margin-bottom: 0.5rem;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        .image-container {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--gray-200);
            transition: all 0.3s ease;
        }

        .image-container:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-xl);
        }

        .image-container h4 {
            background: linear-gradient(135deg, var(--gray-800) 0%, var(--gray-700) 100%);
            color: white;
            padding: 1rem;
            margin: 0;
            font-size: 1rem;
            font-weight: 600;
        }

        .image-container img {
            width: 100%;
            display: block;
            transition: transform 0.3s ease;
        }

        .image-container:hover img {
            transform: scale(1.02);
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--gray-500);
            font-size: 0.9rem;
            border-top: 1px solid var(--gray-200);
        }

        @media (max-width: 768px) {
            h1 {
                font-size: 1.75rem;
            }

            .main-content {
                padding: 1.5rem;
            }

            .parameter-grid {
                grid-template-columns: 1fr;
            }

            .image-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 LLaVA Attention Warping Studio</h1>
            <p class="subtitle">Transform images through the lens of AI attention</p>
        </div>

        <div class="main-content">
            <form id="warpForm" enctype="multipart/form-data">
                <div class="upload-zone" id="uploadZone">
                    <div class="upload-icon">📸</div>
                    <div class="upload-text">Click to upload or drag and drop</div>
                    <div class="upload-subtext">PNG, JPG, WEBP up to 16MB</div>
                    <input type="file" id="image" name="image" accept="image/*" required>
                    <div class="preview-container" id="previewContainer">
                        <img id="preview" alt="Preview">
                    </div>
                </div>

                <div class="form-group">
                    <label for="query">💬 What would you like to know?</label>
                    <textarea id="query" name="query" rows="3" placeholder="Ask anything about the image... e.g., 'What objects are on the desk?' or 'Describe the scene'" required></textarea>
                </div>

                <div class="parameter-section">
                    <h3>⚙️ Warping Parameters</h3>
                    <div class="parameter-grid">
                        <div class="form-group">
                            <label for="transform">Transform Function</label>
                            <select id="transform" name="transform">
                                <option value="identity">Identity (No transform)</option>
                                <option value="square">Square (x²)</option>
                                <option value="sqrt" selected>Square Root (√x)</option>
                                <option value="exp">Exponential (eˣ)</option>
                                <option value="log">Logarithmic (ln x)</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="exp_scale">Exponential Scale</label>
                            <input type="number" id="exp_scale" name="exp_scale" value="1.0" step="0.1" min="0.1" max="10">
                        </div>

                        <div class="form-group">
                            <label for="exp_divisor">Exponential Divisor</label>
                            <input type="number" id="exp_divisor" name="exp_divisor" value="1.0" step="0.1" min="0.1" max="10">
                        </div>

                        <div class="form-group">
                            <label for="attention_alpha">Attention Overlay Alpha</label>
                            <input type="number" id="attention_alpha" name="attention_alpha" value="0.4" step="0.05" min="0" max="1">
                        </div>
                    </div>

                    <div class="form-group" style="margin-top: 1rem;">
                        <div class="checkbox-group">
                            <input type="checkbox" id="apply_inverse" name="apply_inverse">
                            <label for="apply_inverse">Apply inverse transformation to marginal profiles</label>
                        </div>
                    </div>
                </div>

                <button type="submit" class="btn btn-primary">
                    ✨ Generate Attention Warp
                </button>
            </form>

            <div id="loading" class="loading">
                <div class="spinner"></div>
                <div class="loading-text">Processing with LLaVA...</div>
                <div class="loading-subtext">Analyzing attention patterns and warping image</div>
            </div>

            <div id="error" class="error"></div>

            <div id="results" class="results">
                <!-- Results will be populated here -->
            </div>
        </div>

        <div class="footer">
            Powered by LLaVA Vision-Language Model • Built with ❤️ for AI Research
        </div>
    </div>

    <script>
        // Image upload and preview
        const uploadZone = document.getElementById('uploadZone');
        const imageInput = document.getElementById('image');
        const preview = document.getElementById('preview');
        const previewContainer = document.getElementById('previewContainer');

        uploadZone.addEventListener('click', () => imageInput.click());

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                imageInput.files = files;
                showPreview(files[0]);
            }
        });

        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                showPreview(e.target.files[0]);
            }
        });

        function showPreview(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        // Form submission
        document.getElementById('warpForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(e.target);
            const loadingDiv = document.getElementById('loading');
            const errorDiv = document.getElementById('error');
            const resultsDiv = document.getElementById('results');
            const submitButton = e.target.querySelector('button');

            // Show loading, hide others
            loadingDiv.style.display = 'block';
            errorDiv.style.display = 'none';
            resultsDiv.style.display = 'none';
            submitButton.disabled = true;

            // Scroll to loading
            loadingDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    displayResults(result);
                } else {
                    throw new Error(result.error || 'Processing failed');
                }
            } catch (err) {
                errorDiv.textContent = '❌ ' + err.message;
                errorDiv.style.display = 'block';
                errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } finally {
                loadingDiv.style.display = 'none';
                submitButton.disabled = false;
            }
        });

        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';

            // Text Answer Section
            if (result.text_answers && result.text_answers.length > 0) {
                const textSection = document.createElement('div');
                textSection.className = 'result-section';
                textSection.innerHTML = `
                    <h3>🤖 LLaVA's Answer</h3>
                    <div class="text-answer">
                        <strong>Question:</strong>
                        <div style="margin-bottom: 1rem; color: var(--gray-700);">${escapeHtml(result.query)}</div>
                        <strong>Answer:</strong>
                        <div style="color: var(--gray-800); font-size: 1.05rem;">${escapeHtml(result.text_answers[0])}</div>
                    </div>
                `;
                resultsDiv.appendChild(textSection);
            }

            // Images Section
            const imagesSection = document.createElement('div');
            imagesSection.className = 'result-section';
            imagesSection.innerHTML = '<h3>🖼️ Generated Visualizations</h3>';

            const imageGrid = document.createElement('div');
            imageGrid.className = 'image-grid';

            const images = [
                { path: result.original_image, label: '📷 Original Image', emoji: '📷' },
                { path: result.masked_overlay, label: '🎯 Attention Overlay', emoji: '🎯' },
                { path: result.warped_image, label: '✨ Warped Result', emoji: '✨' },
                { path: result.visualization, label: '📊 Full Comparison', emoji: '📊' }
            ];

            images.forEach(img => {
                if (img.path) {
                    const container = document.createElement('div');
                    container.className = 'image-container';
                    container.innerHTML = `
                        <h4>${img.label}</h4>
                        <img src="/results/${img.path}" alt="${img.label}" loading="lazy">
                    `;
                    imageGrid.appendChild(container);
                }
            });

            imagesSection.appendChild(imageGrid);
            resultsDiv.appendChild(imagesSection);

            resultsDiv.style.display = 'block';

            // Scroll to results
            setTimeout(() => {
                resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""

# ============================================================================
# FLASK WEB SERVER
# ============================================================================

app = None
if FLASK_AVAILABLE:
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = '/content/demo_book_uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/process', methods=['POST'])
    def process_image():
        try:
            # Get form data
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400

            image_file = request.files['image']
            query = request.form.get('query', '')
            transform = request.form.get('transform', 'sqrt')
            exp_scale = float(request.form.get('exp_scale', '1.0'))
            exp_divisor = float(request.form.get('exp_divisor', '1.0'))
            attention_alpha = float(request.form.get('attention_alpha', '0.4'))
            apply_inverse = request.form.get('apply_inverse') == 'on'

            if not image_file.filename:
                return jsonify({'error': 'No image file selected'}), 400

            if not query.strip():
                return jsonify({'error': 'No query provided'}), 400

            # Save uploaded image temporarily
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

            # Create output directory
            base_output_dir = "/content/demo_book_results"
            os.makedirs(base_output_dir, exist_ok=True)

            run_id = 0
            while True:
                current_run_dir = os.path.join(base_output_dir, f"web_run_{run_id}")
                if not os.path.exists(current_run_dir):
                    os.makedirs(current_run_dir)
                    break
                run_id += 1

            # Process with LLaVA
            global images, queries, mota_mask, attention_maps, text_answers, masked_images

            # Set global variables for processing
            images = [image_path]
            queries = [query]

            # Run the workflow
            example_workflow()

            # Save outputs using save_warped_image
            original_image_save_path = os.path.join(current_run_dir, "original_image.png")
            masked_overlay_save_path = os.path.join(current_run_dir, "masked_overlay_image.png")
            warped_image_save_path = os.path.join(current_run_dir, "warped_image.png")
            visualization_save_path = os.path.join(current_run_dir, "visualization.png")

            # Handle mota_mask
            mota_mask_to_use = mota_mask
            if isinstance(mota_mask, list) and len(mota_mask) > 0:
                mota_mask_to_use = mota_mask[0]
            elif isinstance(mota_mask, list) and len(mota_mask) == 0:
                mota_mask_to_use = np.ones((500, 500), dtype=np.float32) * 128

            success = save_warped_image(
                image_path=image_path,
                att_map=mota_mask_to_use,
                original_image_save_path=original_image_save_path,
                masked_overlay_save_path=masked_overlay_save_path,
                output_path=warped_image_save_path,
                vis_path=visualization_save_path,
                width=500,
                height=500,
                transform=transform,
                exp_scale=exp_scale,
                exp_divisor=exp_divisor,
                apply_inverse=apply_inverse,
                attention_alpha=attention_alpha
            )

            if not success:
                return jsonify({'error': 'Image processing failed'}), 500

            # Save text answers
            text_answers_path = os.path.join(current_run_dir, "llava_text_answers.txt")
            with open(text_answers_path, "w") as f:
                f.write("LLAVA Text Answers\n")
                f.write("="*50 + "\n\n")
                for i, (question, answer) in enumerate(zip(queries, text_answers)):
                    f.write(f"Question {i+1}: {question}\n")
                    f.write(f"Answer {i+1}: {answer}\n")
                    f.write("-" * 30 + "\n\n")

            # Return result paths (relative to web server)
            result = {
                'original_image': f'web_run_{run_id}/original_image.png',
                'masked_overlay': f'web_run_{run_id}/masked_overlay_image.png',
                'warped_image': f'web_run_{run_id}/warped_image.png',
                'visualization': f'web_run_{run_id}/visualization.png',
                'query': query,
                'text_answers': text_answers,
                'transform': transform,
                'run_id': run_id
            }

            # Clean up uploaded file
            try:
                os.remove(image_path)
            except:
                pass

            return jsonify(result)

        except Exception as e:
            print(f"Error processing request: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/results/<path:filename>')
    def serve_result(filename):
        base_output_dir = "/content/demo_book_results"
        return send_from_directory(base_output_dir, filename)

    def run_server(host='0.0.0.0', port=5000, debug=False):
        """Run the Flask web server"""
        if IN_COLAB:
            print("🌐 Detected Colab environment - setting up Cloudflare tunnel...")

            # Try Cloudflare tunnel first (easiest!)
            try:
                print("Setting up Cloudflare tunnel...")
                output_queue, metrics_port = Queue(), randint(8100, 9000)
                thread = Timer(2, cloudflared_tunnel, args=(port, metrics_port, output_queue))
                thread.start()
                thread.join()
                tunnel_url = output_queue.get()
                print(f"🚀 SUCCESS! Access your app at: {tunnel_url}")
                print(f"Local development URL: http://localhost:{port}")
                print("Press Ctrl+C to stop")
            except Exception as e:
                print(f"Cloudflare tunnel failed: {e}")
                print("Falling back to Colab's built-in tunneling...")

                # Fallback to Colab's built-in tunneling
                if COLAB_MODULE_AVAILABLE:
                    try:
                        import google.colab.output as colab_output
                        print("Setting up Colab tunnel...")
                        colab_output.serve_kernel_port_unstable(port, path="/")
                        print("🚀 Colab tunnel is active!")
                        print("Look for the pop-up notification in Colab with your access link")
                        print(f"Local development URL: http://localhost:{port}")
                        print("Press Ctrl+C to stop")
                    except Exception as e2:
                        print(f"Failed to set up Colab tunnel: {e2}")
                        print(f"Server running locally at http://localhost:{port}")
                        print("Note: You may not be able to access this URL in Colab")
                else:
                    print("No tunneling options available")
                    print(f"Server running locally at http://localhost:{port}")
                    print("Note: You may not be able to access this URL in Colab")
        else:
            print(f"Starting web server at http://localhost:{port}")
            print("Press Ctrl+C to stop")

        app.run(host=host, port=port, debug=debug, use_reloader=False)

# Allow external override via environment variables
img_env = os.getenv("IMAGE_PATH")
q_env   = os.getenv("QUESTION_TEXT")

# Manual override for Colab detection
if os.getenv("FORCE_COLAB") == "true":
    IN_COLAB = True
    print("FORCED Colab mode via environment variable")

default_image = "/shared/nas2/dwip2/CLIP/images/image2.png"
default_query = "On the right desk, what is to the left of the laptop?"

images  = [img_env] if img_env else [default_image]
queries = [q_env] if q_env else [default_query]

# Global variables for LLaVA outputs
mota_mask = None
attention_maps = None
text_answers = None
masked_images = None

def example_workflow():
    global images, mota_mask, attention_maps, text_answers, masked_images  # Make results accessible to main
    # Using our custom enhanced llava_api directly (returns 4 values: masked_images, attention_maps, mota_masks, text_answers)
    import torch
    # import pdb
    # 'images' and 'queries' are already set above from environment variables or defaults
    masked_images, attention_maps, mota_mask, text_answers = custom_llava_api(images, queries, model_name="llava-v1.5-7b")

    # Print the text answers from LLaVA
    print("\n" + "="*50)
    print("LLAVA TEXT ANSWERS:")
    print("="*50)
    for i, (question, answer) in enumerate(zip(queries, text_answers)):
        print(f"Question {i+1}: {question}")
        print(f"Answer {i+1}: {answer}")
        print("-" * 30)
    print("All outputs will be saved to the run directory after processing...")


# Define attention map transformation functions
def identity_transform(x):
    """No transformation, returns input as is."""
    return x

def identity_inverse(x):
    """Inverse of identity is identity."""
    return x

def square_transform(x):
    """Square the attention values."""
    return x**2

def square_inverse(x):
    """Inverse of square is square root."""
    return np.sqrt(np.maximum(x, 0))

def sqrt_transform(x):
    """Square root of attention values."""
    return np.sqrt(np.maximum(x, 0))

def sqrt_inverse(x):
    """Inverse of square root is square."""
    return x**2

# Configurable parameters for exponential transform
EXP_SCALE = 1.0  # Multiplier for input: exp(EXP_SCALE * x)
EXP_DIVISOR = 1.0  # Divisor for output: exp(x) / EXP_DIVISOR

# Apply inverse flag - enables "apply transform, take marginal, apply inverse" workflow
APPLY_INVERSE_TO_MARGINALS = False

def exp_transform(x):
    """Exponential of attention values with configurable scaling."""
    return np.exp(EXP_SCALE * x) / EXP_DIVISOR

def exp_inverse(x):
    """Inverse of exponential."""
    return np.log(np.maximum(x * EXP_DIVISOR, 1e-9)) / EXP_SCALE

def log_transform(x):
    """Log of attention values (with small epsilon to avoid log(0))."""
    return np.log(x + 1e-5)

def log_inverse(x):
    """Inverse of log is exp."""
    return np.exp(x) - 1e-5

# Mapping of transforms to their inverses
INVERSE_TRANSFORMS = {
    identity_transform: identity_inverse,
    square_transform: square_inverse,
    sqrt_transform: sqrt_inverse,
    exp_transform: exp_inverse,
    log_transform: log_inverse
}

# Choose the transformation function to apply
ATTENTION_TRANSFORM = sqrt_transform  # Default transform

# Constants
EPSILON = 1e-9
BASE_ATTENTION = 1e-9 # Adjust visualization sensitivity

# --- Core Warping Logic ---
def warp_image_by_attention(image, att_map, new_width, new_height):
    """
    Warps an image based on attention map.
    Assumes image and att_map have the same HxW dimensions.
    """

    # Debug breakpoint removed for non-interactive runs

    h, w = image.shape[:2]
    att_map_float = att_map.astype(np.float64)
    att_map_float = np.maximum(att_map_float, 0)
    #pdb.set_trace()
    # Apply the selected transformation to attention map
    att_map_transformed = ATTENTION_TRANSFORM(att_map_float)

    att_map_biased = att_map_transformed + BASE_ATTENTION

    # Calculate Marginal Attention Profiles
    att_profile_x = np.sum(att_map_biased, axis=0) # Shape: (w,)
    att_profile_y = np.sum(att_map_biased, axis=1) # Shape: (h,)

    # Apply inverse function to marginal profiles if enabled
    if APPLY_INVERSE_TO_MARGINALS and ATTENTION_TRANSFORM in INVERSE_TRANSFORMS:
        inverse_func = INVERSE_TRANSFORMS[ATTENTION_TRANSFORM]
        # Remove BASE_ATTENTION before applying inverse
        att_profile_x = inverse_func(att_profile_x - BASE_ATTENTION * h)
        att_profile_y = inverse_func(att_profile_y - BASE_ATTENTION * w)
        # Add back bias after inverse
        att_profile_x = att_profile_x + BASE_ATTENTION * h
        att_profile_y = att_profile_y + BASE_ATTENTION * w

    total_att_x = np.sum(att_profile_x)
    total_att_y = np.sum(att_profile_y)

    if total_att_x < EPSILON or total_att_y < EPSILON:
        print("Warning: Total attention is near zero.", file=sys.stderr)
        att_profile_x = np.ones(w, dtype=np.float64)
        att_profile_y = np.ones(h, dtype=np.float64)
        total_att_x = w * (np.mean(att_map_biased) * h) # Approximate total
        total_att_y = h * (np.mean(att_map_biased) * w) # Approximate total
        # Avoid division by zero later
        total_att_x = max(total_att_x, EPSILON)
        total_att_y = max(total_att_y, EPSILON)

    # Calculate Cumulative Profiles -> Forward Mapping
    cum_att_x = np.cumsum(att_profile_x)
    norm_cum_att_x = cum_att_x / total_att_x
    x_orig_coords = np.arange(w)
    x_new_map_fwd = np.concatenate(([0], norm_cum_att_x)) * new_width
    x_orig_map_fwd = np.concatenate(([0], x_orig_coords + 1))

    cum_att_y = np.cumsum(att_profile_y)
    norm_cum_att_y = cum_att_y / total_att_y
    y_orig_coords = np.arange(h)
    y_new_map_fwd = np.concatenate(([0], norm_cum_att_y)) * new_height
    y_orig_map_fwd = np.concatenate(([0], y_orig_coords + 1))

    x_new_map_fwd[-1] = new_width
    y_new_map_fwd[-1] = new_height

    # Inverse Mapping for cv2.remap
    x_target_coords = np.arange(new_width)
    y_target_coords = np.arange(new_height)
    map_x_orig = np.interp(x_target_coords, x_new_map_fwd, x_orig_map_fwd)
    map_y_orig = np.interp(y_target_coords, y_new_map_fwd, y_orig_map_fwd)

    final_map_x, final_map_y = np.meshgrid(map_x_orig, map_y_orig)
    final_map_x = final_map_x.astype(np.float32)
    final_map_y = final_map_y.astype(np.float32)

    # Apply Warp
    warped_image = cv2.remap(
        image, final_map_x, final_map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    # Ensure target shape (handle potential small deviations from remap)
    if warped_image.shape[0] != new_height or warped_image.shape[1] != new_width:
         final_channels = image.shape[2] if image.ndim == 3 else 1
         warped_image = cv2.resize(warped_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
         # Handle potential channel drop during resize
         if warped_image.ndim == 2 and final_channels == 3:
             warped_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
         elif warped_image.ndim == 3 and final_channels == 1 and image.ndim == 2:
             warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    return warped_image

def generate_visualization(image, att_map, warped_image, output_path, transform_name, attention_alpha=0.5):
    """Generate visualization with original image, attention map, and warped result"""
    if image is None or att_map is None or warped_image is None:
        print("Cannot generate visualization: missing data")
        return

    # Normalize attention map for visualization
    att_map_norm = att_map.copy()
    min_val, max_val = np.min(att_map), np.max(att_map)
    if max_val > min_val + EPSILON:
        att_map_norm = (att_map_norm - min_val) / (max_val - min_val)
    else:
        att_map_norm = np.zeros_like(att_map)

    # Convert to heatmap visualization for overlay
    att_map_color = cv2.applyColorMap((att_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Create the overlay image
    overlay = image.copy()
    cv2.addWeighted(att_map_color, attention_alpha, image, 1 - attention_alpha, 0, overlay)

    # Create a combined visualization
    h, w = image.shape[:2]
    h_warped, w_warped = warped_image.shape[:2]

    # Make all images the same height for visualization
    target_height = max(h, h_warped)

    # Resize if needed
    if h != target_height:
        scale = target_height / h
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, target_height))
        overlay = cv2.resize(overlay, (new_w, target_height))
        w = new_w

    if h_warped != target_height:
        scale = target_height / h_warped
        new_w_warped = int(w_warped * scale)
        warped_image = cv2.resize(warped_image, (new_w_warped, target_height))
        w_warped = new_w_warped

    # Create visualization with original + attention + warped
    visualization = np.zeros((target_height, w + w + w_warped, 3), dtype=np.uint8)
    visualization[:, :w] = image
    visualization[:, w:2*w] = overlay
    visualization[:, 2*w:] = warped_image

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualization, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, "Attention Map", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(visualization, f"Warped ({transform_name})", (2*w + 10, 30), font, 1, (255, 255, 255), 2)

    # Add separator lines
    cv2.line(visualization, (w, 0), (w, target_height), (255, 255, 255), 2)
    cv2.line(visualization, (2*w, 0), (2*w, target_height), (255, 255, 255), 2)

    # Add grid to warped image
    grid_spacing = 20
    for x in range(2*w, 2*w + w_warped, grid_spacing):
        cv2.line(visualization, (x, 0), (x, target_height), (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, target_height, grid_spacing):
        cv2.line(visualization, (2*w, y), (2*w + w_warped, y), (255, 255, 255), 1, cv2.LINE_AA)

    # Save visualization
    cv2.imwrite(output_path, visualization)

    print(f"Visualization saved to {output_path}")

def resize_image_to_match_attmap(image, att_map):
    """Resizes image to match attention map dimensions if necessary."""
    if image is None or att_map is None:
        return None

    target_h, target_w = att_map.shape[:2]
    current_h, current_w = image.shape[:2]

    if (current_h, current_w) == (target_h, target_w):
        print("Image and attention map dimensions match.")
        return image.copy()
    else:
        print(f"Resizing image from {image.shape[:2]} to match attention map {att_map.shape[:2]}")
        try:
            resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if resized_image.shape[:2] != (target_h, target_w):
                raise RuntimeError("Resize resulted in unexpected shape.")
            print("Image resized successfully.")
            return resized_image
        except Exception as e:
            print(f"Error resizing image: {e}")
            return None

def set_transform_function(transform_name, exp_scale=1.0, exp_divisor=1.0, apply_inverse=False):
    """Sets the global transformation function based on name."""
    global ATTENTION_TRANSFORM, EXP_SCALE, EXP_DIVISOR, APPLY_INVERSE_TO_MARGINALS

    # Set exponential parameters if provided
    EXP_SCALE = exp_scale
    EXP_DIVISOR = exp_divisor
    APPLY_INVERSE_TO_MARGINALS = apply_inverse

    # Set the transformation function
    if transform_name == "identity":
        ATTENTION_TRANSFORM = identity_transform
    elif transform_name == "square":
        ATTENTION_TRANSFORM = square_transform
    elif transform_name == "sqrt":
        ATTENTION_TRANSFORM = sqrt_transform
    elif transform_name == "exp":
        ATTENTION_TRANSFORM = exp_transform
    elif transform_name == "log":
        ATTENTION_TRANSFORM = log_transform
    else:
        print(f"Unknown transform: {transform_name}. Using identity transform.")
        ATTENTION_TRANSFORM = identity_transform
        return "identity"

    return transform_name

def save_warped_image(image_path, att_map,
                      original_image_save_path,  # New path for resized original
                      masked_overlay_save_path, # New path for masked overlay
                      output_path, # Path for warped image
                      vis_path=None,
                      width=500, height=500, transform="identity", # width/height now act as defaults/guide for viz
                      exp_scale=1.0, exp_divisor=1.0, apply_inverse=False, attention_alpha=0.5):
    """Process and save warped image, original, and masked overlay, all in original input image dimensions."""
    try:
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        else:
            # If image_path is already a PIL image
            image = np.array(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        input_original_h, input_original_w = image.shape[:2]

        # Save a copy of the original image (no resize needed here for this specific output)
        if original_image_save_path:
            cv2.imwrite(original_image_save_path, image.copy())
            print(f"Original image saved to {original_image_save_path} with shape {image.shape[:2]}")

        # Handle attention map
        if isinstance(att_map, np.ndarray):
            pass
        elif isinstance(att_map, Image.Image):
            att_map = np.array(att_map)
        elif isinstance(att_map, list):
            if len(att_map) > 0:
                first_element = att_map[0]
                if isinstance(first_element, np.ndarray):
                    att_map = first_element
                elif isinstance(first_element, Image.Image):
                    att_map = np.array(first_element)
                else:
                    att_map = np.array(first_element)
            else:
                # Use provided width/height for default att_map if list is empty
                att_map = np.ones((height, width), dtype=np.float32) * 128

        if att_map.ndim == 3:
            att_map = np.mean(att_map, axis=2)
        elif att_map.ndim != 2:
            raise ValueError(f"Attention map must be 2D, got shape {att_map.shape}")

        # Create and save masked overlay image (using original input dimensions)
        if masked_overlay_save_path:
            # Base image for overlay is the original image itself
            overlay_base_image = image.copy()
            if overlay_base_image.ndim == 2: # if grayscale
                overlay_base_image = cv2.cvtColor(overlay_base_image, cv2.COLOR_GRAY2BGR)

            # Resize attention map to original input dimensions for overlay
            att_map_resized_for_overlay = cv2.resize(att_map.copy(), (input_original_w, input_original_h), interpolation=cv2.INTER_LINEAR)

            att_map_norm_overlay = att_map_resized_for_overlay.copy()
            min_val_ov, max_val_ov = np.min(att_map_norm_overlay), np.max(att_map_norm_overlay)
            if max_val_ov > min_val_ov + EPSILON:
                att_map_norm_overlay = (att_map_norm_overlay - min_val_ov) / (max_val_ov - min_val_ov)
            else:
                att_map_norm_overlay = np.zeros_like(att_map_norm_overlay)

            att_map_color_overlay = cv2.applyColorMap((att_map_norm_overlay * 255).astype(np.uint8), cv2.COLORMAP_JET)

            masked_overlay_image = cv2.addWeighted(att_map_color_overlay, attention_alpha, overlay_base_image, 1 - attention_alpha, 0)
            cv2.imwrite(masked_overlay_save_path, masked_overlay_image)
            print(f"Masked overlay image saved to {masked_overlay_save_path} with shape {masked_overlay_image.shape[:2]}")

        # Image for warping: resize original image to match attention map dimensions (as expected by warp_image_by_attention)
        image_for_warping = resize_image_to_match_attmap(image, att_map)
        if image_for_warping is None:
            raise ValueError("Failed to resize image to match attention map dimensions for warping")

        # Set transform function
        transform_name = set_transform_function(transform, exp_scale, exp_divisor, apply_inverse)

        # Process image (Warping) - output will be of input_original_w, input_original_h
        warped_image = warp_image_by_attention(image_for_warping, att_map, input_original_w, input_original_h)
        if warped_image is None:
            raise ValueError("Warping failed")

        # Save warped image (now in original input dimensions)
        cv2.imwrite(output_path, warped_image)
        print(f"Warped image saved to {output_path} with shape {warped_image.shape[:2]}")

        # Generate and save visualization strip if requested
        # Inputs to generate_visualization are:
        # image_for_warping (original image resized to att_map dims),
        # att_map (original resolution att_map),
        # warped_image (warped to original input image dims)
        if vis_path:
            generate_visualization(image_for_warping, att_map, warped_image, vis_path, transform_name, attention_alpha)

        return True

    except Exception as e:
        print(f"Error during processing: {e}")
        return False

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Attention-Based Non-Uniform Image Warping")

    # Required arguments
    parser.add_argument("--image", required=False, default="/shared/nas2/dwip2/CLIP/images/image2.png", help="Path to input image file")
    parser.add_argument("--attention-map", required=False, default="/shared/nas2/dwip2/CLIP/mota_mask.npy", help="Path to attention map .npy file")
    # Output base filename for warped image, directory will be auto-generated
    parser.add_argument("--output", required=False, default="mota_warped.png", help="Base filename for warped output image")

    # Optional arguments
    # Visualization base filename, directory will be auto-generated
    parser.add_argument("--visualization", default=None, help="Base filename for visualization with input, attention map, and output")
    parser.add_argument("--width", type=int, default=500, help="Target width for warped image")
    parser.add_argument("--height", type=int, default=500, help="Target height for warped image")
    parser.add_argument("--transform", choices=["identity", "square", "sqrt", "exp", "log"],
                        default="identity", help="Attention transformation function")
    parser.add_argument("--exp-scale", type=float, default=1.0, help="Scale for exponential transform")
    parser.add_argument("--exp-divisor", type=float, default=1.0, help="Divisor for exponential transform")
    parser.add_argument("--apply-inverse", action="store_true",
                        help="Apply inverse transform to marginal profiles")
    parser.add_argument("--attention-alpha", type=float, default=0.4,
                        help="Alpha blending value for attention map overlay (0.0-1.0)")

    args = parser.parse_args()

    # --- Create unique output directory for this run ---
    base_output_dir = "/shared/nas2/dwip2/CLIP/output_runs"
    os.makedirs(base_output_dir, exist_ok=True)

    run_id = 0
    while True:
        current_run_dir = os.path.join(base_output_dir, f"run_{run_id}")
        if not os.path.exists(current_run_dir):
            os.makedirs(current_run_dir)
            break
        run_id += 1
    print(f"Saving outputs to: {current_run_dir}")

    # Define full paths for all outputs within the run-specific directory
    original_image_save_path = os.path.join(current_run_dir, "original_image.png") # Renamed
    masked_overlay_save_path = os.path.join(current_run_dir, "masked_overlay_image.png") # Renamed
    warped_image_save_path = os.path.join(current_run_dir, os.path.basename(args.output))

    visualization_save_path = None
    if args.visualization:
        visualization_save_path = os.path.join(current_run_dir, os.path.basename(args.visualization))

    # Handle mota_mask being a list for the main function
    mota_mask_to_use = mota_mask
    if isinstance(mota_mask, list):
        if mota_mask and len(mota_mask) > 0:
            print(f"Main: mota_mask is a list, using first element of {len(mota_mask)} items")
            mota_mask_to_use = mota_mask[0]
        else:
            print("Main: mota_mask is an empty list, creating default mask")
            mota_mask_to_use = np.ones((args.height, args.width), dtype=np.float32) * 128

    # Process image
    success = save_warped_image(
        image_path=images[0],
        att_map=mota_mask_to_use,  # Use the properly handled mask
        original_image_save_path=original_image_save_path,
        masked_overlay_save_path=masked_overlay_save_path,
        output_path=warped_image_save_path,
        vis_path=visualization_save_path,
        width=args.width,
        height=args.height,
        transform=args.transform,
        exp_scale=args.exp_scale,
        exp_divisor=args.exp_divisor,
        apply_inverse=args.apply_inverse,
        attention_alpha=args.attention_alpha
    )

    # Save LLaVA outputs to the run directory
    if success:
        print(f"Saving LLaVA outputs to {current_run_dir}...")

        # Save text answers
        text_answers_path = os.path.join(current_run_dir, "llava_text_answers.txt")
        with open(text_answers_path, "w") as f:
            f.write("LLAVA Text Answers\n")
            f.write("="*50 + "\n\n")
            for i, (question, answer) in enumerate(zip(queries, text_answers)):
                f.write(f"Question {i+1}: {question}\n")
                f.write(f"Answer {i+1}: {answer}\n")
                f.write("-" * 30 + "\n\n")
        print(f"Text answers saved to: {text_answers_path}")

        # Save attention maps as numpy arrays
        for i, att_map in enumerate(attention_maps):
            if isinstance(att_map, torch.Tensor):
                att_map = att_map.detach().cpu().numpy()
            att_map_path = os.path.join(current_run_dir, f"attention_map_{i}.npy")
            np.save(att_map_path, att_map)
            print(f"Attention map {i} saved to: {att_map_path}")

        # Save mota_masks
        if isinstance(mota_mask, list):
            print(f"mota_mask is a list with {len(mota_mask)} elements")
            for i, mask in enumerate(mota_mask):
                if isinstance(mask, Image.Image):
                    # Convert to grayscale if not already
                    if mask.mode != 'L':
                        mask_gray = mask.convert('L')
                    else:
                        mask_gray = mask
                    # Save as grayscale image
                    mask_img_path = os.path.join(current_run_dir, f"mota_mask_{i}_gray.png")
                    mask_gray.save(mask_img_path)
                    # Convert to numpy array and save
                    mask_np = np.array(mask_gray)
                    mask_npy_path = os.path.join(current_run_dir, f"mota_mask_{i}.npy")
                    np.save(mask_npy_path, mask_np)
                    print(f"Mota mask {i} saved as image: {mask_img_path} and numpy: {mask_npy_path}")
        else:
            # Handle single mota_mask
            if isinstance(mota_mask, Image.Image):
                # Convert to grayscale if not already
                if mota_mask.mode != 'L':
                    mota_mask_gray = mota_mask.convert('L')
                else:
                    mota_mask_gray = mota_mask
                # Save as grayscale image
                mota_mask_img_path = os.path.join(current_run_dir, "mota_mask_gray.png")
                mota_mask_gray.save(mota_mask_img_path)
                # Convert to numpy array and save
                mota_mask_np = np.array(mota_mask_gray)
                mota_mask_npy_path = os.path.join(current_run_dir, "mota_mask.npy")
                np.save(mota_mask_npy_path, mota_mask_np)
                print(f"Mota mask saved as image: {mota_mask_img_path} and numpy: {mota_mask_npy_path}")

        # Save masked images
        if masked_images and len(masked_images) > 0:
            for i, masked_img in enumerate(masked_images):
                if isinstance(masked_img, Image.Image):
                    # Convert PIL Image to numpy array
                    masked_img_np = np.array(masked_img)
                    # Convert RGB to BGR for OpenCV
                    if len(masked_img_np.shape) == 3 and masked_img_np.shape[2] == 3:
                        masked_img_np = cv2.cvtColor(masked_img_np, cv2.COLOR_RGB2BGR)
                    # Save using cv2
                    masked_img_path = os.path.join(current_run_dir, f"masked_image_{i}.png")
                    cv2.imwrite(masked_img_path, masked_img_np)
                    print(f"Masked image {i} saved to: {masked_img_path}")
        else:
            print("No masked images to save")

    return 0 if success else 1

if __name__ == '__main__':
    # Check if running with no arguments - run server mode
    if len(sys.argv) == 1:
        if not FLASK_AVAILABLE:
            print("Flask is required for web server mode. Install with: pip install flask")
            sys.exit(1)
        print("Running in web server mode...")
        print(f"IN_COLAB: {IN_COLAB}, CLOUDFLARE_AVAILABLE: {CLOUDFLARE_AVAILABLE}")

        if IN_COLAB:
            print("🚀 Colab detected - will use Cloudflare tunnel (super easy!)")
            print("Just run: !python run.py")
        else:
            print("Open your browser to http://localhost:5000")
        run_server()
    else:
        # Run command-line mode
        example_workflow()
        sys.exit(main())


import time
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
import gradio as gr

from sod_model import SODNet



# GLOBALS 

MODEL       = None
DEVICE      = "cpu"
IMAGE_SIZE  = 224
THRESHOLD   = 0.5

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)



# INFERENCE

def predict(pil_image: Image.Image, threshold: float = 0.5):
    # predict image
    if MODEL is None:
        raise RuntimeError("Model not loaded!")

    original_size = pil_image.size   

    # Pre-process
    img = pil_image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225])
    t = t.unsqueeze(0).to(DEVICE)

    # model prediction
    start = time.perf_counter()
    with torch.no_grad():
        pred = MODEL(t)
    elapsed_ms = (time.perf_counter() - start) * 1000

    #  output mask
    pred_np = pred.squeeze().cpu().numpy()              
    mask_bin = (pred_np > threshold).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_bin, mode="L").resize(
        original_size, Image.NEAREST)

    # Overlay: red channel on original
    orig_np  = np.array(pil_image.convert("RGB"), dtype=np.float32)
    mask_rs  = np.array(mask_pil, dtype=np.float32) / 255.0
    overlay  = orig_np.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] * 0.6 + mask_rs * 180, 0, 255)
    overlay_pil = Image.fromarray(overlay.astype(np.uint8))

    return mask_pil, overlay_pil, f"{elapsed_ms:.1f} ms"



# GRADIO UI

def gradio_predict(image, threshold):
    if image is None:
        return None, None, "No image provided."
    pil = Image.fromarray(image)
    mask, overlay, t = predict(pil, threshold=threshold)
    return np.array(mask), np.array(overlay), f"⏱ Inference time: {t}"


def build_app():
    with gr.Blocks(title="Salient Object Detection Demo") as demo:
        gr.Markdown("## 🔍 Salient Object Detection\nUpload an image to see the saliency mask.")

        with gr.Row():
            inp_img     = gr.Image(label="Input Image", type="numpy")
            out_mask    = gr.Image(label="Saliency Mask (grayscale)")
            out_overlay = gr.Image(label="Overlay")

        with gr.Row():
            thresh_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                      label="Threshold")
            time_box = gr.Textbox(label="Inference Time", interactive=False)

        run_btn = gr.Button("Run Inference", variant="primary")
        run_btn.click(fn=gradio_predict,
                      inputs=[inp_img, thresh_slider],
                      outputs=[out_mask, out_overlay, time_box])

        gr.Markdown(
            "### How to use\n"
            "1. Upload any image on the left.\n"
            "2. Click **Run Inference**.\n"
            "3. Adjust the threshold slider if needed."
        )
    return demo


# MAIN

def main():
    global MODEL, DEVICE, IMAGE_SIZE

    parser = argparse.ArgumentParser(description="SOD Gradio demo")
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--image_size",   type=int, default=224)
    parser.add_argument("--port",         type=int, default=7860)
    args = parser.parse_args()

    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_SIZE = args.image_size

    MODEL = SODNet(base_filters=args.base_filters).to(DEVICE)
    MODEL.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    MODEL.eval()
    print(f"Model loaded: {args.model_path}  (device={DEVICE})")

    demo = build_app()
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()

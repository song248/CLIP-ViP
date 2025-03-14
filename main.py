import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from configs.config import shared_configs
from modeling.VidCLIP import VidCLIP
from utils.load_save import load_state_dict_with_mismatch
from utils.misc import set_random_seed

# Model & Data
MODEL_PATH = "assets/pretrain_clipvip_base_32.pt"
VIDEO_PATH = "video/falldown.mp4"
OUTPUT_PATH = "output/features.npy"

# pretraining_config
cfg = shared_configs.get_pretraining_args()

if "patch32" in cfg.clip_config:
    cfg.max_img_size = 224  # ViT-B/32
elif "patch16" in cfg.clip_config:
    cfg.max_img_size = 384  # ViT-B/16
else:
    cfg.max_img_size = 448  # default
print(f"🔹 Using max_img_size: {cfg.max_img_size}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weight files do not exist: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(cfg.seed)

# CLIP-ViP Model Load
def setup_model(cfg, device=None):
    print("🔹 Setting up CLIP-ViP model...")
    model = VidCLIP(cfg)    

    if MODEL_PATH:
        print(f"🔹 Loading weights from {MODEL_PATH}...")
        load_state_dict_with_mismatch(model, MODEL_PATH)
    
    if hasattr(cfg, "freeze_text_model") and cfg.freeze_text_model:
        freeze_text_proj = hasattr(cfg, "freeze_text_proj") and cfg.freeze_text_proj
        print(f"🔹 Freezing CLIP text model. Projection freeze: {freeze_text_proj}")
        model.freeze_text_encoder(freeze_text_proj)

    model.to(device)
    print(" - Model setup complete!")
    return model

# Video frame sampling
def extract_frames(video_path, num_frames=cfg.num_frm):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = frame_count // (num_frames - 1)
    selected_frames = np.arange(0, frame_count, frame_step)
    # selected_frames = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def preprocess_frames(frames, image_size=cfg.max_img_size):
    frames = [cv2.resize(x, (image_size, image_size)) for x in frames]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])
    return torch.stack([transform(frame) for frame in frames])

# Extract video vector
def extract_video_features(model, frames):
    with torch.no_grad():
        frames = frames.unsqueeze(0).to(device)  # (1, N, C, H, W)
        features = model.forward_video(frames)
    return features.cpu().numpy()

if __name__ == "__main__":
    model = setup_model(cfg, device)

    print(f"Extracting frames from {VIDEO_PATH}...")
    frames = extract_frames(VIDEO_PATH)

    print("Preprocessing frames...")
    frames_tensor = preprocess_frames(frames)

    print("Extracting video features using CLIP-ViP...")
    video_features = extract_video_features(model, frames_tensor)

    print(f"Saving features to {OUTPUT_PATH}...")
    np.save(OUTPUT_PATH, video_features)

    print(" * Feature extraction completed!")

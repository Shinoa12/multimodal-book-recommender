from PIL import Image
import open_clip
import torch

def load_openclip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        cache_dir="models/openclip"
    )
    model.eval()
    return model, preprocess

def generate_query_visual_embedding(image_path, model, preprocess):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features
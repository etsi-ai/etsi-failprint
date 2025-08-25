import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
embedding_extractor = torch.nn.Sequential(*list(model.children())[:-1])
embedding_extractor.to(device) # Move model to GPU
embedding_extractor.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def generate_image_embeddings(image_paths: list) -> torch.Tensor:
    """Generates embeddings for a list of images using a pre-trained ResNet50."""
    batch_tensors = []
    for path in image_paths:
        try:
            with Image.open(path).convert("RGB") as img:
                img_t = preprocess(img)
                batch_tensors.append(img_t)
        except Exception as e:
            print(f"Warning: Could not load image {path} for embedding. Skipping. Error: {e}")
            continue
    
    if not batch_tensors:
        return torch.empty((0, 2048))

 
    batch = torch.stack(batch_tensors).to(device)

    embeddings = embedding_extractor(batch)
    
    return embeddings.squeeze().cpu()
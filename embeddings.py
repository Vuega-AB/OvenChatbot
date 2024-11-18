import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

# Load models
text_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Load EfficientNet model
image_model = EfficientNet.from_pretrained('efficientnet-b0')
image_model.eval()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
image_model.to(device)

# Image preprocessing for EfficientNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fully connected layer to reduce image embedding to 384 dimensions
class ImageEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1280, 384)

    def forward(self, x):
        return self.fc(x)

# Initialize the image embedding network
image_embedding_net = ImageEmbeddingNet().to(device)

def generate_text_embedding(text):
    text_embedding= text_model.encode(text, normalize_embeddings=True)
    split_embeddings = np.array_split(text_embedding, 384)
    pooled_embedding = np.array([np.mean(chunk) for chunk in split_embeddings])
    return pooled_embedding

def generate_image_embedding(image_path):
    try:
        with Image.open(image_path) as image:
            # Convert image to grayscale (black and white)
            image = image.convert("L")
            
            # Convert grayscale image back to 3 channels for model compatibility
            image = image.convert("RGB")
            
            # Preprocess the image
            image = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Get feature vector from EfficientNet
                image_embedding = image_model.extract_features(image)
                image_embedding = image_embedding.mean([2, 3])  # Global average pooling
                image_embedding = image_embedding_net(image_embedding).cpu().numpy().flatten()
                image_embedding = image_embedding / np.linalg.norm(image_embedding)  # Normalize
        
        return image_embedding
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        return np.zeros(384)  # Return zero vector in case of error


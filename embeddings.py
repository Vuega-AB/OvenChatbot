import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sentence_transformers import SentenceTransformer
import fitz
import os
import uuid

# Load models
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load ResNet-50 model
image_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
image_model.eval()

# Remove the fully connected layer to get the raw feature map (2048-dimensional)
image_model = nn.Sequential(*list(image_model.children())[:-1])  # Remove the last FC layer

# Image preprocessing
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
        self.fc = nn.Linear(2048, 384)  # Reduce to 384 dimensions (same as text)

    def forward(self, x):
        return self.fc(x)

# Initialize the image embedding network
image_embedding_net = ImageEmbeddingNet()

def generate_text_embedding(text):
    return text_model.encode(text)

def generate_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    image = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        image_embedding = image_model(image).squeeze().numpy()  # Shape (2048,)
        image_embedding = torch.tensor(image_embedding, dtype=torch.float32)
        image_embedding = image_embedding_net(image_embedding).numpy()
    
    return image_embedding


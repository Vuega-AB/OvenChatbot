from flask import Flask, request, jsonify, render_template
from pdf2image import convert_from_path
import fitz  # PyMuPDF for text extraction
import pytesseract
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch
import os
import uuid
from retrieval import retrieve_component
import clip

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    # Get description and uploaded image from the form
    description = request.form["description"]
    image_file = request.files["image"]
    
    # Save the image to a temporary path
    image_path = "./data/uploaded_image.jpg"
    image_file.save(image_path)

    # Retrieve the best matching component(s)
    result = retrieve_component(description, image_path)
    for component in result:
        if isinstance(component.get("embedding"), np.ndarray):  # Check if embedding is a NumPy array
            component["embedding"] = component["embedding"].tolist()

    # Return the result as a JSON response
    return jsonify({"response": result})


# Directories to store extracted images
extracted_image_dir = "./data/extracted_images/"
# os.makedirs(extracted_image_dir, exist_ok=True)

# Initialize models
text_model = SentenceTransformer('all-MiniLM-L6-v2')
vit_model = models.vit_b_16(pretrained=True)
vit_model.eval()
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    pdf_file = request.files["pdf"]
    pdf_path = f"./data/{uuid.uuid4()}.pdf"
    pdf_file.save(pdf_path)

    # Step 1: Extract text and images from the PDF
    text_data = extract_text_from_pdf(pdf_path)
    image_paths = extract_images_from_pdf(pdf_path)

    # Step 2: Generate embeddings and match text descriptions to images
    text_embedding = generate_text_embedding(text_data)
    matches = match_text_to_image([text_data], image_paths)  # Matches text with images

    # Step 3: Prepare results in a structured format and convert data types
    result = {
        "extracted_text": text_data,
        "image_matches": [
            {
                "image_path": img, 
                "match_score": float(score)  # Convert float32 to float
            } 
            for img, score in matches
        ]
    }

    return jsonify({"status": "success", "result": result})


# Helper Functions
def extract_images_from_pdf(pdf_path):
    """Converts PDF pages to images and saves each image."""
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(extracted_image_dir, f"{uuid.uuid4()}.jpg")
        img.save(img_path, "JPEG")
        image_paths.append(img_path)
    return image_paths

def extract_text_from_pdf(pdf_path):
    """Extracts text from the entire PDF using PyMuPDF."""
    pdf_document = fitz.open(pdf_path)
    text_data = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        text_data.append(text)
    pdf_document.close()
    return "\n".join(text_data)

clip_model, preprocess = clip.load("ViT-B/32", device="cpu")  # Adjust device as needed

# Modify image transformation to use CLIP's preprocessing
def generate_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0)  # Preprocess with CLIP's preprocessing
    with torch.no_grad():
        embedding = clip_model.encode_image(image).cpu().numpy().flatten()
    return embedding

def generate_text_embedding(text):
    # Truncate the input text to fit the CLIP model's context length limit (77 tokens)
    context_length = 77  # Adjust if different for your model
    truncated_text = text[:context_length]

    text_tokens = clip.tokenize([truncated_text])
    with torch.no_grad():
        embedding = clip_model.encode_text(text_tokens).cpu().numpy().flatten()
    return embedding



def match_text_to_image(texts, images):
    """Matches text descriptions with images using embeddings and returns matches."""
    text_embeddings = [generate_text_embedding(text) for text in texts]
    image_embeddings = [generate_image_embedding(img) for img in images]

    matches = []
    for text_embed in text_embeddings:
        similarities = [np.dot(text_embed, img_embed) for img_embed in image_embeddings]
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]
        matches.append((images[best_match_index], best_match_score))
    return matches

if __name__ == "__main__":
    app.run(debug=True)

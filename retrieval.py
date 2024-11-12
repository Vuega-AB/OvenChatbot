import faiss
import numpy as np
import pandas as pd
import os
from embeddings import generate_text_embedding, generate_image_embedding

# Load CSV file
data_path = './data/components.csv'
df = pd.read_csv(data_path)

# Set embedding dimension (384 based on our custom network)
dimension = 384
index = faiss.IndexFlatL2(dimension)

components = []

# Process components from CSV
for idx, row in df.iterrows():
    description = row['Description & Specification']
    image_filename = row['Image']
    image_path = os.path.join('./data/images', image_filename)
    
    # Generate embeddings for description and image
    text_embedding = generate_text_embedding(description)
    image_embedding = generate_image_embedding(image_path)
    
    # Average text and image embeddings (if description is provided)
    combined_embedding = (text_embedding + image_embedding) / 2 if description else image_embedding
    
    components.append({
        'id': row['No.'],
        'name': row['Material Code'],
        'description': description,
        'image': image_filename,
        'embedding': combined_embedding  # Store combined embedding
    })
    
    # Add embedding to FAISS index
    index.add(np.array([combined_embedding], dtype=np.float32))

def retrieve_component(description, image_path):
    # Generate embedding from input image
    image_embedding = generate_image_embedding(image_path)
    
    # If description is provided, combine embeddings
    if description:
        text_embedding = generate_text_embedding(description)
        combined_input_embedding = (text_embedding + image_embedding) / 2
    else:
        combined_input_embedding = image_embedding

    combined_input_embedding = combined_input_embedding.reshape(1, -1).astype('float32')

    # Perform FAISS search
    _, indices = index.search(combined_input_embedding, k=3)
    
    # Get matching components
    matching_components = [components[i] for i in indices[0]]
    return [{"id": comp['id'], "name": comp['name']} for comp in matching_components]

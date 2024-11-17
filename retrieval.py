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
    
    try:
        # Generate text embedding
        text_embedding = generate_text_embedding(description) if description else np.zeros(dimension)
    except Exception as e:
        print(f"Error generating text embedding for row {idx}: {e}")
        text_embedding = np.zeros(dimension)
    
    try:
        # Generate image embedding
        if os.path.exists(image_path):
            image_embedding = generate_image_embedding(image_path)
        else:
            print(f"Image file not found: {image_path}")
            image_embedding = np.zeros(dimension)
    except Exception as e:
        print(f"Error generating image embedding for {image_path}: {e}")
        image_embedding = np.zeros(dimension)
    
    # Combine embeddings (average if both are present)
    combined_embedding = (text_embedding + image_embedding) / 2 if np.linalg.norm(text_embedding) > 0 else image_embedding

    # Normalize the combined embedding
    combined_embedding /= np.linalg.norm(combined_embedding) if np.linalg.norm(combined_embedding) > 0 else 1.0
    
    components.append({
        'id': row['No.'],
        'name': row['Material Code'],
        'description': description,
        'image': image_filename,
        'embedding': combined_embedding  # Store combined embedding
    })
    
    # Add embedding to FAISS index
    if np.linalg.norm(combined_embedding) > 0:  # Check for valid embedding
        index.add(np.array([combined_embedding], dtype=np.float32))
    else:
        print(f"Invalid embedding for component {row['No.']} - Skipped.")

def retrieve_component(description, image_path, k=3):
    try:
        # Generate embedding from input image
        image_embedding = generate_image_embedding(image_path)
    except Exception as e:
        print(f"Error generating input image embedding: {e}")
        image_embedding = np.zeros(dimension)
    
    # Combine input embeddings if description is provided
    if description:
        try:
            text_embedding = generate_text_embedding(description)
        except Exception as e:
            print(f"Error generating input text embedding: {e}")
            text_embedding = np.zeros(dimension)
        combined_input_embedding = (text_embedding + image_embedding) / 2
    else:
        combined_input_embedding = image_embedding

    # Normalize the combined input embedding
    combined_input_embedding /= np.linalg.norm(combined_input_embedding) if np.linalg.norm(combined_input_embedding) > 0 else 1.0
    combined_input_embedding = combined_input_embedding.reshape(1, -1).astype('float32')

    # Perform FAISS search
    _, indices = index.search(combined_input_embedding, k=k)
    
    # Retrieve matching components
    matching_components = [
        {
            "id": components[i]['id'],
            "name": components[i]['name'],
            "image": components[i]['image'],
            "score": _
        }
        for i in indices[0]
    ]
    return matching_components

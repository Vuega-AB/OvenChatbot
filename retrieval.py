import faiss
import numpy as np
import pandas as pd
import os
from embeddings import generate_text_embedding, generate_image_embedding

# Load CSV file
data_path = './data/components.csv'
df = pd.read_csv(data_path)

# Initialize FAISS index
dimension = 384  # Assuming 512-dimensional embeddings (you can adjust this)
index = faiss.IndexFlatL2(dimension)

components = []

# Process components from CSV
for idx, row in df.iterrows():
    # Extract relevant fields
    description = row['Description & Specification']
    image_filename = row['Image']
    image_path = os.path.join('./data/images', image_filename)
    
    # Generate embeddings for the description and image
    text_embedding = generate_text_embedding(description)
    image_embedding = generate_image_embedding(image_path)
    
    # Combine text and image embeddings (you can tweak this part for more sophisticated approaches)
    combined_embedding = (text_embedding + image_embedding) / 2  # Average of both embeddings (example approach)
    
    # Store component data
    components.append({
        'id': row['No.'],
        'name': row['Material Code'],
        'description': description,
        'image': image_filename,
        'embedding': combined_embedding
    })

    print("Text embedding shape:", text_embedding.shape)
    print("Image embedding shape:", image_embedding.shape)
    print("Combined embedding shape:", combined_embedding.shape)

        
    # Add combined embedding to the FAISS index
    index.add(np.array([combined_embedding], dtype=np.float32))

def retrieve_component(description, image_path):
    # Generate embeddings for the input description and image
    text_embedding = generate_text_embedding(description).reshape(1, -1)
    image_embedding = generate_image_embedding(image_path).reshape(1, -1)
    
    # Combine the embeddings (average them or use another approach)
    combined_input_embedding = (text_embedding + image_embedding) / 2
    
    # Perform FAISS search
    _, indices = index.search(combined_input_embedding, k=3)  # k=3: retrieve top 3 matching components
    
    # Get matching components
    matching_components = [components[i] for i in indices[0]]
    
    return matching_components


import streamlit as st
from pdf2image import convert_from_bytes
import faiss
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from dotenv import load_dotenv
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
import numpy as np

load_dotenv()

# Configure Gemini API
api_key = os.getenv("api_key")
genai.configure(api_key=api_key)

def pdf_to_images(pdf_bytes):
    print("Converting PDF to images...")
    return convert_from_bytes(pdf_bytes.read(), dpi=200)

def summarize_page(image):
    print("Summarizing page...")
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "You are processing a user manual page. Extract key information that will help "
        "answer user questions about the device. Focus on functionality, instructions, warnings, "
        "and any important technical details. Avoid generalizations and prioritize useful content."
        "The summary would be used as a way to understand the content of the page and choses the pages relative to user query."
        "Instructions:\n"
        "- just respond with the summary directly.\n"
    )
    response = model.generate_content([prompt, image])
    return response.text

def process_manual(pdf_file):
    print("Processing uploaded manual...")
    images = pdf_to_images(pdf_file)
    
    with ThreadPoolExecutor() as executor:
        summaries = list(executor.map(summarize_page, images))  # Parallel processing

    page_data = [{"image": img, "summary": sum_text} for img, sum_text in zip(images, summaries)]
    print(f"Finished processing {len(images)} pages.")
    return page_data

# --- Fast Page Retrieval with FAISS ---
def index_summaries(manual):
    summaries = [page["summary"] for page in manual]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(summaries).toarray()
    
    index = faiss.IndexFlatL2(vectors.shape[1])  # L2 (Euclidean distance)
    index.add(np.array(vectors, dtype=np.float32))
    
    return index, vectorizer

def find_relevant_pages(query, index, vectorizer, manual):
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    _, indices = index.search(query_vector, 4)  # Get top 5 pages
    return indices.flatten()

def ask_gemini(query, relevant_pages, manual, uploaded_image=None):
    """Sends a structured query to Gemini using relevant manual pages and user-uploaded image."""
    print(f"Querying Gemini with question: {query}")
    print(f"Relevant pages: {relevant_pages}")
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # context = "\n\n".join([manual[i]["summary"] for i in relevant_pages])
    images = [manual[i]["image"] for i in relevant_pages]
    
    prompt = (
    "You are an AI assistant helping users understand their device by using the provided user manual. "
    "Your task is to answer the user's question in a helpful and clear manner, based on the relevant pages of the manual. "
    "You should not just reference the pages, but also include the relevant information from those pages in your response. "
    "Incorporate key details such as functionality, setup instructions, troubleshooting steps, safety warnings, "
    "and any important technical aspects mentioned in the manual. If any images or diagrams are provided, "
    "describe them or use them to enrich the response where appropriate."
    "\n\n"
    f"User's question: {query}\n"
    "Here are the most relevant sections of the manual that will help answer your question:\n"
    "--------------------------------------------------\n"
    "Manual Sections:\n"
)

    
    parts = [prompt]
    
    # Attach relevant manual images
    for idx, image in enumerate(images):
        print(f"Attaching manual image {idx + 1}")
        parts.append(image)
    
    # Attach user-uploaded image (if any)
    if uploaded_image:
        print("User uploaded an additional image for reference.")
        parts.append("The user also uploaded this image, which may help in answering the question:")
        parts.append(uploaded_image)

    print("try send to Gemini.")
    try:
        print(parts)
        response = model.generate_content(parts)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request. Please try again later."

# Streamlit UI
st.title("Device Manual Chatbot")

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

pdf_file = st.file_uploader("Upload a user manual (PDF)", type=["pdf"], key=f"file_uploader_{st.session_state.file_uploader_key}")
if pdf_file:
    st.info("Processing manual... This may take a few minutes.")
    page_data = process_manual(pdf_file)
    st.session_state["manual"] = page_data
    st.session_state["faiss_index"], st.session_state["vectorizer"] = index_summaries(st.session_state["manual"])
    st.success("Manual processed successfully!")

    # Reset the file uploader by incrementing the key
    st.session_state.file_uploader_key += 1
    st.rerun()  # Force a rerun to update the UI immediately

query = st.text_input("Ask a question about your device:")
uploaded_image = st.file_uploader("Upload an image related to your question (optional)", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Open and resize the image to a reasonable size for preview
    user_image = Image.open(uploaded_image)
    user_image.thumbnail((300, 300))  # Resize the image to 300x300 while maintaining aspect ratio
    st.image(user_image, caption="Uploaded Image Preview")

# Create columns to align button to the right
col1, col2 = st.columns([4, 1])

submit_clicked = col2.button("Send", use_container_width=True)

if submit_clicked and query and "manual" in st.session_state:
    relevant_pages = find_relevant_pages(query, st.session_state["faiss_index"], st.session_state["vectorizer"], st.session_state["manual"])
    print(relevant_pages)
    # Convert uploaded image to PIL format if provided
    user_image = Image.open(uploaded_image) if uploaded_image else None
    
    with st.spinner("Generating response..."):
        answer = ask_gemini(query, relevant_pages, st.session_state["manual"], user_image)
    
    st.subheader("Answer:")
    st.write(answer)
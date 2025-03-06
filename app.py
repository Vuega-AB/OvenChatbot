import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
api_key=os.getenv("api_key")
genai.configure(api_key=api_key)

def pdf_to_images(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(pdf_bytes.read())
        images = convert_from_bytes(open(tmpfile.name, "rb").read(), dpi=300)
    return images

def summarize_page(image):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = "Summarize the content of this manual page."
    response = model.generate_content([prompt, image])
    return response.text

def process_manual(pdf_file):
    images = pdf_to_images(pdf_file)
    page_data = []
    summaries = []
    for image in images:
        summary = summarize_page(image)
        page_data.append({"image": image, "summary": summary})
        summaries.append(summary)
    return page_data, summaries

def find_relevant_pages(query, summaries):
    vectorizer = TfidfVectorizer().fit_transform([query] + summaries)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    return similarities.argsort()[-3:][::-1]  # Get top 3 relevant pages

def ask_gemini(query, relevant_pages, manual):
    model = genai.GenerativeModel("gemini-2.0-flash")
    context = "\n\n".join([manual[i]["summary"] for i in relevant_pages])
    images = [manual[i]["image"] for i in relevant_pages]
    prompt = f"Using the following images and context, answer this question:\n{query}\n\n{context}"

    # Display the images before sending to Gemini
    st.subheader("Images Sent to Gemini:")
    for idx, image in enumerate(images):
        st.image(image, caption=f"Relevant Page Image {idx+1}")

    parts = [prompt]
    for image in images:
        parts.append(image)

    response = model.generate_content(parts)
    return response.text

# Streamlit UI
st.title("Device Manual Chatbot")

pdf_file = st.file_uploader("Upload a user manual (PDF)", type=["pdf"])
if pdf_file:
    st.info("Processing manual... This may take a few minutes.")
    page_data, summaries = process_manual(pdf_file)
    st.session_state["manual"] = page_data
    st.session_state["summaries"] = summaries
    st.success("Manual processed successfully!")


    # Display images with summaries
    st.subheader("Manual Pages Preview")
    for idx, data in enumerate(page_data):
        st.image(data["image"], caption=f"Page {idx+1}")
        st.write(f"**Summary:** {data['summary']}")

query = st.text_input("Ask a question about your device:")
if query and "manual" in st.session_state:
    relevant_pages = find_relevant_pages(query, st.session_state["summaries"])
    answer = ask_gemini(query, relevant_pages, st.session_state["manual"])
    st.write(answer)

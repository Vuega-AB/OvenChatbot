import streamlit as st
from retrieval import retrieve_component
from PIL import Image, ImageOps
import os

# Set up the Streamlit app
st.title("Component Retrieval System")
st.write("Upload an image and optionally enter a description to find matching components.")

# File upload widget
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Text input for the description
description = st.text_input("Enter a description (optional)")

# Button to perform the search
if st.button("Retrieve Components"):
    if uploaded_image is not None:
        # Save the uploaded image temporarily
        image_path = "./data/uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Open the uploaded image
        original_image = Image.open(image_path)

        # Create a black-and-white version of the image
        bw_image = ImageOps.grayscale(original_image)
        bw_image = bw_image.convert("RGB")  # Convert back to 3 channels for consistency

        # Resize images for display
        display_size = (150, 150)  # Smaller dimensions
        original_image_resized = original_image.resize(display_size)
        bw_image_resized = bw_image.resize(display_size)

        # Display the original and black-and-white images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image_resized, caption="Original Image", use_container_width=True)
        with col2:
            st.image(bw_image_resized, caption="Black-and-White Image", use_container_width=True)

        # Perform retrieval
        results = retrieve_component(description, image_path)

        # Display the results
        if results:
            st.success("Matching components found!")
            for result in results:
                st.write(f"**Component ID:** {result['id']}")
                st.write(f"**Component Name:** {result['name']}")

                # Get the image filename from the result (Use .get() to avoid key errors)
                image_filename = result.get('image')

                if image_filename:
                    # Build the path to the image using os.path.join for cross-platform compatibility
                    image_file_path = os.path.join("data", "images", image_filename)
                    image_file_path = os.path.normpath(image_file_path)  # Normalize path to handle slashes

                    # Debugging: print out the image path for verification
                    st.write(f"Checking if image exists at: {image_file_path}")

                    # Check if the image exists
                    if os.path.exists(image_file_path):
                        try:
                            # Try to open the image
                            component_image = Image.open(image_file_path)

                            # Resize component image
                            component_image_resized = component_image.resize(display_size)

                            # Display the resized component image
                            st.image(component_image_resized, caption=f"Component {result['id']}", use_container_width=True)
                        except Exception as e:
                            # In case there's an issue opening the image
                            st.warning(f"Error loading image for Component ID {result['id']}: {e}")
                    else:
                        # Warn the user if the image doesn't exist
                        st.warning(f"Image file not found for Component ID {result['id']}. Expected file: {image_filename}")
                else:
                    # Warn if no image is associated with the component
                    st.warning(f"No image file associated with Component ID {result['id']}.")
        else:
            st.warning("No matching components found.")
    else:
        st.error("Please upload an image to proceed.")

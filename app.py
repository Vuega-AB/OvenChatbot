from flask import Flask, request, jsonify, render_template
from retrieval import retrieve_component

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

    # Return the result as a JSON response
    return jsonify({"response": result})


if __name__ == "__main__":
    app.run(debug=True)

import os
from models.nav import nav_call
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS  # Optional: use if you need CORS headers

# Create the Flask app; set static_folder to 'assets' and static_url_path to '/assets'
app = Flask(__name__, static_folder='assets', static_url_path='/assets')
CORS(app)  # This adds CORS headers to all responses (optional)

@app.route('/')
def homepage():
    # Serve homepage.html from the same directory as flask_app.py
    return send_from_directory(os.path.dirname(__file__), 'homepage.html')

# If you need to serve other HTML files in the root directory, you can add similar routes.

# Optional: if you want a route for other static pages
@app.route('/<path:filename>')
def other_files(filename):
    """
    If you have additional HTML files in the same directory,
    you can serve them by requesting /filename.
    """
    return send_from_directory(os.path.dirname(__file__), filename)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get JSON data from the request
    # For demonstration, let's assume our "prediction" is the sum of the coordinates.
    coordinates = data.get("coordinates", [])
    try:
        prediction = sum(float(num) for num in coordinates)
    except ValueError:
        return jsonify(error="Invalid input"), 400

    return jsonify(result=prediction)

# Simulating the model function that takes four elements as input and returns coordinates
def process_model(element1, element2, element3, element4):
    # Replace this with your actual model logic
    output=nav_call(element1, element2, element3, element4)
    return output

@app.route('/process', methods=['POST'])
def process():
    data = request.json  # Get JSON data from frontend
    print("Received data from frontend:", data)  # Debug print
    elements = data.get("data")  # Extract elements array

    if not elements or len(elements) != 4:
        error_message = f"Invalid input, expected 4 elements, got: {elements}"
        print(error_message)
        return jsonify({"error": error_message}), 400

    element1, element2, element3, element4 = elements  # Unpack elements
    try:
        print("Calling nav_call with:", element1, element2, element3, element4)
        output = nav_call(float(element1), float(element2), float(element3), float(element4))
        print("nav_call returned:", output)
    except Exception as e:
        print("Error in nav_call:", e)
        output = {"error": str(e)}
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)


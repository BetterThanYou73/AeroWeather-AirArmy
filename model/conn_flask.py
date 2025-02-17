from flask import Flask, request, jsonify
from nav import nav_call
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Simulating the model function that takes four elements as input and returns coordinates
def process_model(element1, element2, element3, element4):
    # Replace this with your actual model logic
    output=nav_call(element1, element2, element3, element4)
    return output

@app.route('/process', methods=['POST'])
def process():
    data = request.json  # Get JSON data from frontend
    elements = data.get("elements")  # Extract elements array
    
    if not elements or len(elements) != 4:
        return jsonify({"error": "Invalid input, expected 4 elements"}), 400

    element1, element2, element3, element4 = elements  # Unpack elements
    result = process_model(element1, element2, element3, element4)  # Process model

    return jsonify(result)  # Return JSON response

if __name__ == '__main__':
    app.run(debug=True)

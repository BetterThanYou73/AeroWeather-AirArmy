from flask import Flask, request, jsonify
import numpy as np
from nav import nav_call
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json["coordinates"] 
    input_data = np.array(data).reshape(1, -1) # Example: Input array from frontend
    if len(input_data) != 4:
        return jsonify({"error": "Invalid input format."}), 400
    latA=input_data[0]
    longA=input_data[1]
    latB=input_data[2]
    longB=input_data[3]
   
    output_data = process_model(latA,longA,latB,longB)  # Process input data
    return output_data

def process_model(latA,longA,latB,longB):
    output_data=nav_call(latA,longA,latB,longB)
    return output_data

if __name__ == '__main__':
    
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify
# from nav import nav_call  # Uncomment if you have a nav_call function

app = Flask(__name__, static_folder="assets")

@app.route('/predict', methods=['POST'])
def process_data():
    # Safely extract the "coordinates" array from the JSON payload
    data = request.json.get("coordinates", [])
    if len(data) != 4:
        return jsonify({"error": "Invalid input format. Expected [latA, lngA, latB, lngB]"}), 400
    
    # Unpack the array
    latA, longA, latB, longB = data  
    print("Received coordinates:")
    print("FROM -> Latitude:", latA, "Longitude:", longA)
    print("TO   -> Latitude:", latB, "Longitude:", longB)
    
    # Process the input data here if needed:
    # output_data = nav_call(latA, longA, latB, longB)
    # return output_data
    
    # For now, simply return a confirmation response
    return jsonify({
        "message": "Data received successfully",
        "coordinates": data
    })

if __name__ == '__main__':
    app.run()  # You can add debug=True and port=5000 if desired

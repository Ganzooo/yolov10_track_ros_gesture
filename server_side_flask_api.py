from flask import Flask, request, jsonify
from pprint import pprint

app = Flask(__name__)

# Route to handle POST request with bbox data
@app.route('/api/bbox', methods=['POST'])
def receive_bbox():
    try:
        # Parse the JSON data from the request body
        bbox_data = request.get_json()

        # Perform any processing (e.g., save to database or perform calculations)
        # For now, we'll just log it and send it back as a response
        print("Received BBox")
        pprint(bbox_data)
        
        data_len = len(bbox_data)
        # Send success response back to client
        return jsonify({"status": "success", "message": "Bounding box received", "data_len": data_len}), 200

    except Exception as e:
        # Handle any exceptions and return an error message
        return jsonify({"status": "error", "message": str(e)}), 400

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, jsonify
import pandas as pd
import logging


app = Flask(__name__)


# Endpoint to serve EV data
@app.route('/get_ev_data', methods=['GET'])
def get_ev_data():
    try:
        df = pd.read_csv('Electric_Vehicle_Population_Data.csv')  # Load your dataset
        return df.to_json(orient='records')  # Return as JSON
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
   

from flask import Flask, request, jsonify, abort, make_response

import mlflow
import mlflow.pyfunc
import os
import pandas as pd

BASE_PATH = os.path.expandvars("$PROJECT_HOME")

model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

app = Flask(__name__)

@app.route("/info", methods = ["GET"])
def info():

    response = make_response(str(model.metadata), 200)
    response.content_type = "text/plain"
    return response

@app.route("/", methods = ["GET"])
def home():
    msg = """
    Welcome to our ML service to predict Secondary market car price\n\n

    This API has two main endpoints:\n
    1. /info: to get info about the deployed model.\n
    2. /predict: to send predict requests to our deployed model.\n

    """

    response = make_response(msg, 200)
    response.content_type = "text/plain"
    return response

# /predict endpoint
@app.route("/predict", methods = ["POST"])
def predict():
	if not request.json:
		abort(400)

	inputs = request.json
	inputs = inputs["inputs"]
	metadata = model.metadata.signature
	inputs_format = metadata.inputs.to_dict()
	inputs = pd.DataFrame.from_dict(inputs, orient="index").T
	for col in inputs_format:
		col_name = col['name']
		if col_name not in inputs.columns:
			abort(400, f"Column {col_name} is missing")
		col_type = col['type']
		try:
			if col_type == "long":
				inputs[col_name] = inputs[col_name].astype(int)
			elif col_type == "double":
				inputs[col_name] = inputs[col_name].astype(float)
		except:
			abort(400, f"Can't parse {col_name} column. Expected type: {col_type}.")

	predictions = model.predict(inputs).flatten()
	response = make_response(jsonify(predictions.tolist()), 200)
	response.content_type = "application/json"
	return response

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
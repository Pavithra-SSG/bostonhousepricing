import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

# Initialize the scaler
scalar = StandardScaler()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    global scalar  # Ensure we're using the global scalar

    # Get data from request
    data = request.json['data']

    # Try to transform the new data using the fitted scaler
    try:
        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    except AttributeError:
        # If scaler is not fitted, fit it with the current data
        scalar.fit(np.array(list(data.values())).reshape(1, -1))
        # Transform the new data using the fitted scaler
        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))

    # Make predictions with the loaded model
    output = regmodel.predict(new_data)

    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    global scalar  # Ensure we're using the global scalar

    data=[float(x) for x in request.form.values()]

    # Try to transform the data using the fitted scaler
    try:
        final_input = scalar.transform(np.array(data).reshape(1,-1))
    except AttributeError:
        # If scaler is not fitted, fit it with the current data
        scalar.fit(np.array(data).reshape(1,-1))
        # Transform the data using the fitted scaler
        final_input = scalar.transform(np.array(data).reshape(1,-1))

    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The house price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)

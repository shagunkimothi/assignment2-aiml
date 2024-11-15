from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler (ensure the paths are correct)
model = joblib.load('best_rf_model (1).pkl')  # Update the path if necessary
scaler = joblib.load('scaler.pkl')  # Load the scaler (if it's saved)

# Define the HTML form inside the Python code with a light blue background
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Form</title>
    <style>
        body {
            background-color: lightblue;
            font-family: Arial, sans-serif;
        }
        h2 {
            color: #333;
        }
        form {
            margin: 0 auto;
            width: 300px;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        label {
            font-weight: bold;
        }
        input[type="number"], input[type="submit"] {
            width: 100%;
            padding: 8px;
            margin-top: 8px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h2>Enter Feature Values for Prediction</h2>
    <form method="POST" action="/predict">
        <label for="combined_figures">Combined Figures (kg/capita/year):</label>
        <input type="number" step="any" name="combined_figures" required><br><br>
        <label for="household_estimate_kg">Household Estimate (kg/capita/year):</label>
        <input type="number" step="any" name="household_estimate_kg" required><br><br>
        <label for="household_estimate_tonnes">Household Estimate (tonnes/year):</label>
        <input type="number" step="any" name="household_estimate_tonnes" required><br><br>
        <label for="retail_estimate_kg">Retail Estimate (kg/capita/year):</label>
        <input type="number" step="any" name="retail_estimate_kg" required><br><br>
        <label for="retail_estimate_tonnes">Retail Estimate (tonnes/year):</label>
        <input type="number" step="any" name="retail_estimate_tonnes" required><br><br>
        <label for="food_service_estimate_kg">Food Service Estimate (kg/capita/year):</label>
        <input type="number" step="any" name="food_service_estimate_kg" required><br><br>
        <label for="food_service_estimate_tonnes">Food Service Estimate (tonnes/year):</label>
        <input type="number" step="any" name="food_service_estimate_tonnes" required><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction_text %}
        <h3>{{ prediction_text }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        combined_figures = float(request.form['combined_figures'])
        household_estimate_kg = float(request.form['household_estimate_kg'])
        household_estimate_tonnes = float(request.form['household_estimate_tonnes'])
        retail_estimate_kg = float(request.form['retail_estimate_kg'])
        retail_estimate_tonnes = float(request.form['retail_estimate_tonnes'])
        food_service_estimate_kg = float(request.form['food_service_estimate_kg'])
        food_service_estimate_tonnes = float(request.form['food_service_estimate_tonnes'])

        # Combine the features into an array for the model
        features = np.array([combined_figures, household_estimate_kg, household_estimate_tonnes, retail_estimate_kg, retail_estimate_tonnes, food_service_estimate_kg, food_service_estimate_tonnes]).reshape(1, -1)

        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features)

        # Make the prediction
        prediction = model.predict(features_scaled)[0]

        # Translate prediction to readable result
        result = "High" if prediction == 1 else "Low"
        return render_template_string(form_html, prediction_text=f"Predicted Food Waste Category: {result}")
    except Exception as e:
        # Print the detailed error message for debugging
        print(f"Error: {str(e)}")
        return render_template_string(form_html, prediction_text="Error occurred. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True)

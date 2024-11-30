from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained sarcasm detection model
model = joblib.load('sarcasm_detection_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the user
    user_input = request.form['user_input']
    
    # Predict sarcasm
    prediction = model.predict([user_input])[0]
    
    # Return result as JSON response
    result = "The input sentence is sarcastic." if prediction == 1 else "The input sentence is not sarcastic."
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=False)

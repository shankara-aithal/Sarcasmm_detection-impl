from joblib import load  # Used to load the model

def load_and_predict(model_path, sentence):
    """
    Load a trained model from a file and predict if the given sentence is sarcastic.
    """
    # Load the trained model
    model = load(model_path)
    # Make a prediction on the given sentence
    prediction = model.predict([sentence])
    return prediction[0]

# Example usage
if __name__ == "__main__":
    model_path = 'sarcasm_model.pkl'  # Path to the saved model file
    input_sentence = input("Enter a sentence for sarcasm prediction: ")
    result = load_and_predict(model_path, input_sentence)

    if result == 1:
        print("The input sentence is sarcastic.")
    else:
        print("The input sentence is not sarcastic.")

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# Function to load data from a JSON file
def load_data(file_path):
    """
    Load the JSON file and return the parsed data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to train the model
def train_models(df):
    """
    Train a sarcasm detection model using a pipeline with text vectorization and LinearSVC.
    """
    # Extract features and target from the DataFrame
    contexts = df['context'].apply(lambda x: ' '.join(x))  # Combine list of context sentences into a single string
    responses = df['utterance']
    X = contexts + ' ' + responses  # Combine context and response
    y = df['sarcasm'].astype(int)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a pipeline for text classification
    pipeline = Pipeline([
        ('vect', CountVectorizer()),       # Convert text to token counts
        ('tfidf', TfidfTransformer()),    # Apply TF-IDF transformation
        ('clf', LinearSVC())              # Linear Support Vector Classifier
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model F1 Score: {f1:.2f}")

    # Return the trained pipeline
    return pipeline

# Function to predict sarcasm
def predict_sarcasm(sentence, model):
    """
    Predict if a given sentence is sarcastic or not.
    """
    prediction = model.predict([sentence])
    return prediction[0]

# Main execution
if __name__ == "__main__":
    # Specify the file path to your dataset
    file_path = r'C:\Users\ASUS\Downloads\Sarcasm_Detection-main\Sarcasm_Detection-main\csvjson.json'

    # Load JSON data
    data = load_data(file_path)
    df = pd.DataFrame(data)

    # Train the sarcasm detection model
    trained_model = train_models(df)

    # Example prediction
    input_sentence = "Since it's not bee season, you can have my epinephrine."
    is_sarcastic = predict_sarcasm(input_sentence, trained_model)

    if is_sarcastic:
        print("The input sentence is sarcastic.")
    else:
        print("The input sentence is not sarcastic.")

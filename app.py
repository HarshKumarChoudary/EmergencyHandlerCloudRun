from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

app = Flask(__name__)

# Load pre-trained models and other resources
lda_dementia = joblib.load('lda_dementia_model.pkl')
lda_emergency = joblib.load('lda_emergency_model.pkl')
feature_columns = ['prp_count', 'VP_count', 'NP_count', 'prp_noun_ratio', 'word_sentence_ratio',
                   'count_pauses', 'count_unintelligible', 'count_repetitions', 'ttr', 'R', 'ARI', 'CLI']

def extract_features_from_text(text):
    # Dummy feature extraction function
    # Replace with actual implementation
    features = np.random.rand(12)
    return features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Text input is required'}), 400

    features = extract_features_from_text(text)
    features = features.reshape(1, -1)  # Reshape for prediction

    dementia_prediction = lda_dementia.predict(features)[0]
    emergency_prediction = lda_emergency.predict(features)[0]

    return jsonify({
        'dementia_prediction': dementia_prediction,
        'emergency_prediction': emergency_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)

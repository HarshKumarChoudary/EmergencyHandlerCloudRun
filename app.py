# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import joblib

# app = Flask(__name__)

# # Load pre-trained models and other resources
# lda_dementia = joblib.load('lda_dementia_model.pkl')
# lda_emergency = joblib.load('lda_emergency_model.pkl')
# feature_columns = ['prp_count', 'VP_count', 'NP_count', 'prp_noun_ratio', 'word_sentence_ratio',
#                    'count_pauses', 'count_unintelligible', 'count_repetitions', 'ttr', 'R', 'ARI', 'CLI']

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     text = data.get('text')

#     if not text:
#         return jsonify({'error': 'Text input is required'}), 400

#     features = extract_features_from_text(text)
#     features = features.reshape(1, -1)  # Reshape for prediction

#     dementia_prediction = lda_dementia.predict(features)[0]
#     emergency_prediction = lda_emergency.predict(features)[0]

#     return jsonify({
#         'dementia_prediction': dementia_prediction,
#         'emergency_prediction': emergency_prediction
#     })

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify

app = Flask(__name__)

def is_emergency(text):
    """
    Check if the text suggests an emergency situation based on simple keyword matching.
    """
    emergency_keywords = [
        'help', 'emergency', 'urgent', 'danger', 'immediate assistance', 'call 911', 
        'medical emergency', 'crisis', 'accident', 'fire', 'injury', 'breakdown', 'threat'
    ]
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Check for emergency keywords
    for keyword in emergency_keywords:
        if keyword in text_lower:
            return True
    
    return False

@app.route('/emergency-check', methods=['POST'])
def emergency_check():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Text input is required'}), 400

    is_emergency_case = is_emergency(text)

    return jsonify({
        'is_emergency': is_emergency_case
    })

if __name__ == '__main__':
    app.run(debug=True)

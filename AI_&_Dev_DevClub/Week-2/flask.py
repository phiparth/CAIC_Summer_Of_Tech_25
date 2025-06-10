from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

def log_cosh_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true)))

app = Flask(__name__)

model = load_model('Model.keras', custom_objects={'log_cosh_loss': log_cosh_loss})

@app.route('/')
def home():
    return "Flask app is running. Use /predict with POST."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        base_keys = [
            'hour','word_count','char_count','polarity','subjectivity',
            'org_mentions','place_mentions','date_mentions','time_mentions',
            'loc_mentions','money_mentions','percent_mentions',
            'product_mentions','event_mentions','law_mentions','language_mentions',
            'art_mentions','syllable_count','sentence_count',
            'bert_mean','bert_max',
            'day_Friday','day_Monday','day_Saturday','day_Sunday',
            'day_Thursday','day_Tuesday','day_Wednesday',
            'company_encoded','sin_time','cos_time',
            'sentiment','flesch_score','tfidf_mean',
            'has_hashtag','company_avg_likes',
            'has_photo','has_video','has_gif','day',
            'weekday','month','year','is_weekend',
            'is_morning','is_evening'
        ]
        base_features = [data[k] for k in base_keys]
        bert_features = [data[f'bert_{i}'] for i in range(768)]

        x = np.array(base_features + bert_features).reshape(1, -1)
        log_pred = model.predict(x, verbose=0)[0][0]
        likes = int(np.expm1(log_pred))

        return jsonify({'predicted_likes': likes})

    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()
from flask import Flask, request, jsonify
from bonus1 import AITweetGenerator
app = Flask(__name__)
ai_generator = AITweetGenerator()

@app.route('/')
def home():
    return "Acha!"
ai_generator = AITweetGenerator()

@app.route('/generate_ai', methods=['POST'])
def generate_ai():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({'error': 'Prompt is required.'}), 400

        tweet = ai_generator.generate_ai_tweet(prompt)
        return jsonify({'generated_tweet': tweet})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__":
    app.run()
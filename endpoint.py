from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    fen = data.get('fen')
    depth = data.get('depth', 15)
    if not fen:
        return jsonify({"error": "FEN string is required"}), 400

    chess_api_url = "https://chess-api.com/v1"
    payload = {
        "fen": fen,
        "depth": depth
    }
    try:
        response = requests.post(chess_api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        extracted = {
            # 'continuation': result.get('continuationArr'),
            'eval': result.get('eval'),
            # 'text': result.get('text'),
        }
        return jsonify(extracted)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, jsonify, request
from classifier import alphabet_detection

app = Flask(__name__)

@app.route('/alphabet_pred', methods=['POST'])
def predictDigit():
    image = request.files.get('alphabet')
    prediction = alphabet_detection(image)

    return jsonify({
        "prediction" : prediction
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
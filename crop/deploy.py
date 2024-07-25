from flask import Flask, render_template, request, jsonify
import pickle


app = Flask(__name__)

# Load the model
model = pickle.load(open('savedmodel.sav', 'rb'))

@app.route('/')
def home():
    return render_template('cro2.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    N_SOIL = data['N_SOIL']
    P_SOIL = data['P_SOIL']
    K_SOIL = data['K_SOIL']
    TEMPERATURE = data['TEMPERATURE']
    HUMIDITY = data['HUMIDITY']
    ph = data['ph']
    RAINFALL = data['RAINFALL']
    CROP_PRICE = data['CROP_PRICE']
    result = model.predict([[N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL, CROP_PRICE]])[0]
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

    
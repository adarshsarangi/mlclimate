from distutils.log import debug
from flask import Flask , render_template, request
import json
import pandas as pd 
from prophet.serialize import model_from_json
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('predict2.html')

def load_model():
    with open('model.json','r') as fin : 
        m = model_from_json(fin.read())
    return m

@app.route('/weather_predict', methods=['GET','POST'])
def weather_predict():
    freq = request.form.get('frequency')[0].lower()
    periods = request.form.get('period')
    m = load_model()
    future_dataframe = m.make_future_dataframe(freq=freq, periods=int(periods))
    predictions = m.predict(future_dataframe)
    predictions = pd.DataFrame(predictions,columns = ['ds','yhat'])
    predictions['ds'] = predictions['ds'].dt.strftime('%Y-%m-%d')
    predictions['yhat'] = round(predictions['yhat'], 3)
    predictions = predictions[1460:1460+int(periods)]
    json_data = predictions.to_json(orient='records')
    json_data = json.loads(json_data)
    print(json_data)
    return render_template('prediction.html',json_data=json_data)

@app.route('/feed')
def feed():
    return render_template('feed.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000, debug=True)

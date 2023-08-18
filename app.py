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

@app.route('/predict', methods=['GET','POST'])
def predict():
    return render_template('predict2.html')

@app.route('/weather_predict', methods=['GET','POST'])
def weather_predict():
    freq = request.form.get('frequency')[0].lower()
    periods = int(request.form.get('period'))
    what = request.form.get('what')
    m = load_precipitation()
    if what == 'temperature' : 
        pass
    elif what == 'wind' :
        m = load_wind()
    elif what == 'weather' :
        pass
    future_dataframe = m.make_future_dataframe(freq=freq, periods=periods)
    predictions = m.predict(future_dataframe)
    df = pd.DataFrame(predictions,columns = ['ds','yhat'])
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d')
    df['yhat'] = round(df['yhat'], 3)
    df = df[1460:1460+int(periods)]
    json_data = df.to_json(orient='records')
    json_data = json.loads(json_data)
    return render_template('prediction.html',json_data=json_data)

def load_precipitation():
    with open('model.json','r') as fin : 
        m = model_from_json(fin.read())
    return m

def load_temp_max():
    with open('model_temp_max.json','r') as fin :
        m = model_from_json(fin.read())
    return m

def load_temp_min() : 
    with open('model_temp_min.json', 'r') as fin : 
        m = model_from_json(fin.read())
    return m

def load_temp(freq, periods) : 
    max_model = load_temp_max()
    min_model = load_temp_min()
    

def load_wind(freq, periods):
    with open('model_temp_mid.json', 'r') as fin : 
        m = model_from_json(fin.read())
    
    return m

@app.route('/feed')
def feed():
    return render_template('feed.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000, debug=True)

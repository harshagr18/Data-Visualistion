import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

vector = pickle.load(open("Models/vector.pkl","rb"))
RFC = pickle.load(open("Models/RFC.pickel","rb"))
DT = pickle.load(open("Models/DT.pickel","rb"))
GBC = pickle.load(open("Models/GBC.pickel","rb"))
LR = pickle.load(open("Models/LR.pickel","rb"))
KN = pickle.load(open("Models/KN.pickel","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global count
    '''
    For rendering results on HTML GUI
    '''
    news = [str(x) for x in request.form.values()]
    testing_news = {"text":[news[0]]}
    new_def_test = pd.DataFrame(testing_news)
    new_x_test = new_def_test["text"]
    new_xv_test = vector.transform(new_x_test)
    pred_RFC = RFC.predict(new_xv_test)
    pred_DT  = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_LR  = LR.predict(new_xv_test)
    pred_KN  = KN.predict(new_xv_test)

    count = (pred_RFC + pred_DT + pred_GBC + pred_LR + pred_KN)/5

    if count < 0.25:
        return render_template('index.html', prediction_text=[str(news[0]),"\n\n This news is fake 😞 be aware!!"])
    elif count < 0.5:
        return render_template('index.html', prediction_text=[str(news[0]), "\n\n This news might be fake 😕"])
    elif count < 0.75:
        return render_template('index.html', prediction_text=[str(news[0]), "\n\n This news might be true 🙂"])
    else:
        return render_template('index.html', prediction_text=[str(news[0]), "\n\n This news is absolutely true 😃"]) 
     


if __name__ == "__main__":
    app.run(debug=True)
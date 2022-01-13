from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
app=Flask(__name__)
model=pickle.load(open('model_KMeans.pkl','rb'))
@app.route("/")
def form():
    return render_template("main.html")
@app.route("/predict",methods=['POST'])
def predict():
    balance=float(request.form['balance'])
    balance_freq=float(request.form['balance_freq'])
    purchases=float(request.form['purchases'])
    oneoff_purchases=float(request.form['oneoff_purchases'])
    installments_purchases=float(request.form['installments_purchases'])
    cash_advance=float(request.form['cash_advance'])
    purchases_freq=float(request.form['purchases_freq'])
    oneoff_purchases_freq=float(request.form['oneoff_purchases_freq'])
    purchases_installments=float(request.form['purchases_installments'])
    cash_adv_freq=float(request.form['cash_adv_freq'])
    cash_adv_trx=float(request.form['cash_adv_trx'])
    purchase_trx=float(request.form['purchase_trx'])
    credit_limit=float(request.form['credit_limit'])
    payments=float(request.form['payments'])
    min_payments=float(request.form['min_payments'])
    prc_full_pay=float(request.form['prc_full_pay'])
    tenure=float(request.form['tenure'])
    
    x_input=[balance,balance_freq,purchases, oneoff_purchases, installments_purchases,cash_advance,purchases_freq,
         oneoff_purchases_freq,purchases_installments,cash_adv_freq,cash_adv_trx,purchase_trx,credit_limit,payments,min_payments,
         prc_full_pay,tenure]
    x_input=scaler.fit_transform([x_input])
    prediction=model.predict(x_input)
    
  
    return render_template('main.html',prediction_text="{}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)



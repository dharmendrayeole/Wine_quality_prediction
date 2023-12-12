#Import the libraries
import pandas as pd
import numpy as np
import requests
import pickle

#get the data
data=pd.read_csv('winequalityN.csv')

#lebel the values in the quality variables
bins=(2,6.5,8)
label=['Bad','Good']
data['quality']=pd.cut(data['quality'],bins=bins,labels=label)

def predict():
    if request.method == 'POST':
        fixed_acidity=float(request.form['fixed acidity'])
        volatile_acidity=float(request.form['volatile acidity'])
        citric_acid=float(request.form['citric acid'])
        residual_sugar=float(request.form['residual sugar'])
        chlorides=float(request.form['chlorides'])
        free_sulphur_dioxide=float(request.form['free sulphur dioxide'])
        density=float(request.form['density'])
        pH=float(request.form['pH'])
        sulphates=float(request.form['sulphates'])
        alcohol=float(request.form['alcohol'])
        #load the pickle file
        filename='random_model.pickle'
        loaded_model=pickle.load(open(filename,'rb'))
        data=np.array([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,
                        chlorides,free_sulphur_dioxide,
                        density,pH,sulphates,alcohol]])
        my_prediction=loaded_model.predict(data)

if __name__ == '__main__':
    appp.run(debug=True)

#Import the libraries
import pandas as pd
import numpy as np

#get the data
data=pd.read_csv('winequalityN.csv')

#lebel the values in the quality variables
bins=(2,6.5,8)
label=['Bad','Good']
data['quality']=pd.cut(data['quality'],bins=bins,labels=label)

#Encode the values in quality in to 0,1
from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()
data['quality']=lc.fit_transform(data['quality'])

#rename the columns for better understanding
data.rename(columns={'pH':'pH','volatile acidity':'volatile_acidity',
                     'alcohol':'alcohol','residual sugar':'residual_sugar',
                     'free sulphur dioxide':'free_sulphur_dioxide','type':'type'
                     },inplace=True)

#split the data in to 'x' and 'y'
X=data.iloc[:,:-1]
Y=data['quality']

#split the data in to train and test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=0)


#Model building
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=0)
rfc.fit(X_train,Y_train)

#predict the model on test_set
rfc_pred=rf.predict(X_test)
print(rfc_pred[:9])

#get the performance of the model on test data
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test,rf_pred))
print(classification_report(y_test,rf_pred))
print(confusion_matrix(y_test,rf_pred))


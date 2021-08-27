#random forest model


#importing the library
#import numpy as np 
import pandas as pd 
#import matplotlib.pyplot as plt

dataset = pd.read_csv('coin_Bitcoin.csv')
#dataset2 = pd.read_csv('bitcoin_price.csv')
x = dataset.iloc[:,[4,5,6,8,9]].values
y = dataset.iloc[:,7].values

#encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

#splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
 
#fitting the randomforest classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion ='entropy')
classifier.fit(x_train,y_train)

#make the prediction
y_pred = classifier.predict(x_test)

#printing original and predicted value
print('printing the original test value')
print(y_test)
print('\n printing the predicted values ')
print(y_pred)

#print score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)



# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 13:06:05 2021

@author: KSHITIJ
"""
##part 1 ##

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
    
#import the data
data = pd.read_csv('coin_Bitcoin.csv', date_parser = True)
#data.tail()

data_save = data_training = data.drop(['SNo','Name','Symbol','Date','Volume','Marketcap'],axis=1)

#creating training and testing set
training_data = data[data['Date']<'2020-12-30']
testing_data = data[data['Date']>'2020-12-30']

data_training = training_data.drop(['SNo','Name','Symbol','Date','Volume','Marketcap'],axis=1)

#feature scaling using MinMaxScaler (normalization)
scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training) 

#creating the required dataset
X_train = [] 
Y_train = []
y=[]
#y.append(data_training[0:60])
#print(y)
#print(training_data.shape[0])
for i in range(60, data_training.shape[0]):
     X_train.append(data_training[i-60:i])
     Y_train.append(data_training[i])
X_train, Y_train = np.array(X_train), np.array(Y_train)


##part 2 ##
#building the LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 

#build the regressor
regressor = Sequential()
#layer 1st with dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 4)))
regressor.add(Dropout(rate = 0.2))
#layer 2 with dropout
regressor.add(LSTM(units = 60,return_sequences = True))
regressor.add(Dropout(0.3))
#layer 3 with dropout 
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(rate = 0.4))
#layer 4 with dropout
regressor.add(LSTM(units = 120))
regressor.add(Dropout(rate = 0.5))
#adding the output layer
regressor.add(Dense(units = 4))

#regressor.summary()

#compilling the LSTM
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')
#fitting the regressor
history = regressor.fit(X_train,Y_train, epochs = 20, batch_size = 50)

##part 3##

#making the prediction
past_60_data = training_data.tail(60)
full_testing = past_60_data.append(testing_data,ignore_index = True)
full_testing = full_testing.drop(['SNo','Name','Symbol','Date','Volume','Marketcap'],axis = 1)   

#feature scaling
inputs = scaler.transform(full_testing)

#building the datastructure
X_test = []
Y_test = []
for i in range (60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    Y_test.append(inputs[i]) 
X_test, Y_test = np.array(X_test), np.array(Y_test)

#predicting the vaues
Y_pred = regressor.predict(X_test) 
#inverse transforming the predicting value
Y_pred_scaled = scaler.inverse_transform(Y_pred)

##part 4##
#printing the graph
index = ['High','Low','Open','Close']
for i in range(0,4):
    #print(Y_pred[:,i])
    plt.figure(figsize=(14,5))
    plt.plot(Y_test[:,i], color = 'red', label = 'Real Bitcoin Price')
    plt.plot(Y_pred[:,i], color = 'green', label = 'Predicted Bitcoin Price')
    plt.title('Bitcoin prediction of '+ str(index[i]))
    plt.xlabel('Time')
    plt.ylabel(str(index[i]))
    plt.legend()
    plt.show()
    
##predict a single value for the next day
    
    
#save the model as pickle
import pickle

properties = {
        "model":regressor,
        "scaler":scaler,
        "data":data_save
        }


with open('lstmModel.pickle','wb') as f:
    pickle.dump(properties,f)
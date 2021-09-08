# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:50:46 2021

@author: KSHITIJ
"""
import pickle
import numpy as np
import pandas as pd
#load the lstm model 
pickle_in = open('lstmModel.pickle','rb')
properties = pickle.load(pickle_in)

model = properties['model']
scaler = properties['scaler']
dataPickled = properties['data']

#start making prediction
def predictor(day_Range = 1):
    #load the past 60 days data
    past_60_data = dataPickled.tail(60)    
    #prepare total data if future is to be predicted    
    full_data = past_60_data
    predicted_value = []
    for i in range(0,day_Range):
        #scale the data
        inputs = scaler.transform(full_data)
        #create a special data structure
        X_test=[]
        X_test.append(inputs[i:60+i]) 
        X_test = np.array(X_test)
        #predict the value
        Y_pred = model.predict(X_test)
        Y_pred_scaled = scaler.inverse_transform(Y_pred)
        #here y_pred_scaled is a numpy array so we have to concert it to dataframe for 
        #concatination
        Y_pred_scaled_dataframe = pd.DataFrame(Y_pred_scaled , columns = ['High','Low','Open','Close'])
        #update the dataset for next value prediction
        full_data = full_data.append(Y_pred_scaled_dataframe,ignore_index = True)
        #store the predicted value
        predicted_value.append(Y_pred_scaled[0])
    return predicted_value
values = predictor(2)
#print(values[1])
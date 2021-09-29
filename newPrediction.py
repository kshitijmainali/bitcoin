# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:50:46 2021

@author: KSHITIJ
"""
import pickle
import numpy as np
import pandas as pd
import keras 
#load the lstm model 
pickle_in = open('lstmModel.pickle','rb')
properties = pickle.load(pickle_in)

model = keras.models.load_model("regressor")
#properties['model']
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
values = predictor(60)
print(values)

#invest the amount 
#how much money will be made if I invest given money in argument
def moneyInvest(money,time = 1):
    #get today price
    today_price  = dataPickled.tail(1)['Close']
    #calculate total buyable share
    buyable = (money/today_price).apply(np.floor)
    if((buyable <= 0).bool()):
        return 'no stock is buyable with that much money'
    #in our principle only real number stock are buyable
    print('You can buy only' + str(buyable) + 'stock' )
    #predict tomorrows share value
    tomorrow_price= predictor(time)
    #predicted valeu is a whole list to show prediction on each day so we need 
    #to take only the last value
    tomorrow_price = tomorrow_price[-1][3]
    #accumulate tomorrow tota amount collectable
    tomorrow_money =tomorrow_price*buyable
    #print the prediction
    return tomorrow_money
asset = moneyInvest(50000)
print(asset)


#predict the amount I would have after given time if I invest given amount of money
#simply call the moneyInvest with time argument which default value is 1 for tomorrow
#price prediction
asset_3days = moneyInvest(100000,3)
print(asset_3days)


'''
The final case Invest after t days and see prediction after t+n days
'''

def after_n_invest(amount,invest_after,see_after):
    #get prediction for invest_after 
    price_at_invest = predictor(invest_after)
    price_at_invest = price_at_invest[-1][3]
    #compute buyable stock
    buyable = np.floor(amount/price_at_invest)
    if(buyable <= 0):
        return 'no stock is buyable with that much money'
    #in our principle only real number stock are buyable
    print('You can buy only ' + str(buyable) + ' stock' )
    #predict price for see_after
    price_at_see = predictor(see_after)
    #compute total price after see
    total_at_see = price_at_see [-1][3]* buyable
    #compute benifit
    benifit = total_at_see - (price_at_invest*buyable)
    return benifit
benifit = after_n_invest(50000,3,5)
print(benifit)
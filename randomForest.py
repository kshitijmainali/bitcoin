#random forest model


#importing the library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

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
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
regressor.fit(x_train,y_train)

#make the prediction
y_pred_randomForest = regressor.predict(x_test)


'''
#visualizing the result 
#we can't visualize accuracy of regression model like we use to do with 
#classification model so we have to rely on the graph

plt.scatter(np.arange(0,len(x_train),1),y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('opening and day high prices vs closing prices')
plt.xlabel('opening and day high prices')
plt.ylabel('closing price')
plt.show
'''
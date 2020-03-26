import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv('taxi.csv')
#print(data.tail())
#print(type(data))
#print (data.shape)
data_x=data.iloc[:,0:-1]
data_y=data.iloc[:,-1]
#print(data_y)
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.3,random_state=0)
reg=LinearRegression()
reg.fit(x_train,y_train)
#print("test score:",reg.score(x_test,y_test))
pickle.dump(reg,open('taxi.pkl','wb'))
model=pickle.load(open('taxi.pkl','rb'))
print("no of weekly ride are",model.predict([[80,1770000,6000,85]]))
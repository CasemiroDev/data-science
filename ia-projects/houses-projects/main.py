import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os
 
#------------------------------------------------------------------
#Predictions X to y
#Load the data, and separate the target
data = os.getcwd()
train = os.path.join(data, './randomic-forest-algorithms/train.csv')
home_data = pd.read_csv(train)
print(home_data.head())

y = home_data.SalePrice

#Create X and the features
features = ['LotArea','MSSubClass','MiscVal','GarageArea','GarageCars','Fireplaces','TotRmsAbvGrd','HalfBath','FullBath','2ndFlrSF','1stFlrSF','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1']
X = home_data[features]

#split into train and validation data
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)

#set the randomic forest model 
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))





#Yoceline Aralí Mata Ledezma A01562116

import LinearRegression as lr 
import pandas as pd

data = pd.read_csv('nutritionalinfo.csv')
y = data['Calorias']
X = data[['Grasas','Proteinas','Carbohidratos']]

model = lr.LinearRegression()
model.fit(X, y)

model.predict(X)

rmse = model.rmse(y, model.y_estimated)

print("Predicción de calorías dados los carbohidratos, grasas y proteínas")
print("RMSE: ", rmse)
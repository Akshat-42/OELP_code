import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from joblib import dump

data = pd.read_csv(r"test_pts1.csv")
#data2 = pd.read_csv(r"back_spin_pts.csv")
#data = pd.concat([data,data2],axis = 0)
data["p_f"] = (data["p_f"].apply(lambda x: list(map(float,x.strip("[]").split(',')))[0:2]))
p_f_coords = data['p_f'].apply(pd.Series)
p_f_coords = p_f_coords.rename(columns={0: 'land_x', 1: 'land_y'})
data = pd.concat([data, p_f_coords], axis=1)
data = data.drop(['p_x','p_z','phi','w_y'], axis=1)

# Velocity_ranges = [0,22,26,np.inf]
# labels = [0,1,2]
# data["v_cat"] = pd.cut(data['v_mag'],bins=Velocity_ranges,labels=labels,right=False)

features = ['land_x','land_y']
labels = ['p_y','v_mag']

X = data[features]
y = data[labels]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train,X_test, y_train, y_test =train_test_split( X_scaled,y_scaled,test_size=0.2,random_state=42)

model = Sequential([
    
    Dense(64,activation='relu',input_shape = (2,)),

    Dense(128, activation='relu'),

    Dense(128, activation='relu'),

    Dense(128, activation='relu'),

    Dense(y_train.shape[1], activation='linear')

])

model.summary()

early_stopper = EarlyStopping(monitor='val_loss', patience = 10)

model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 150, validation_data = (X_test, y_test))

model.save(r'test/Test1.keras')

dump(scaler_X,r'test/scaler_X(test).joblib')
dump(scaler_y,r'test/scaler_Y(test).joblib')

print("Training done and model saved")
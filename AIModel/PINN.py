import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from joblib import dump
from PINNSimulator import air_sim_tf

data = pd.read_csv(r"final_pts.csv")
# data2 = pd.read_csv(r"back_spin_pts.csv")
# data = pd.concat([data,data2],axis = 0)
data["p_f"] = (data["p_f"].apply(lambda x: list(map(float,x.strip("[]").split(',')))[0:2]))
p_f_coords = data['p_f'].apply(pd.Series)
p_f_coords = p_f_coords.rename(columns={0: 'land_x', 1: 'land_y'})
data = pd.concat([data, p_f_coords], axis=1)
data = data.drop('p_f', axis=1)

features = ['land_x','land_y']
labels = ['p_x','p_y','p_z','v_mag','phi','w_y']

X = data[features]
y = data[labels]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train,X_test, y_train, y_test =train_test_split( X_scaled,y_scaled,test_size=0.2,random_state=42)

mean_x = tf.constant(scaler_X.mean_, dtype=tf.float32)
scale_x = tf.constant(scaler_X.scale_, dtype=tf.float32)
mean_y = tf.constant(scaler_y.mean_, dtype=tf.float32)
scale_y = tf.constant(scaler_y.scale_, dtype=tf.float32)

model = load_model(r'temp/saved_model_3layer_128n.keras')

print("Model Loaded")

print("Starting PINN fine-tuning...")

optimizer = Adam(learning_rate=0.0001)
loss_fn = MeanSquaredError()
epochs = 10


train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train, tf.float32), tf.cast(X_train, tf.float32))
).batch(128).prefetch(tf.data.AUTOTUNE).cache()

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    for step, (x_batch, y_batch_target) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            pred_initial_cond_scaled = model(x_batch)
            pred_initial_cond = pred_initial_cond_scaled*scale_y + mean_y

            simulated_land_points = air_sim_tf(pred_initial_cond)

            simulated_land_points_scaled = (simulated_land_points - mean_x) / scale_x

            loss = loss_fn(y_batch_target, simulated_land_points_scaled)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step%100 == 0:
            print(f"  Step {step}: Loss = {loss.numpy()}")

print("PINN training complete")
model_name = 'PINN_model_3layer_128n.keras'
model.save(f"temp\{model_name}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from joblib import dump

# 1. Load Data
data = pd.read_csv(r"test_pts_vy.csv")

# 2. Process 'p_f' (Landing Coordinates)
# Convert string "[x, y]" to separate columns
data["p_f"] = (data["p_f"].apply(lambda x: list(map(float,x.strip("[]").split(',')))[0:2]))
p_f_coords = data['p_f'].apply(pd.Series)
p_f_coords = p_f_coords.rename(columns={0: 'land_x', 1: 'land_y'})
data = pd.concat([data, p_f_coords], axis=1)

# --- CRITICAL CHANGE 1: STOP DROPPING USEFUL COLUMNS ---
# We previously dropped these, but now we need them if we want to train on them.
# Only drop 'p_f' since we extracted x and y from it.
# If p_x and p_z are constant (e.g. always 0 and 2), you generally don't predict them.
# If they vary, remove them from this drop list.
data = data.drop(['p_f'], axis=1) 

# 3. Define Features (Inputs) and Labels (Outputs)

# Input: Where do we want the ball to land?
features = ['land_x','land_y']

# --- CRITICAL CHANGE 2: ADD NEW TARGETS ---
# Output: How do we achieve that? (Launch params)
# Added 'w_y' (Spin Magnitude) and 'phi' (Launch Angle)
labels = ['v_mag',"p_y"] 

X = data[features]
y = data[labels]

print(f"Training on {len(data)} samples.")
print(f"Inputs: {features}")
print(f"Outputs: {labels}")

# 4. Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 6. Build Model
model = Sequential([
    # Input shape matches number of features (2: land_x, land_y)
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    # Output shape automatically matches number of labels (now 4: p_y, v_mag, w_y, phi)
    Dense(y_train.shape[1], activation='linear')
])

model.summary()

# 7. Train
#early_stopper = EarlyStopping(monitor='val_loss', patience=10)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test),)# callbacks=[early_stopper])

# 8. Save Artifacts
model.save(r'test/Test2.keras')
dump(scaler_X, r'test/scaler_X(test2).joblib')
dump(scaler_y, r'test/scaler_Y(test2).joblib')

print("Training done and model saved")
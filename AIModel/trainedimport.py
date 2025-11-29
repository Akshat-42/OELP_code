import numpy as np
from keras.models import load_model
from joblib import load
import pandas as pd
from Forward import formatter,air_sim

#import dataset
data = pd.read_csv(r"test_pts1.csv")
data["p_f"] = (data["p_f"].apply(lambda x: list(map(float,x.strip("[]").split(',')))[0:2]))
p_f_coords = data['p_f'].apply(pd.Series)
p_f_coords = p_f_coords.rename(columns={0: 'land_x', 1: 'land_y'})
data = pd.concat([data, p_f_coords], axis=1)
data = data.drop('p_f', axis=1)

# Velocity_ranges = [0,22,26,np.inf]
# labels = [0,1,2]
# data["v_cat"] = pd.cut(data['v_mag'],bins=Velocity_ranges,labels=labels,right=False)

sample_size = 2000

test_rows = data.sample(sample_size)

print("Loading the model and scalers...")

np.set_printoptions(suppress=True, precision=2)

model_name = r'test/Test1.keras'
model = load_model(f'temp/{model_name}')


scaler_X = load(r'test/scaler_X(test).joblib')
scaler_y = load(r'test/scaler_y(test).joblib')

print("Model and scalers loaded successfully.")


test_vals = test_rows[['land_x','land_y']]

errors = []

new_input_scaled = scaler_X.transform(test_vals)


predicted_conditions_scaled = model.predict(new_input_scaled)

predicted_conditions = scaler_y.inverse_transform(predicted_conditions_scaled)


print("\n--- New Prediction ---")
# print(f"For Input (land_x, land_y): {test_vals}")
# print(f"Predicted Initial Conditions (p_x, p_y, p_z, v_mag, phi, w_y):\n{predicted_conditions}")
for i in range(len(test_vals)):
    errors.append((np.array(test_vals.iloc[i,:]) - np.array(air_sim(*formatter(*predicted_conditions[i])))))
errors_array = np.array(errors)
x_errors = errors_array[:,0]
y_errors = errors_array[:,1]
results = f"""Standard deviationX = {np.std(x_errors)}\nStandard deviationY = {np.std(y_errors)}
Mean errorX = {np.mean(x_errors)}\nMean errorY = {np.mean(y_errors)}"""
with open(r"test/Results.txt",'+a') as file:
    file.write('\n')
    file.write(model_name)
    file.write('\n')
    file.write(results)
print(results)



#model.summary()
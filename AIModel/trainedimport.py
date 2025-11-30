import numpy as np
from keras.models import load_model
from joblib import load
import pandas as pd
from Forward import formatter,air_sim

# #import dataset
# data = pd.read_csv(r"test_pts1.csv")
# data["p_f"] = (data["p_f"].apply(lambda x: list(map(float,x.strip("[]").split(',')))[0:2]))
# p_f_coords = data['p_f'].apply(pd.Series)
# p_f_coords = p_f_coords.rename(columns={0: 'land_x', 1: 'land_y'})
# data = pd.concat([data, p_f_coords], axis=1)
# data = data.drop('p_f', axis=1)

sample_size = 4000

# 1. Generate Random Data
test_rows = pd.DataFrame({
    # CHANGED: Now randomizing p_x (e.g., between -5 and 5 meters)
    "p_x": np.random.uniform(-5, 5, sample_size), #[0]* sample_size,#
    "p_y": np.random.uniform(-1.2, 1.2, sample_size),
    "p_z": np.random.uniform(1.8, 2.2, sample_size), #[2] * sample_size,
    "v_mag": np.random.uniform(18, 30, sample_size),
    "phi": np.random.uniform(0, 5, sample_size), #[3]*sample_size,#
    "w_y": np.random.uniform(180, 256, sample_size) 
})

print("Loading the model and scalers...")

np.set_printoptions(suppress=True, precision=2)

model_name = r'/home/akshat/code/OELP/OELP_code/temp/saved_model_3layer_128n.keras'
model = load_model(f'{model_name}')


scaler_X = load(r'/home/akshat/code/OELP/OELP_code/temp/scaler_X.joblib')
scaler_y = load(r'/home/akshat/code/OELP/OELP_code/temp/scaler_y.joblib')

print("Model and scalers loaded successfully.")


test_vals_list = []
for i in range(sample_size):
    # FIXED: Use 'test_rows' instead of 'data' to use the random generated values
    sim_inputs = formatter(
        p_x=test_rows["p_x"][i],  # New
        p_y=test_rows["p_y"][i],
        p_z = test_rows["p_z"][i], 
        v_mag=test_rows["v_mag"][i],
        phi=test_rows["phi"][i],  # New
        w_y=test_rows["w_y"][i]   # New
    )
    test_vals_list.append(air_sim(*sim_inputs))

# FIXED: Convert to DataFrame with columns so Scaler accepts it (Fixes ValueError)
test_vals = pd.DataFrame(test_vals_list, columns=['land_x', 'land_y'])



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
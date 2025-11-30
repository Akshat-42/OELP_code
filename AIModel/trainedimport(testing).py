import numpy as np
from keras.models import load_model
from joblib import load
import pandas as pd
from Forward import formatter,air_sim

# #import dataset
# data = pd.read_csv(r"/home/akshat/code/OELP/OELP_code/AIModel/test_pts_v.csv")
# data["p_f"] = (data["p_f"].apply(lambda x: list(map(float,x.strip("[]").split(',')))[0:2]))
# p_f_coords = data['p_f'].apply(pd.Series)
# p_f_coords = p_f_coords.rename(columns={0: 'land_x', 1: 'land_y'})
# data = pd.concat([data, p_f_coords], axis=1)
# data = data.drop('p_f', axis=1)

# Velocity_ranges = [0,22,26,np.inf]
# labels = [0,1,2]
# data["v_cat"] = pd.cut(data['v_mag'],bins=Velocity_ranges,labels=labels,right=False)

sample_size = 2000


# 1. Generate Random Data
test_rows = pd.DataFrame({
    # CHANGED: Now randomizing p_x (e.g., between -5 and 5 meters)
    "p_x": [0]* sample_size,#np.random.uniform(-5, 5, sample_size), 
    "p_y": np.random.uniform(-1.2, 1.2, sample_size),#[-1.0]*sample_size,#
    "p_z": [2] * sample_size,
    "v_mag": np.random.uniform(18, 30, sample_size),
    "phi": [3]*sample_size,#np.random.uniform(0, 5, sample_size),     
    "w_y":[200]*sample_size # np.random.uniform(180, 256, sample_size), 
})

#test_rows = data.sample(sample_size)

print("Loading the model and scalers...")

np.set_printoptions(suppress=True, precision=2)

model_name = r'Test1.keras'
model = load_model(f'/home/akshat/code/OELP/OELP_code/AIModel/test/Test2.keras')


scaler_X = load(r'/home/akshat/code/OELP/OELP_code/AIModel/test/scaler_X(test2).joblib')
scaler_y = load(r'/home/akshat/code/OELP/OELP_code/AIModel/test/scaler_Y(test2).joblib')

print("Model and scalers loaded successfully.")


# FIXED: Use a list for speed and correct 2D structure
test_vals_list = []
for i in range(sample_size):
    # FIXED: Use 'test_rows' instead of 'data' to use the random generated values
    sim_inputs = formatter(
        p_x=test_rows["p_x"][i],  # New
        p_y=test_rows["p_y"][i], 
        v_mag=test_rows["v_mag"][i],
        phi=test_rows["phi"][i],  # New
        w_y=test_rows["w_y"][i]   # New
    )
    test_vals_list.append(air_sim(*sim_inputs))

# FIXED: Convert to DataFrame with columns so Scaler accepts it (Fixes ValueError)
test_vals = pd.DataFrame(test_vals_list, columns=['land_x', 'land_y'])

#test_vals = data[['land_x','land_y']]

errors = []

new_input_scaled = scaler_X.transform(test_vals)


predicted_conditions_scaled = model.predict(new_input_scaled)

predicted_conditions = scaler_y.inverse_transform(predicted_conditions_scaled)

np.savetxt(r"Pred.txt",predicted_conditions)

print("\n--- New Prediction ---")
# print(f"For Input (land_x, land_y): {test_vals}")
# print(f"Predicted Initial Conditions (p_x, p_y, p_z, v_mag, phi, w_y):\n{predicted_conditions}")
for i in range(len(test_vals)):
    target_point = np.array(test_vals.iloc[i, :])
    
    # Extract ALL 5 predicted variables
    # WARNING: Check your training script labels order! 
    # This assumes labels = ['p_x', 'p_y', 'v_mag', 'w_y', 'phi']
    #pred_p_x = predicted_conditions[i][0]
    pred_p_y = predicted_conditions[i][1]
    pred_v_mag = predicted_conditions[i][0]
    #pred_w_y = predicted_conditions[i][2]
    #pred_phi = predicted_conditions[i][4]
    
    # Feed predictions back into simulator
    sim_inputs = formatter(
        p_x=0,
        p_y=pred_p_y, 
        v_mag=pred_v_mag,
        #w_y=pred_w_y,
        #phi=pred_phi
    )
    
    simulated_point = np.array(air_sim(*sim_inputs))
    errors.append(target_point - simulated_point)

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
import tensorflow as tf
from enum import IntEnum
import numpy as np

#Constants
ball_mass = tf.constant(.160, dtype=tf.float32) #kg
gf = tf.constant([0.0,0.0,-9.81], dtype=tf.float32) * ball_mass #N
ball_radius = tf.constant(0.035, dtype=tf.float32) #m
cor = tf.constant(0.5, dtype=tf.float32) #resititution
cf = tf.constant(0.35, dtype=tf.float32) #friction
dcf = tf.constant(0.45, dtype=tf.float32) #drag coefficient
rho = tf.constant(1.293, dtype=tf.float32)
c_m = tf.constant(0.025, dtype=tf.float32)
ball_area = np.pi*ball_radius.numpy()**2

class Comp(IntEnum):
    x = 0
    y = 1
    z = 2
@tf.function
def air_sim_tf(initial_conditions_batch):
    # trajectory = []
    # add_data(trajectory,0,pos_in.tolist(),v_in.tolist())

    px, py, pz = initial_conditions_batch[:,0], initial_conditions_batch[:,1], initial_conditions_batch[:,2]
    vmag, phi, w_y = initial_conditions_batch[:,3], initial_conditions_batch[:,4], initial_conditions_batch[:,5]
    
    curr_pos = tf.stack([px, py, pz], axis=1) # Shape (32, 3)
    
    curr_vel = tf.stack([vmag * tf.cos(phi), 
                         tf.zeros_like(vmag), # Use zeros_like to match batch shape
                         vmag * tf.sin(phi)], axis=1) # Shape (32, 3)
    
    spin_vect = tf.stack([tf.zeros_like(w_y), 
                          w_y, 
                          tf.zeros_like(w_y)], axis=1) # Shape (32, 3)
    
    t_step = tf.constant(0.04, dtype=tf.float32)
    
    loop_vars = (curr_pos,curr_vel)

    def condition(p,v):
        return tf.reduce_any(p[:, 2] > ball_radius)
    
    def body(p,v):
        
        v_mag_sq = tf.reduce_sum(tf.square(v), axis=1, keepdims=True) # Shape (32, 1)
        v_mag = tf.sqrt(v_mag_sq + 1e-6) # Shape (32, 1)
        
        drag_f = -0.5 * rho * ball_area * dcf * v * v_mag # v(32,3) * v_mag(32,1) = (32,3)
        
        spin_mag = tf.linalg.norm(spin_vect, axis=1, keepdims=True) + 1e-6 # Shape (32, 1)
        
        Cl = 0.54 * tf.pow((spin_mag * ball_radius / v_mag), 0.4) # Shape (32, 1)
        
        # tf.linalg.cross works on batches automatically
        lift_f = Cl * (0.5) * rho * ball_area * tf.linalg.cross(spin_vect, v) * v_mag / spin_mag
        
        total_force = drag_f + gf + lift_f # gf will "broadcast"
        
        # --- 2. Update State (Vectorized) ---
        acc_new = total_force / ball_mass
        v_new = v + acc_new * t_step
        p_new = p + v_new * t_step 
        
        # --- 3. Masking ---
        # Stop updating balls that have already landed
        # (This is advanced, but good. For now, you can skip it,
        # but it's more efficient to 'freeze' them)
        is_airborne = p[:, 2] > ball_radius
        is_airborne_float = tf.cast(is_airborne, dtype=tf.float32)[:, tf.newaxis]
        
        # Only apply updates to the airborne balls
        v_new = v_new * is_airborne_float + v * (1.0 - is_airborne_float)
        p_new = p_new * is_airborne_float + p * (1.0 - is_airborne_float)

        return p_new, v_new
    
    final_p, final_v = tf.while_loop(condition,body,loop_vars)

    return final_p[:,0:2]
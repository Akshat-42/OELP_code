import numpy as np
from enum import IntEnum
import csv
from tqdm import tqdm


#Constants
ball_mass = .160 #kg
gf = np.array([0,0,-9.81])*ball_mass #N
ball_radius = 0.035 #m
cor = 0.5 #resititution
cf = 0.35 #friction
dcf = 0.45 #drag coefficient
rho = 1.293
c_m = 0.025

#conversions
# spin_mag_rads = spin_mag*2*np.pi
# spin_angle_rad = spin_angle*np.pi/180

ball_area = np.pi*ball_radius**2

class Comp(IntEnum):
    x = 0
    y = 1
    z = 2



def add_data(lst,sample,position,velocity):
    lst.append({
        "sample" : sample,
        "position":position,
        "velocity":velocity,
    })


def bounce_calc(v,spin_vect):
    vz_prev = v[Comp.z]
    v[Comp.z] = -v[Comp.z]*cor

    v_surf = np.cross([0,0,ball_radius],spin_vect)

    if np.linalg.norm(v_surf)>0:

        max_friction_impulse_magnitude = cf*ball_mass*(1+cor)*vz_prev

        impulse_stop_slip = np.linalg.norm(v_surf)*ball_mass
        friction_impulse_magnitude = min(max_friction_impulse_magnitude,impulse_stop_slip)
        
        impulse_unit_vector = -v_surf/np.linalg.norm(v_surf)
        friction_impulse = impulse_unit_vector*friction_impulse_magnitude

        v += friction_impulse/ball_mass


def air_sim(pos_in,v_in,sps,gf,mb,w_in):
    # trajectory = []
    # add_data(trajectory,0,pos_in.tolist(),v_in.tolist())
    curr_pos = pos_in
    curr_vel = v_in
    curr_acc = gf/mb
    sample = 1
    t_step = 1/sps
    spin_vect = w_in
    spin_mag = np.linalg.norm(spin_vect)
    while curr_pos[Comp.z] > ball_radius or curr_vel[Comp.z]>0:
        # t = sample/sps
        drag_f = -0.5*rho*ball_area*dcf*curr_vel*np.linalg.norm(curr_vel)
        drag_acc = drag_f/mb
        Cl = 0.54*(spin_mag*ball_radius/np.linalg.norm(curr_vel))**0.4
        lift_f = Cl*(1/2)*rho*ball_area*np.cross(spin_vect,curr_vel)*np.linalg.norm(curr_vel)/spin_mag
        lift_acc = lift_f/mb
        curr_acc = drag_acc + gf/mb + lift_acc
        curr_vel += curr_acc*t_step
        curr_pos += curr_vel*t_step
        # add_data(trajectory,sample,curr_pos.tolist(),curr_vel.tolist())
        # sample+=1
    #add_data(final_pts,sample,curr_pos.tolist(),curr_vel.tolist())
    # bounce_calc(curr_vel,spin_vect)

    # while curr_pos[Comp.x] < 20.12:
    #     # t = sample/sps
    #     curr_vel += curr_acc*t_step
    #     curr_pos += curr_vel*t_step
    #     add_data(trajectory,sample,curr_pos.tolist(),curr_vel.tolist())
    #     sample+=1

    # return trajectory
    return curr_pos.tolist()[0:2]


# test_trajectory = air_sim(pos_in,v_in,60,gf,ball_mass)



# output_file = r"Trajectory.json"
# with open(output_file,"w") as file:
#     json.dump(test_trajectory,file,indent=4) 

final_pts = []
P_x = [0]#np.linspace(0,2,10)
P_y = np.linspace(-1.2,1.2,25) #[-1.0]#
P_z = [2]#np.linspace(1.8,2.2,4)
V_mag = np.linspace(18,30,100)
Phi = [3]#np.linspace(0,5,6)
W_y = [200]#np.linspace(180,256,40)#


if __name__ == '__main__':
    final_pts_file = r"test_pts_vy.csv"
    with open(final_pts_file,"w+",newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["p_x","p_y","p_z","v_mag","phi","w_y","p_f"])
        for p_x in P_x:
            for p_y in P_y:
                for p_z in P_z:
                    for v_mag in V_mag:
                        for phi in Phi:
                            phi_rad = phi*np.pi/180
                            for w_y in W_y:
                                loc_in = np.array([p_x,p_y,p_z])
                                v_in = np.array([v_mag*np.cos(phi_rad),0,v_mag*np.sin(phi_rad)])
                                spin_vect = np.array([0,w_y,0])
                                p_f = air_sim(loc_in,v_in,60,gf,ball_mass,w_in=spin_vect)
                                # if(p_f[0]<20 and p_f[0]>10 and p_f[1]<1.5 and p_f[1]>-1.5):      
                                #     writer.writerow([p_x,p_y,p_z,v_mag,phi,w_y,p_f])
                                # else:
                                #     break
                                writer.writerow([p_x,p_y,p_z,v_mag,phi,w_y,p_f])
        print("----Completed----")                            
# (p_x, p_y, p_z, v_mag, phi, w_y) = (0.19,-1.22,1.8,18.68,0.,221.18)
def formatter(p_x = 0, p_y = -1.0, p_z = 2, v_mag = 24, phi = 3, w_y = 200):
    loc_in = np.array([p_x,p_y,p_z])
    v_in = np.array([v_mag*np.cos(phi*np.pi/180),0,v_mag*np.sin(phi*np.pi/180)])
    spin_vect = np.array([0,w_y,0])
    return (loc_in,v_in,60,gf,ball_mass,spin_vect)

#print(air_sim(*formatter(p_x, p_y, p_z, v_mag, phi, w_y)))
     
    



        

        
        

        


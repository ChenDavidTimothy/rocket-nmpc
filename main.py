# main.py

from src.simulation.rocket_simulation import RocketMPCSimulation, RocketMPCParams
from src.simulation.rocket_animation import RocketMPCAnimation, RocketAnimationParams
import numpy as np

def main():
    # Initialize MPC parameters
    params = RocketMPCParams(
        # Time parameters
        T=0.05,              # Time step [s]
        N=120,               # Prediction horizon
        
        # Rocket physical parameters
        mass=300.0,             # Mass [kg]
        g=9.81,                 # Gravity [m/s^2]
        l=7.5,                  # Thrust moment arm [m]
        J=9500.0,               # Moment of inertia [kg*m^2]
        
        # Control constraints
        F_max=10000.0,          # Maximum thrust [N]
        F_min=0.0,              # Minimum thrust [N]
        gimbal_max=np.pi/30,    # Maximum gimbal angle [rad] 
        gimbal_min=-np.pi/30,   # Minimum gimbal angle [rad] 
        
        # State constraints
        x_max=200.0,            # Maximum altitude [m]
        x_min=0.0,              # Minimum altitude [m]
        z_max=45.0,             # Maximum lateral position [m]
        z_min=-45.0,            # Minimum lateral position [m]
        theta_max=16*np.pi/2,  # Maximum pitch angle [rad] 
        theta_min=-16*np.pi/2  # Minimum pitch angle [rad] 
    )
    
    # Updated cost matrices with stronger attitude tracking
    params.Q = np.diag([
        3000.0,     # x position (altitude)
        100.0,      # z position (lateral)
        200.0,      # u velocity (vertical)
        200.0,      # w velocity (lateral)
        3000.0,     # theta (attitude) 
        500.0       # q (angular velocity) 
    ])
    
    # Reduced control costs to allow more aggressive control
    params.R = np.diag([
        1e-6,     # Thrust cost 
        100       # Gimbal angle cost
    ])
    
    # Terminal cost matrix with even stronger attitude weights
    params.Q_terminal = np.diag([
        5000.0,     # x position
        1000.0,     # z position
        500.0,      # u velocity
        500.0,      # w velocity
        5000.0,     # theta 
        1000.0      # q 
    ])
    
    # Initialize animation parameters based on simulation parameters
    animation_params = RocketAnimationParams(
        # Physical parameters from simulation
        mass=params.mass,
        g=params.g,
        l=params.l,
        J=params.J,
        
        # State constraints from simulation
        x_max=params.x_max,
        x_min=params.x_min,
        z_max=params.z_max,
        z_min=params.z_min,
        theta_max=params.theta_max,
        theta_min=params.theta_min,
        
        # Control constraints from simulation
        F_max=params.F_max,
        F_min=params.F_min,
        gimbal_max=params.gimbal_max,
        gimbal_min=params.gimbal_min,
        
        # Visualization parameters
        rocket_length=15.0,
        rocket_width=2,
        nozzle_length=0.4,
        nozzle_width=0.2,
        fin_length=0.4,
        fin_width=0.2,
        margin_factor=1.2,
        ground_height_factor=0.05
    )
    
    # Create simulation instance
    rocket_sim = RocketMPCSimulation(params)
    
    # Initial and target states
    x0 = np.array([
        180,            # Initial altitude [m]
        30,             # Initial lateral position [m]
        0.0,            # Initial vertical velocity [m/s]
        -2.0,           # Initial lateral velocity [m/s]
        1*np.pi/6,      # Initial attitude [rad]
        0               # Initial angular velocity [rad/s]
    ])
    
    xs = np.array([
        30,             # Target altitude [m]
        0.0,            # Target lateral position [m]
        0.0,            # Target vertical velocity [m/s]
        0.0,            # Target lateral velocity [m/s]
        0.0,            # Target attitude [rad]
        0.0             # Target angular velocity [rad/s]
    ])
    
    print("Starting Rocket MPC Simulation...")
    print("\nInitial State:")
    print(f"Altitude: {x0[0]:.1f} m")
    print(f"Lateral position: {x0[1]:.1f} m")
    print(f"Vertical velocity: {x0[2]:.1f} m/s")
    print(f"Lateral velocity: {x0[3]:.1f} m/s")
    print(f"Attitude: {np.rad2deg(x0[4]):.1f} degrees")
    print(f"Angular velocity: {np.rad2deg(x0[5]):.1f} deg/s")
    
    print("\nTarget State:")
    print(f"Altitude: {xs[0]:.1f} m")
    print(f"Lateral position: {xs[1]:.1f} m")
    print(f"Vertical velocity: {xs[2]:.1f} m/s")
    print(f"Lateral velocity: {xs[3]:.1f} m/s")
    print(f"Attitude: {np.rad2deg(xs[4]):.1f} degrees")
    print(f"Angular velocity: {np.rad2deg(xs[5]):.1f} deg/s")
    
    try:
        # Run simulation
        xx, t, xx1, u_cl = rocket_sim.simulate(x0, xs)
        
        # Print results
        final_state = xx[:, -1]
        final_error = np.linalg.norm(final_state - xs)
        
        print("\nSimulation Complete!")
        print("\nFinal State:")
        print(f"Altitude: {final_state[0]:.2f} m")
        print(f"Lateral position: {final_state[1]:.2f} m")
        print(f"Vertical velocity: {final_state[2]:.2f} m/s")
        print(f"Lateral velocity: {final_state[3]:.2f} m/s")
        print(f"Attitude: {np.rad2deg(final_state[4]):.2f} degrees")
        print(f"Angular velocity: {np.rad2deg(final_state[5]):.2f} deg/s")
        print(f"Final error: {final_error:.4f}")
        
        # Calculate flight statistics
        max_altitude = np.max(xx[0, :])
        max_lateral = np.max(abs(xx[1, :]))
        max_vert_vel = np.max(abs(xx[2, :]))
        max_lat_vel = np.max(abs(xx[3, :]))
        max_attitude = np.max(abs(xx[4, :]))
        max_thrust = np.max([u[0] for u in u_cl]) if u_cl else 0
        max_gimbal = np.max([abs(u[1]) for u in u_cl]) if u_cl else 0
        
        print("\nFlight Statistics:")
        print(f"Maximum altitude: {max_altitude:.2f} m")
        print(f"Maximum lateral displacement: {max_lateral:.2f} m")
        print(f"Maximum vertical velocity: {max_vert_vel:.2f} m/s")
        print(f"Maximum lateral velocity: {max_lat_vel:.2f} m/s")
        print(f"Maximum attitude: {np.rad2deg(max_attitude):.2f} degrees")
        print(f"Maximum thrust: {max_thrust:.2f} N")
        print(f"Maximum gimbal angle: {np.rad2deg(max_gimbal):.2f} degrees")
        
        # Create and display animation
        print("\nGenerating animation...")
        animator = RocketMPCAnimation(params=animation_params, fps=30)
        animation = animator.create_animation(
            t=t,
            xx=xx,
            xx1=xx1,
            u_cl=u_cl,
            xs=xs,
            N=params.N,
            T=params.T,
            save_path='rocket_mpc_simulation.mp4'
        )
        
        print("\nDisplaying animation...")
        animator.show()
        
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
        print("Check solver settings and parameters if optimization fails.")

if __name__ == "__main__":
    main()
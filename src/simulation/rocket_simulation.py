import casadi as ca
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

@dataclass
class RocketMPCParams:
    T: float = 0.1              # Time step [s]
    N: int = 20                 # Prediction horizon
    
    # Rocket physical parameters
    mass: float = 10.0      # Mass [kg]
    g: float = 9.81         # Gravity [m/s^2]
    l: float = 7.5          # Thrust moment arm [m]
    J: float = 2.0          # Moment of inertia [kg*m^2]
    
    # Control constraints
    F_max: float = 150.0         # Maximum thrust [N]
    F_min: float = 0.0           # Minimum thrust [N]
    gimbal_max: float = np.pi/6  # Maximum gimbal angle [rad]
    gimbal_min: float = -np.pi/6 # Minimum gimbal angle [rad]
    
    # State constraints - Modified to allow wider angle range
    x_max: float = 30.0             # Maximum altitude [m]
    x_min: float = 0.0              # Minimum altitude [m]
    z_max: float = 15.0             # Maximum lateral position [m]
    z_min: float = -15.0            # Minimum lateral position [m]
    theta_max: float = 4*np.pi      # Maximum pitch angle [rad] - Increased to allow multiple rotations
    theta_min: float = -4*np.pi     # Minimum pitch angle [rad] - Increased to allow multiple rotations
    
    # Cost matrices with adjusted weights
    Q: np.ndarray = field(default_factory=lambda: np.diag([
        100.0,    # x position (altitude)
        100.0,    # z position (lateral)
        200.0,    # u velocity (vertical)
        200.0,    # w velocity (lateral)
        3000.0,   # theta (attitude)
        500.0     # q (angular velocity)
    ]))
    
    R: np.ndarray = field(default_factory=lambda: np.diag([
        1e-6,    # Thrust cost
        100.0    # Gimbal angle cost
    ]))
    
    Q_terminal: np.ndarray = field(default_factory=lambda: np.diag([
        1000.0,   # x position
        1000.0,   # z position
        500.0,    # u velocity
        500.0,    # w velocity
        5000.0,   # theta 
        1000.0    # q 
    ]))

class RocketMPCSimulation:
    def __init__(self, params: RocketMPCParams):
        self.params = params
        self._setup_optimization_problem()

    def _setup_optimization_problem(self):
        # State and control variables remain the same
        self.x = ca.SX.sym('x')
        self.z = ca.SX.sym('z')
        self.u = ca.SX.sym('u')
        self.w = ca.SX.sym('w')
        self.theta = ca.SX.sym('theta')
        self.q = ca.SX.sym('q')
        
        self.states = ca.vertcat(self.x, self.z, self.u, self.w, self.theta, self.q)
        self.n_states = self.states.size1()
        
        self.F = ca.SX.sym('F')
        self.mu_p = ca.SX.sym('mu_p')
        self.controls = ca.vertcat(self.F, self.mu_p)
        self.n_controls = self.controls.size1()
        
        # System dynamics
        rhs = ca.vertcat(
            self.u * ca.cos(self.theta) + self.w * ca.sin(self.theta),
            -self.u * ca.sin(self.theta) + self.w * ca.cos(self.theta),
            self.F * ca.cos(self.mu_p) / self.params.mass - self.params.g * ca.cos(self.theta) - self.q * self.w,
            -self.F * ca.sin(self.mu_p) / self.params.mass - self.params.g * ca.sin(self.theta) + self.q * self.u,
            self.q,
            -self.F * self.params.l * ca.sin(self.mu_p) / self.params.J
        )
        
        self.f = ca.Function('f', [self.states, self.controls], [rhs], ['states', 'controls'], ['rhs'])
        
        # Decision variables
        self.U = ca.SX.sym('U', self.n_controls, self.params.N)
        self.P = ca.SX.sym('P', self.n_states + self.n_states)
        self.X = ca.SX.sym('X', self.n_states, self.params.N + 1)
        
        # Initialize objective and constraints
        obj = 0
        g = []
        
        # Initial condition constraint
        g.append(self.X[:, 0] - self.P[:self.n_states])
        
        # Get reference states
        ref_x = self.P[self.n_states:]
        
        # Build optimization problem with improved angle handling
        for k in range(self.params.N):
            st = self.X[:, k]
            con = self.U[:, k]
            st_next = self.X[:, k + 1]
            
            # Create state error vector with improved angle handling
            state_error = st - ref_x
            angle_error = st[4] - ref_x[4]
            state_error[4] = ca.atan2(ca.sin(angle_error), ca.cos(angle_error))
            
            # Cost function
            obj += ca.mtimes([state_error.T, self.params.Q, state_error])
            obj += ca.mtimes([con.T, self.params.R, con])
            obj += 0.01 * con[0]  
            
            k1 = self.f(states=st, controls=con)['rhs']
            k2 = self.f(states=st + self.params.T/2 * k1, controls=con)['rhs']
            k3 = self.f(states=st + self.params.T/2 * k2, controls=con)['rhs']
            k4 = self.f(states=st + self.params.T * k3, controls=con)['rhs']
            st_next_rk4 = st + self.params.T/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            g.append(st_next - st_next_rk4)
            
            if k == self.params.N - 1:
                final_error = st_next - ref_x
                final_angle_error = st_next[4] - ref_x[4]
                final_error[4] = ca.atan2(ca.sin(final_angle_error), ca.cos(final_angle_error))
                obj += ca.mtimes([final_error.T, self.params.Q_terminal, final_error])
        
        OPT_variables = ca.vertcat(
            ca.reshape(self.X, self.n_states * (self.params.N + 1), 1),
            ca.reshape(self.U, self.n_controls * self.params.N, 1)
        )
        
        nlp_prob = {
            'f': obj,
            'x': OPT_variables,
            'g': ca.vertcat(*g),
            'p': self.P
        }

        opts = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6,
            'ipopt.warm_start_init_point': 'yes'
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        self.args = self._setup_constraints()
    
    def _setup_constraints(self) -> Dict:
        lbx = []
        ubx = []
        
        for _ in range(self.params.N + 1):
            lbx.extend([self.params.x_min,      # x lower bound
                       self.params.z_min,       # z lower bound
                       -100,                    # u lower bound
                       -100,                    # w lower bound
                       self.params.theta_min,   # theta lower bound 
                       -8*np.pi])               # q lower bound
            
            ubx.extend([self.params.x_max,      # x upper bound
                       self.params.z_max,       # z upper bound
                       100,                     # u upper bound
                       100,                     # w upper bound
                       self.params.theta_max,   # theta upper bound 
                       8*np.pi])                # q upper bound
        
        for _ in range(self.params.N):
            lbx.extend([self.params.F_min, self.params.gimbal_min])
            ubx.extend([self.params.F_max, self.params.gimbal_max])
        
        return {
            'lbg': np.zeros(self.n_states * (self.params.N + 1)),
            'ubg': np.zeros(self.n_states * (self.params.N + 1)),
            'lbx': lbx,
            'ubx': ubx
        }
    
    @staticmethod
    def shift(T: float, t0: float, x0: np.ndarray, u: np.ndarray, f) -> Tuple[float, np.ndarray, np.ndarray]:
        st = x0
        con = u[0, :]
        f_value = f(states=st, controls=con)['rhs']
        st = np.array(st + T * f_value).flatten()
        t0 = t0 + T
        u_rest = np.vstack((u[1:, :], u[-1, :]))
        return t0, st, u_rest
    
    def simulate(self, x0: np.ndarray, xs: np.ndarray, sim_time: float = 20.0) -> Tuple[np.ndarray, List, List, List]:
        t0 = 0
        x = x0.copy()
        xx = np.zeros((self.n_states, 1))
        xx[:, 0] = x0
        t = [t0]
        
        u0 = np.zeros((self.params.N, self.n_controls))
        X0 = np.tile(x0, (self.params.N + 1, 1))
        
        mpciter = 0
        xx1 = []
        u_cl = []
        
        while mpciter < sim_time / self.params.T:
            pos_error = np.linalg.norm(x[0:2] - xs[0:2])
            vel_error = np.linalg.norm(x[2:4] - xs[2:4])
            
            angle_error = np.arctan2(np.sin(x[4] - xs[4]), np.cos(x[4] - xs[4]))
            rate_error = abs(x[5] - xs[5])
            
            if (mpciter > 10 and
                pos_error < 0.1 and 
                vel_error < 0.1 and 
                abs(angle_error) < np.deg2rad(2) and
                rate_error < np.deg2rad(2)):
                break
            
            self.args['p'] = np.concatenate((x, xs))
            self.args['x0'] = np.concatenate((X0.flatten(), u0.flatten()))
            
            sol = self.solver(
                x0=self.args['x0'],
                lbx=self.args['lbx'],
                ubx=self.args['ubx'],
                lbg=self.args['lbg'],
                ubg=self.args['ubg'],
                p=self.args['p']
            )
            
            u = np.reshape(sol['x'][self.n_states*(self.params.N+1):].full().T, 
                         (self.params.N, self.n_controls))
            xx1.append(np.reshape(sol['x'][:self.n_states*(self.params.N+1)].full().T, 
                                (self.params.N+1, self.n_states)))
            u_cl.append(u[0, :])
            
            t.append(t0)
            t0, x, u0 = self.shift(self.params.T, t0, x, u, self.f)
            
            xx = np.hstack((xx, x.reshape(self.n_states, 1)))
            
            X0 = np.reshape(sol['x'][:self.n_states*(self.params.N+1)].full().T, 
                          (self.params.N+1, self.n_states))
            X0 = np.vstack((X0[1:], X0[-1]))
            
            mpciter += 1
        
        return xx, t, xx1, u_cl
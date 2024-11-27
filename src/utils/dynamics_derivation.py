

from sympy import Matrix, symbols, cos, sin, simplify, init_printing
init_printing(use_unicode=True)

# 1. Define symbolic variables
# ===========================
# States and angles
phi, theta, psi = symbols('phi theta psi')  # Euler angles
u, v, w = symbols('u v w')                  # Linear velocities
p, q, r = symbols('p q r')                  # Angular velocities

# Physical parameters
m, g, F, S, d, l = symbols('m g F S d l')   # Mass, gravity, thrust, area, distances
Jl, Jt = symbols('J_l J_t')                 # Inertia terms
mup, mu2 = symbols('mu_p mu2')              # Control angles

# Aerodynamic coefficients
CA, CY, CN = symbols('C_A C_Y C_N')         # Force coefficients
Cl, Cm, Cn = symbols('C_l C_m C_n')         # Moment coefficients

def c(x): return cos(x)
def s(x): return sin(x)

# 2. Reference Frame Fransformations
# ================================
# Define rotation matrices
R_x = Matrix([
    [1, 0, 0],
    [0, c(phi), -s(phi)],
    [0, s(phi), c(phi)]
])

R_y = Matrix([
    [c(theta), 0, s(theta)],
    [0, 1, 0],
    [-s(theta), 0, c(theta)]
])

R_z = Matrix([
    [c(psi), -s(psi), 0],
    [s(psi), c(psi), 0],
    [0, 0, 1]
])

# Combined rotation matrix
R_3D = simplify(R_z * R_y * R_x)

# 3. Forces and Moments
# ====================
# Forces in 3D
fg_3D = R_3D.transpose() * Matrix([-m*g, 0, 0])
fp_3D = Matrix([
    F * c(mup) * c(mu2),
    -F * c(mup) * s(mu2),
    -F * s(mup)
])
fa_3D = Matrix([-CA * S, CY * S, -CN * S])

# Moments in 3D
tau_p_3D = Matrix([
    0,
    -F * s(mup) * l,
    F * c(mup) * s(mu2) * l,
])
tau_a_3D = Matrix([
    Cl * S * d,
    Cm * S * d,
    Cn * S * d
])

# 4. State Space Representations
# ============================

def print_state_space(name, R, fg, fp, fa, tau_p, tau_a, subs=None):
    print(f"\n=== {name} ===")
    
    # Apply substitutions if provided
    if subs:
        R = R.subs(subs)
        fg = fg.subs(subs)
        fp = fp.subs(subs)
        fa = fa.subs(subs)
        tau_p = tau_p.subs(subs)
        tau_a = tau_a.subs(subs)
    
    # Fotal forces and moments
    f_total = fg + fp + fa
    tau_total = tau_p + tau_a
    
    # Velocity vector
    v_b = Matrix([u, v, w])
    if subs:
        v_b = v_b.subs(subs)
    
    # Angular velocity vector
    omega = Matrix([p, q, r])
    if subs:
        omega = omega.subs(subs)
    
    # Skew symmetric matrix
    def skew(v):
        return Matrix([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    S_omega = skew(omega)
    
    # Inertia matrix
    J = Matrix([
        [Jl, 0, 0],
        [0, Jt, 0],
        [0, 0, Jt]
    ])
    
    # Position derivatives (inertial frame)
    pos_dot = R * v_b
    print("\nPosition Derivatives (Inertial Frame):")
    print("ẋ =", simplify(pos_dot[0]))
    print("ẏ =", simplify(pos_dot[1]))
    print("ż =", simplify(pos_dot[2]))
    
    # Velocity derivatives (body frame)
    v_dot = -S_omega * v_b + f_total/m
    print("\nVelocity Derivatives (Body Frame):")
    print("u̇ =", simplify(v_dot[0]))
    print("v̇ =", simplify(v_dot[1]))
    print("ẇ =", simplify(v_dot[2]))
    
    # Angular velocity derivatives
    omega_dot = J.inv() * (-S_omega * J * omega + tau_total)
    print("\nAngular Velocity Derivatives:")
    print("ṗ =", simplify(omega_dot[0]))
    print("q̇ =", simplify(omega_dot[1]))
    print("ṙ =", simplify(omega_dot[2]))
    
    # Euler angle derivatives
    # Fhis matrix relates euler rates to body rates
    H = Matrix([
        [1, s(phi)*s(theta)/c(theta), c(phi)*s(theta)/c(theta)],
        [0, c(phi), -s(phi)],
        [0, s(phi)/c(theta), c(phi)/c(theta)]
    ])
    if subs:
        H = H.subs(subs)
    
    euler_dot = H * omega
    print("\nEuler Angle Derivatives:")
    print("φ̇ =", simplify(euler_dot[0]))
    print("θ̇ =", simplify(euler_dot[1]))
    print("ψ̇ =", simplify(euler_dot[2]))

# Print full 3D state space
print_state_space("Full 3D State Space", R_3D, fg_3D, fp_3D, fa_3D, tau_p_3D, tau_a_3D)

# Print 3D state space without aerodynamics
zero_aero = {CA: 0, CY: 0, CN: 0, Cl: 0, Cm: 0, Cn: 0, S: 0}
print_state_space("3D State Space without Aerodynamics", 
                 R_3D, fg_3D, fp_3D, fa_3D, tau_p_3D, tau_a_3D, 
                 zero_aero)

# Print 2D state space
planar_subs = {
    v: 0, p: 0, r: 0, phi: 0, psi: 0, mu2: 0,  # Planar motion
    CA: 0, CY: 0, CN: 0, Cl: 0, Cm: 0, Cn: 0, S: 0  # No aerodynamics
}
print_state_space("2D Simplified State Space", 
                 R_3D, fg_3D, fp_3D, fa_3D, tau_p_3D, tau_a_3D, 
                 planar_subs)
"""
=== Full 3D State Space ===

Position Derivatives (Inertial Frame):
ẋ = u*cos(psi)*cos(theta) + v*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi)) + w*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))
ẏ = u*sin(psi)*cos(theta) + v*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)) - w*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi))
ż = -u*sin(theta) + v*sin(phi)*cos(theta) + w*cos(phi)*cos(theta)

Velocity Derivatives (Body Frame):
u̇ = -C_A*S/m + F*cos(mu2)*cos(mu_p)/m - g*cos(psi)*cos(theta) - q*w + r*v
v̇ = (C_Y*S - F*sin(mu2)*cos(mu_p) - g*m*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi)) + m*(p*w - r*u))/m
ẇ = (-C_N*S - F*sin(mu_p) - g*m*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi)) + m*(-p*v + q*u))/m

Angular Velocity Derivatives:
ṗ = C_l*S*d/J_l
q̇ = (C_m*S*d - J_l*p*r + J_t*p*r - F*l*sin(mu_p))/J_t
ṙ = (C_n*S*d + J_l*p*q - J_t*p*q + F*l*sin(mu2)*cos(mu_p))/J_t

Euler Angle Derivatives:
φ̇ = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
θ̇ = q*cos(phi) - r*sin(phi)
ψ̇ = (q*sin(phi) + r*cos(phi))/cos(theta)

=== 3D State Space without Aerodynamics ===

Position Derivatives (Inertial Frame):
ẋ = u*cos(psi)*cos(theta) + v*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi)) + w*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))
ẏ = u*sin(psi)*cos(theta) + v*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)) - w*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi))
ż = -u*sin(theta) + v*sin(phi)*cos(theta) + w*cos(phi)*cos(theta)

Velocity Derivatives (Body Frame):
u̇ = F*cos(mu2)*cos(mu_p)/m - g*cos(psi)*cos(theta) - q*w + r*v
v̇ = -F*sin(mu2)*cos(mu_p)/m - g*sin(phi)*sin(theta)*cos(psi) + g*sin(psi)*cos(phi) + p*w - r*u
ẇ = -F*sin(mu_p)/m - g*sin(phi)*sin(psi) - g*sin(theta)*cos(phi)*cos(psi) - p*v + q*u

Angular Velocity Derivatives:
ṗ = 0
q̇ = (-J_l*p*r + J_t*p*r - F*l*sin(mu_p))/J_t
ṙ = (J_l*p*q - J_t*p*q + F*l*sin(mu2)*cos(mu_p))/J_t

Euler Angle Derivatives:
φ̇ = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
θ̇ = q*cos(phi) - r*sin(phi)
ψ̇ = (q*sin(phi) + r*cos(phi))/cos(theta)

=== 2D Simplified State Space ===

Position Derivatives (Inertial Frame):
ẋ = u*cos(theta) + w*sin(theta)
ẏ = 0
ż = -u*sin(theta) + w*cos(theta)

Velocity Derivatives (Body Frame):
u̇ = F*cos(mu_p)/m - g*cos(theta) - q*w
v̇ = 0
ẇ = -F*sin(mu_p)/m - g*sin(theta) + q*u

Angular Velocity Derivatives:
ṗ = 0
q̇ = -F*l*sin(mu_p)/J_t
ṙ = 0

Euler Angle Derivatives:
φ̇ = 0
θ̇ = q
ψ̇ = 0
"""
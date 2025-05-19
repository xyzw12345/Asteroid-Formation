import sympy

# Define symbols for coordinates and mass
# Source particle (m_j at r_j)
x_j, y_j, z_j, m_j = sympy.symbols('x_j y_j z_j m_j', real=True)
# Field point (where we calculate potential/force, at r_i)
x_i, y_i, z_i = sympy.symbols('x_i y_i z_i', real=True)
# Gravitational constant
G = sympy.Symbol('G', positive=True)

# --- Define vector components ---
# Position vector of the field point (where we feel the force)
R_i_vec = sympy.Matrix([x_i, y_i, z_i])
# Position vector of the source particle j
r_j_vec = sympy.Matrix([x_j, y_j, z_j])

# Vector from source particle j to field point i
R_vec = R_i_vec - r_j_vec # This is r_i - r_j

# Magnitude squared of R_vec
R_sq = R_vec.dot(R_vec) # (x_i-x_j)^2 + (y_i-y_j)^2 + (z_i-z_j)^2
R_mag = sympy.sqrt(R_sq)

# --- Gravitational Potential ---
# Potential at R_i due to m_j at r_j: Phi_j = -G * m_j / |R_i - r_j|
Phi_j = -G * m_j / R_mag
print("Exact Potential Phi_j:")
sympy.pprint(Phi_j)
print("-" * 30)

# --- Multipole Expansion ---
# We expand around the origin (0,0,0) of the source distribution.
# The expansion is valid when |r_j| << |R_i|, i.e., the field point is far from the source distribution.
# Let R_i be the field point (observation point) and r_j be the source point.
# We are expanding 1 / |R_i - r_j|
# Let r = r_j and R = R_i for standard multipole expansion notation.
# So we are expanding 1 / |R - r| where |r| < |R|

# For SymPy's series expansion, it's easier to define symbols for the components of R and r
Rx, Ry, Rz = sympy.symbols('Rx Ry Rz', real=True) # Field point components (large)
sx, sy, sz = sympy.symbols('sx sy sz', real=True) # Source point components (small, for expansion)
# m = sympy.Symbol('m') # Mass of the source particle for the expansion term

# Expression to expand: 1 / sqrt((Rx-sx)^2 + (Ry-sy)^2 + (Rz-sz)^2)
# We will expand this for small sx, sy, sz around sx=0, sy=0, sz=0.
inv_R_minus_r = 1 / sympy.sqrt((Rx - sx)**2 + (Ry - sy)**2 + (Rz - sz)**2)

# Perform Taylor expansion for sx, sy, sz around 0, up to a certain order.
# Let's go up to order 2 in sx, sy, sz (which gives terms up to quadrupole)
# For series expansion in multiple variables, we do it one by one.
# Order n=3 means terms up to x^2, y^2, z^2, xy, xz, yz, etc. (total power <= 2)
# sympy.series(expr, var, x0, n) -> expands around x0 up to power (n-1)
expansion_order = 3 # This will give terms up to power 2 (e.g., sx^2, sx*sy)

print(f"Expanding 1/|R-s| for small s around s=0, up to terms of total power {expansion_order-1}\n")

# Expand in sx
series_sx = sympy.series(inv_R_minus_r, sx, 0, expansion_order)
# Expand the result in sy
series_sx_sy = sympy.series(series_sx.removeO(), sy, 0, expansion_order) # removeO() removes the O(var^n) term
# Expand the result in sz
series_sx_sy_sz = sympy.series(series_sx_sy.removeO(), sz, 0, expansion_order)

# The final expanded form of 1/|R-s|
expanded_inv_dist = series_sx_sy_sz.removeO()

print("Expanded 1/|R-s|:")
sympy.pprint(expanded_inv_dist)
print("-" * 30)

# Now, let's group the terms according to multipole moments.
# R = sqrt(Rx^2 + Ry^2 + Rz^2)
R_magnitude_symbol = sympy.sqrt(Rx**2 + Ry**2 + Rz**2) # This is just 'R'

# --- Monopole Term (0th order in s) ---
# Coefficient of s^0
monopole_term_factor = expanded_inv_dist.coeff(sx, 0).coeff(sy, 0).coeff(sz, 0)
print("Monopole factor (coefficient of s^0):")
sympy.pprint(monopole_term_factor) # Should be 1/R
# Verify:
# sympy.pprint(sympy.simplify(monopole_term_factor - 1/R_magnitude_symbol)) # Should be 0

# Potential for monopole: -G * M_total * (1/R)
# Where M_total = sum(m_j) -> for a single particle, M_total = m
# For a distribution, we would sum m_j * (term_factor_for_m_j)

# --- Dipole Term (1st order in s) ---
# Coefficients of sx, sy, sz
# These terms look like (Rx*sx + Ry*sy + Rz*sz) / R^3
# Or (R_vec . s_vec) / R^3
# Collect all terms linear in sx, sy, or sz
dipole_terms_collected = sympy.collect(expanded_inv_dist, [sx, sy, sz], evaluate=False)
dipole_sx_factor = sympy.collect(expanded_inv_dist, sx).coeff(sx, 1).coeff(sy,0).coeff(sz,0)
dipole_sy_factor = sympy.collect(expanded_inv_dist, sy).coeff(sy, 1).coeff(sx,0).coeff(sz,0)
dipole_sz_factor = sympy.collect(expanded_inv_dist, sz).coeff(sz, 1).coeff(sx,0).coeff(sy,0)

print("\nDipole factor for sx (coefficient of sx * s^0_y * s^0_z):")
sympy.pprint(dipole_sx_factor) # Should be Rx / R^3
print("Dipole factor for sy:")
sympy.pprint(dipole_sy_factor) # Should be Ry / R^3
print("Dipole factor for sz:")
sympy.pprint(dipole_sz_factor) # Should be Rz / R^3

# Dipole moment vector P = sum(m_j * s_j_vec)
# Potential for dipole: -G * (P_vec . R_vec) / R^3
# (Note: some definitions have +G, depending on potential definition)
# Our expansion factor is (R_vec . s_vec) / R^3.
# So potential is -G * m * ( (Rx*sx + Ry*sy + Rz*sz) / R_magnitude_symbol**3 )

# --- Quadrupole Term (2nd order in s) ---
# Coefficients of sx^2, sy^2, sz^2, sx*sy, sx*sz, sy*sz
# These terms look like (sum_{alpha,beta} (3*R_alpha*R_beta - delta_alpha_beta*R^2) * s_alpha*s_beta) / (2*R^5)
# More simply, terms like (3*(Rx*sx)^2 - (s_vec.s_vec)*R^2) / (2*R^5) or (3*Rx*Ry*sx*sy) / R^5 etc.

quad_sxsx_factor = sympy.collect(expanded_inv_dist, sx**2).coeff(sx, 2).coeff(sy,0).coeff(sz,0)
quad_sysy_factor = sympy.collect(expanded_inv_dist, sy**2).coeff(sy, 2).coeff(sx,0).coeff(sz,0)
quad_szsz_factor = sympy.collect(expanded_inv_dist, sz**2).coeff(sz, 2).coeff(sx,0).coeff(sy,0)

quad_sxsy_factor = sympy.collect(sympy.collect(expanded_inv_dist, sx*sy).coeff(sx*sy,1), [sx,sy,sz], evaluate=False).coeff(sx,0).coeff(sy,0).coeff(sz,0)
# Simpler way for mixed terms:
# Extract terms with sx, then from those extract terms with sy
term_with_sx = sympy.S(0)
for term in sympy.Add.make_args(expanded_inv_dist):
    if term.has(sx) and not term.has(sx**2) and not term.has(sx**3): # Linear in sx
        term_with_sx += term
term_with_sx_sy = sympy.S(0)
for term in sympy.Add.make_args(term_with_sx):
     if term.has(sy) and not term.has(sy**2) and not term.has(sy**3): # Linear in sy
        term_with_sx_sy += term
quad_sxsy_factor_alt = sympy.collect(term_with_sx_sy, sx*sy).coeff(sx*sy,1)


print("\nQuadrupole factor for sx*sx (coefficient of sx^2 * s^0_y * s^0_z):")
sympy.pprint(quad_sxsx_factor) # Should be (3*Rx^2 - R^2) / (2*R^5)
print("Quadrupole factor for sy*sy:")
sympy.pprint(quad_sysy_factor) # Should be (3*Ry^2 - R^2) / (2*R^5)
print("Quadrupole factor for sz*sz:")
sympy.pprint(quad_szsz_factor) # Should be (3*Rz^2 - R^2) / (2*R^5)

print("Quadrupole factor for sx*sy (coefficient of sx*sy * s^0_z):")
sympy.pprint(quad_sxsy_factor_alt) # Should be (3*Rx*Ry) / R^5

# Quadrupole tensor Q_ab = sum(m_j * (3*s_ja*s_jb - delta_ab * |s_j|^2))
# Potential for quadrupole: -G * (1/2) * sum_{a,b} Q_ab * (R_a * R_b / R^5) (simplified, various forms exist)
# Or more directly from expansion: -G * m * (sum_{a,b} s_a s_b * ( (3 R_a R_b - delta_ab R^2) / (2 R^5) ) )
# The factors we extracted are ( (3 R_a R_b - delta_ab R^2) / (2 R^5) ) for diagonal terms like sx*sx
# and (3 R_a R_b) / R^5 for off-diagonal terms like sx*sy.

print("-" * 30)
# --- Constructing the Potential Expansion for a single source particle 'm' at (sx, sy, sz) ---
# (and field point R = (Rx, Ry, Rz))
# Potential Phi = -G * m * [ (1/R) +
#                             (Rx*sx + Ry*sy + Rz*sz)/R^3 +
#                             (1/2) * ( sx^2*(3Rx^2-R^2)/R^5 + sy^2*(3Ry^2-R^2)/R^5 + sz^2*(3Rz^2-R^2)/R^5 +
#                                       2*sx*sy*(3RxRy)/R^5 + 2*sx*sz*(3RxRz)/R^5 + 2*sy*sz*(3RyRz)/R^5 ) + ... ]
# Note: The factor of 1/2 for quadrupole terms and factor of 2 for mixed terms come from the Taylor expansion
# (e.g. d^2f/dxdy is coeff of dx*dy, but Taylor is sum (1/k!) d^k f * (dx)^k )
# The factors sympy gives directly are the coefficients of s_alpha * s_beta in the expansion of 1/|R-s|.

# Let's reconstruct the potential using the extracted factors (manually for clarity)
# This assumes R_magnitude_symbol is R = |(Rx,Ry,Rz)|
R = R_magnitude_symbol
Phi_expanded_manual = -G * m_j * (
    monopole_term_factor + # Term for s^0
    dipole_sx_factor * sx + dipole_sy_factor * sy + dipole_sz_factor * sz + # Linear terms
    quad_sxsx_factor * sx**2 + quad_sysy_factor * sy**2 + quad_szsz_factor * sz**2 + # s_alpha^2 terms
    quad_sxsy_factor_alt * sx * sy + # sx*sy term (need similar for sx*sz, sy*sz)
    # To get other mixed terms properly:
    # (sympy.series((sympy.series(inv_R_minus_r, sx, 0, 3)).removeO(), sy, 0, 3).removeO()).coeff(sz,1).coeff(sx,1)
    # This gets tedious quickly. The 'expanded_inv_dist' already has all these.
    sympy.S(0) # Placeholder for more quadrupole terms if done manually
)
# A better way is to use the full 'expanded_inv_dist' directly:
Phi_expanded_sympy = -G * m_j * expanded_inv_dist

print("Reconstructed Potential from Sympy's full expansion (up to order 2 in s):")
sympy.pprint(Phi_expanded_sympy)
print("-" * 30)

# --- Gravitational Force F = -nabla_R (Phi) ---
# Where nabla_R is the gradient with respect to the field point coordinates (Rx, Ry, Rz)
# We differentiate Phi_expanded_sympy

Fx_expanded = -sympy.diff(Phi_expanded_sympy, Rx)
Fy_expanded = -sympy.diff(Phi_expanded_sympy, Ry)
Fz_expanded = -sympy.diff(Phi_expanded_sympy, Rz)

print("Fx (x-component of Force) from expanded potential:")
sympy.pprint(Fx_expanded)
# print("\nFy from expanded potential:")
# sympy.pprint(Fy_expanded)
# print("\nFz from expanded potential:")
# sympy.pprint(Fz_expanded)
print("-" * 30)

# --- Extracting Force Orders ---
# To get the force for a distribution of masses m_k at s_k, you would sum:
# F_total_x = sum_k (-G * m_k * diff(expanded_inv_dist_with_s_k, Rx))

# Monopole Force (0th order in s): Differentiate the monopole potential term
Phi_monopole_part = -G * m_j * monopole_term_factor # -G * m_j * (1/R)
Fx_monopole = -sympy.diff(Phi_monopole_part, Rx)
print("Fx Monopole part:")
sympy.pprint(Fx_monopole) # Should be -G * m_j * Rx / R^3 (attractive towards origin if m_j is at origin)
                          # Wait, if m_j is at (sx,sy,sz) and M_total = m_j,
                          # and we are expanding around the CoM (which is (sx,sy,sz) for a single particle),
                          # the monopole force should be -G * m_j * Rx / R^3 if s=0 (source at origin).
                          # The expansion assumes s is small deviation from origin.
                          # If we consider M_total = m_j at the origin (sx=sy=sz=0),
                          # then monopole potential is -G*m_j/R. Force is -G*m_j*Rx/R^3.
                          # The 'monopole_term_factor' is 1/R from the expansion of 1/|R-s|.
                          # So force is -G * m_j * d/dRx (1/R) = G * m_j * Rx / R^3

# Dipole Force (1st order in s): Differentiate the dipole potential term
Phi_dipole_part = -G * m_j * (dipole_sx_factor * sx + dipole_sy_factor * sy + dipole_sz_factor * sz)
Fx_dipole = -sympy.diff(Phi_dipole_part, Rx)
print("\nFx Dipole part (linear in sx, sy, sz):")
sympy.pprint(Fx_dipole)
# This will give terms like G*m_j * ( (sx/R^3) - 3*Rx*(Rx*sx + Ry*sy + Rz*sz)/R^5 )

# Quadrupole Force (2nd order in s): Differentiate the quadrupole potential term
# This gets lengthy very quickly.
# The general approach is:
# 1. Write down the expanded potential Phi(R, s) up to the desired order in s.
# 2. Group terms: Phi = Phi_mono(R) + Phi_dipole(R, s) + Phi_quad(R, s) + ...
#    where Phi_mono contains terms of s^0, Phi_dipole contains terms linear in s, Phi_quad quadratic in s.
# 3. Force F = -nabla_R Phi
#    F_mono = -nabla_R Phi_mono
#    F_dipole = -nabla_R Phi_dipole
#    F_quad = -nabla_R Phi_quad

# Example: Full Fx up to 2nd order in s (from differentiating Phi_expanded_sympy)
# We already have Fx_expanded. It contains all orders.
# To get ONLY the monopole part of the force: set sx=0, sy=0, sz=0 in Fx_expanded
Fx_expanded_mono_only = Fx_expanded.subs([(sx,0), (sy,0), (sz,0)])
print("\nFx from full expansion, then s=0 (Monopole force component):")
sympy.pprint(Fx_expanded_mono_only) # This is -G*m_j * d/dRx (1/R) = G*m_j*Rx / (Rx**2+Ry**2+Rz**2)**(3/2)

# To get ONLY the dipole part of the force:
# These are terms in Fx_expanded that are linear in sx, sy, or sz.
Fx_expanded_dipole_part = sympy.S(0)
# Iterate through terms of Fx_expanded and collect linear ones
# This is tricky with sympy. A better way is to differentiate the dipole part of potential.
# We already did this with Fx_dipole.

print("\nNote: For a *distribution* of source masses m_k at positions s_k:")
print("1. Monopole: M_total = sum(m_k). Potential uses M_total / R. Force uses M_total * R_vec / R^3.")
print("2. Dipole: P_vec = sum(m_k * s_k_vec). Potential uses (P_vec . R_vec) / R^3.")
print("3. Quadrupole: Q_ab_tensor = sum(m_k * (3*s_ka*s_kb - delta_ab * |s_k|^2)). Potential involves Q_ab * R_a*R_b / R^5 terms.")
print("The SymPy expansion gives factors for a single source m_j at s_j. To get total potential/force, you sum these contributions or use the defined multipole moments.")

# Example: if we define the total mass M = m_j, and dipole moment Px = m_j*sx, Py = m_j*sy, Pz = m_j*sz
# Then the potential due to dipole is G * (Rx*Px + Ry*Py + Rz*Pz) / R^3 (if potential is +G * P.R / R^3)
# Our expansion yielded -G*m_j * (Rx*sx + Ry*sy + Rz*sz)/R^3 for the dipole part of a single particle.
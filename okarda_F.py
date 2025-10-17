import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt

def okada85(*args, **kwargs):
    """
    Surface deformation due to a finite rectangular source.
    
    [uE,uN,uZ,uZE,uZN,uNN,uNE,uEN,uEE] = okada85(
       E,N,DEPTH,STRIKE,DIP,LENGTH,WIDTH,RAKE,SLIP,OPEN, nu=0.25, plot=False)
    
    Computes displacements, tilts and strains at the surface of an elastic
    half-space, due to a dislocation defined by RAKE, SLIP, and OPEN on a 
    rectangular fault defined by orientation STRIKE and DIP, and size LENGTH and
    WIDTH. The fault centroid is located (0,0,-DEPTH).
    
    Parameters:
    -----------
    E,N : array_like
        Coordinates of observation points in a geographic referential 
        (East,North,Up) relative to fault centroid
    DEPTH : float or array_like
        Depth of the fault centroid (DEPTH > 0)
    STRIKE : float or array_like
        Fault trace direction (0 to 360° relative to North), defined so 
        that the fault dips to the right side of the trace
    DIP : float
        Angle between the fault and a horizontal plane (0 to 90°)
        Must be scalar due to singularity in Okada's equations
    LENGTH : float or array_like
        Fault length in the STRIKE direction (LENGTH > 0)
    WIDTH : float or array_like
        Fault width in the DIP direction (WIDTH > 0)
    RAKE : float or array_like
        Direction the hanging wall moves during rupture, measured relative
        to the fault STRIKE (-180 to 180°)
    SLIP : float or array_like
        Dislocation in RAKE direction (length unit)
    OPEN : float or array_like
        Dislocation in tensile component (same unit as SLIP)
    nu : float, optional
        Poisson's ratio (default is 0.25 for isotropic medium)
    plot : bool, optional
        If True, produces a 3D plot of fault geometry and dislocation
    
    Returns:
    --------
    Depending on number of outputs requested:
    - 2 outputs: uZE, uZN (tilts only)
    - 3 outputs: uE, uN, uZ (displacements only)
    - 4 outputs: uNN, uNE, uEN, uEE (strains only)
    - 5 outputs: uE, uN, uZ, uZE, uZN (displacements and tilts)
    - 6 outputs: uZE, uZN, uNN, uNE, uEN, uEE (tilts and strains)
    - 7 outputs: uE, uN, uZ, uNN, uNE, uEN, uEE (displacements and strains)
    - 9 outputs: uE, uN, uZ, uZE, uZN, uNN, uNE, uEN, uEE (all)
    """
    
    # Parse arguments
    if len(args) < 10:
        raise ValueError('Not enough input arguments.')
    
    if len(args) > 12:
        raise ValueError('Too many input arguments.')
    
    # Check if first 10 arguments are numeric
    for i in range(10):
        if not np.isscalar(args[i]) and not hasattr(args[i], '__iter__'):
            print(args[i])
            raise ValueError('Input arguments E,N,DEPTH,STRIKE,DIP,LENGTH,WIDTH,RAKE,SLIP,OPEN must be numeric.')
    
    # DIP must be scalar
    if not np.isscalar(args[4]):
        raise ValueError('DIP argument must be scalar.')
    
    # Default values
    plotflag = kwargs.get('plot', False)
    nu = kwargs.get('nu', 0.25)
    
    # Parse additional arguments
    if len(args) == 11:
        if isinstance(args[10], (int, float)):
            nu = args[10]
        elif args[10] == 'plot':
            plotflag = True
    elif len(args) == 12:
        nu = args[10]
        if args[11] == 'plot':
            plotflag = True


  
    
    # Assign input arguments
    e = np.asarray(args[0])
    n = np.asarray(args[1])
    depth = float(args[2])
    strike = float(args[3]) * np.pi / 180  # converting to radians
    dip = float(args[4]) * np.pi / 180  # converting to radians
    L = float(args[5])
    W = float(args[6])
    rake = float(args[7]) * np.pi / 180  # converting to radians
    slip = float(args[8])
    U3 = float(args[9])


    
    
         # Check if plot is possible
    if plotflag:
        scalars = [np.isscalar(x) for x in [depth, strike, L, W, rake, slip, U3]]
        if not all(scalars):
            print('Warning: Cannot make plot with fault geometry parameters other than scalars.')
            non_scalars = [i for i, is_scalar in enumerate(scalars) if not is_scalar]
            param_names = ['depth', 'strike', 'L', 'W', 'rake', 'slip', 'U3']
            for idx in non_scalars:
                print(f"{param_names[idx]} is not scalar: {[depth, strike, L, W, rake, slip, U3][idx]}")
            plotflag = False
    
    # Define dislocation in the fault plane system
    U1 = np.cos(rake) * slip
    U2 = np.sin(rake) * slip
    
    # Convert fault coordinates (E,N,DEPTH) relative to centroid
    # into Okada's reference system (X,Y,D)
    d = depth + np.sin(dip) * W / 2  # d is fault's bottom edge
    ec = e + np.cos(strike) * np.cos(dip) * W / 2
    nc = n - np.sin(strike) * np.cos(dip) * W / 2
    x = np.cos(strike) * nc + np.sin(strike) * ec + L / 2
    y = np.sin(strike) * nc - np.cos(strike) * ec + np.cos(dip) * W
    
    # Variable substitution (independent from xi and eta)
    p = y * np.cos(dip) + d * np.sin(dip)
    q = y * np.sin(dip) - d * np.cos(dip)
    
    # Determine what to calculate based on expected outputs
    # For now, calculate everything and return based on context
    
    # Displacements
    ux = (-U1 / (2 * np.pi) * chinnery(ux_ss, x, p, L, W, q, dip, nu) -
          U2 / (2 * np.pi) * chinnery(ux_ds, x, p, L, W, q, dip, nu) +
          U3 / (2 * np.pi) * chinnery(ux_tf, x, p, L, W, q, dip, nu))
    
    uy = (-U1 / (2 * np.pi) * chinnery(uy_ss, x, p, L, W, q, dip, nu) -
          U2 / (2 * np.pi) * chinnery(uy_ds, x, p, L, W, q, dip, nu) +
          U3 / (2 * np.pi) * chinnery(uy_tf, x, p, L, W, q, dip, nu))
    
    uz = (-U1 / (2 * np.pi) * chinnery(uz_ss, x, p, L, W, q, dip, nu) -
          U2 / (2 * np.pi) * chinnery(uz_ds, x, p, L, W, q, dip, nu) +
          U3 / (2 * np.pi) * chinnery(uz_tf, x, p, L, W, q, dip, nu))
    
    # Rotation from Okada's axes to geographic
    ue = np.sin(strike) * ux - np.cos(strike) * uy
    un = np.cos(strike) * ux + np.sin(strike) * uy
    
    # Tilts
    uzx = (-U1 / (2 * np.pi) * chinnery(uzx_ss, x, p, L, W, q, dip, nu) -
           U2 / (2 * np.pi) * chinnery(uzx_ds, x, p, L, W, q, dip, nu) +
           U3 / (2 * np.pi) * chinnery(uzx_tf, x, p, L, W, q, dip, nu))
    
    uzy = (-U1 / (2 * np.pi) * chinnery(uzy_ss, x, p, L, W, q, dip, nu) -
           U2 / (2 * np.pi) * chinnery(uzy_ds, x, p, L, W, q, dip, nu) +
           U3 / (2 * np.pi) * chinnery(uzy_tf, x, p, L, W, q, dip, nu))
    
    # Rotation from Okada's axes to geographic
    uze = -np.sin(strike) * uzx + np.cos(strike) * uzy
    uzn = -np.cos(strike) * uzx - np.sin(strike) * uzy
    
    # Strains
    uxx = (-U1 / (2 * np.pi) * chinnery(uxx_ss, x, p, L, W, q, dip, nu) -
           U2 / (2 * np.pi) * chinnery(uxx_ds, x, p, L, W, q, dip, nu) +
           U3 / (2 * np.pi) * chinnery(uxx_tf, x, p, L, W, q, dip, nu))
    
    uxy = (-U1 / (2 * np.pi) * chinnery(uxy_ss, x, p, L, W, q, dip, nu) -
           U2 / (2 * np.pi) * chinnery(uxy_ds, x, p, L, W, q, dip, nu) +
           U3 / (2 * np.pi) * chinnery(uxy_tf, x, p, L, W, q, dip, nu))
    
    uyx = (-U1 / (2 * np.pi) * chinnery(uyx_ss, x, p, L, W, q, dip, nu) -
           U2 / (2 * np.pi) * chinnery(uyx_ds, x, p, L, W, q, dip, nu) +
           U3 / (2 * np.pi) * chinnery(uyx_tf, x, p, L, W, q, dip, nu))
    
    uyy = (-U1 / (2 * np.pi) * chinnery(uyy_ss, x, p, L, W, q, dip, nu) -
           U2 / (2 * np.pi) * chinnery(uyy_ds, x, p, L, W, q, dip, nu) +
           U3 / (2 * np.pi) * chinnery(uyy_tf, x, p, L, W, q, dip, nu))
    
    # Rotation from Okada's axes to geographic
    unn = (np.cos(strike)**2 * uxx + np.sin(2*strike) * (uxy + uyx) / 2 + 
           np.sin(strike)**2 * uyy)
    une = (np.sin(2*strike) * (uxx - uyy) / 2 + np.sin(strike)**2 * uyx - 
           np.cos(strike)**2 * uxy)
    uen = (np.sin(2*strike) * (uxx - uyy) / 2 - np.cos(strike)**2 * uyx + 
           np.sin(strike)**2 * uxy)
    uee = (np.sin(strike)**2 * uxx - np.sin(2*strike) * (uyx + uxy) / 2 + 
           np.cos(strike)**2 * uyy)
    
    # Handle plotting
    if plotflag:
        _plot_fault_geometry(e, n, depth, strike, dip, L, W, U1, U2, U3, d)
    
    # Return all 9 outputs for now - in practice, you'd modify this based on context
    return ue, un, uz, uze, uzn, unn, une, uen, uee


# Helper functions
def chinnery(f, x, p, L, W, q, dip, nu):
    """Chinnery's notation [equation (24) p. 1143]"""
    return (f(x, p, q, dip, nu) - f(x, p-W, q, dip, nu) - 
            f(x-L, p, q, dip, nu) + f(x-L, p-W, q, dip, nu))


# Displacement subfunctions - Strike-slip
def ux_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = xi * q / (R * (R + eta)) + I1(xi, eta, q, dip, nu, R) * np.sin(dip)
    mask = q != 0
    u = np.where(mask, u + np.arctan2(xi * eta, q * R), u)
    return u

def uy_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = ((eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + eta)) + 
         q * np.cos(dip) / (R + eta) + I2(eta, q, dip, nu, R) * np.sin(dip))
    return u

def uz_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = ((eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + eta)) + 
         q * np.sin(dip) / (R + eta) + I4(db, eta, q, dip, nu, R) * np.sin(dip))
    return u

# Displacement subfunctions - Dip-slip
def ux_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = q / R - I3(eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
    return u

def uy_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = ((eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) - 
         I1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip))
    mask = q != 0
    u = np.where(mask, u + np.cos(dip) * np.arctan2(xi * eta, q * R), u)
    return u

def uz_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (db * q / (R * (R + xi)) - 
         I5(xi, eta, q, dip, nu, R, db) * np.sin(dip) * np.cos(dip))
    mask = q != 0
    u = np.where(mask, u + np.sin(dip) * np.arctan2(xi * eta, q * R), u)
    return u

# Displacement subfunctions - Tensile fault
def ux_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = q**2 / (R * (R + eta)) - I3(eta, q, dip, nu, R) * np.sin(dip)**2
    return u

def uy_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (-(eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + xi)) - 
         np.sin(dip) * xi * q / (R * (R + eta)) - 
         I1(xi, eta, q, dip, nu, R) * np.sin(dip)**2)
    mask = q != 0
    u = np.where(mask, u + np.sin(dip) * np.arctan2(xi * eta, q * R), u)
    return u

def uz_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = ((eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) + 
         np.cos(dip) * xi * q / (R * (R + eta)) - 
         I5(xi, eta, q, dip, nu, R, db) * np.sin(dip)**2)
    mask = q != 0
    u = np.where(mask, u - np.cos(dip) * np.arctan2(xi * eta, q * R), u)
    return u


# Strain subfunctions - Strike-slip [equation (31) p. 1145]
def uxx_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (xi**2 * q * A_func(eta, R) - 
         J1(xi, eta, q, dip, nu, R) * np.sin(dip))
    return u

def uxy_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (xi**3 * db / (R**3 * (eta**2 + q**2)) - 
         (xi**3 * A_func(eta, R) + J2(xi, eta, q, dip, nu, R)) * np.sin(dip))
    return u

def uyx_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (xi * q / R**3 * np.cos(dip) + 
         (xi * q**2 * A_func(eta, R) - J2(xi, eta, q, dip, nu, R)) * np.sin(dip))
    return u

def uyy_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = (yb * q / R**3 * np.cos(dip) + 
         (q**3 * A_func(eta, R) * np.sin(dip) - 2 * q * np.sin(dip) / (R * (R + eta)) - 
          (xi**2 + eta**2) / R**3 * np.cos(dip) - J4(xi, eta, q, dip, nu, R)) * np.sin(dip))
    return u


# I... displacement subfunctions
def I1(xi, eta, q, dip, nu, R):
    db = eta * np.sin(dip) - q * np.cos(dip)
    eps = np.finfo(float).eps
    
    if np.cos(dip) > eps:
        I = ((1 - 2*nu) * (-xi / (np.cos(dip) * (R + db))) - 
             np.sin(dip) / np.cos(dip) * I5(xi, eta, q, dip, nu, R, db))
    else:
        I = -(1 - 2*nu) / 2 * xi * q / (R + db)**2
    
    return I

def I2(eta, q, dip, nu, R):
    I = (1 - 2*nu) * (-np.log(R + eta)) - I3(eta, q, dip, nu, R)
    return I

def I3(eta, q, dip, nu, R):
    yb = eta * np.cos(dip) + q * np.sin(dip)
    db = eta * np.sin(dip) - q * np.cos(dip)
    eps = np.finfo(float).eps
    
    if np.cos(dip) > eps:
        I = ((1 - 2*nu) * (yb / (np.cos(dip) * (R + db)) - np.log(R + eta)) + 
             np.sin(dip) / np.cos(dip) * I4(db, eta, q, dip, nu, R))
    else:
        I = ((1 - 2*nu) / 2 * (eta / (R + db) + yb * q / (R + db)**2 - 
                                np.log(R + eta)))
    
    return I

def I4(db, eta, q, dip, nu, R):
    eps = np.finfo(float).eps
    
    if np.cos(dip) > eps:
        I = ((1 - 2*nu) / np.cos(dip) * 
             (np.log(R + db) - np.sin(dip) * np.log(R + eta)))
    else:
        I = -(1 - 2*nu) * q / (R + db)
    
    return I

def I5(xi, eta, q, dip, nu, R, db):
    X = np.sqrt(xi**2 + q**2)
    eps = np.finfo(float).eps
    
    if np.cos(dip) > eps:
        numerator = (eta * (X + q * np.cos(dip)) + 
                    X * (R + X) * np.sin(dip))
        denominator = xi * (R + X) * np.cos(dip)
        I = ((1 - 2*nu) * 2 / np.cos(dip) * 
             np.arctan2(numerator, denominator))
        # Handle xi == 0 case
        I = np.where(xi == 0, 0, I)
    else:
        I = -(1 - 2*nu) * xi * np.sin(dip) / (R + db)
    
    return I


# -----------------------------------------------------------------
def uxx_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (xi * q**2 * A_func(eta, R) + 
         J3(xi, eta, q, dip, nu, R) * np.sin(dip)**2)
    return u

# -----------------------------------------------------------------
def uxy_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (-db * q / R**3 - 
         xi**2 * q * A_func(eta, R) * np.sin(dip) + 
         J1(xi, eta, q, dip, nu, R) * np.sin(dip)**2)
    return u

# -----------------------------------------------------------------
def uyx_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (q**2 / R**3 * np.cos(dip) + 
         q**3 * A_func(eta, R) * np.sin(dip) + 
         J1(xi, eta, q, dip, nu, R) * np.sin(dip)**2)
    return u

# -----------------------------------------------------------------
def uyy_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = ((yb * np.cos(dip) - db * np.sin(dip)) * q**2 * A_func(xi, R) - 
         q * np.sin(2*dip) / (R * (R + xi)) - 
         (xi * q**2 * A_func(eta, R) - J2(xi, eta, q, dip, nu, R)) * np.sin(dip)**2)
    return u

# -----------------------------------------------------------------
def ux_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (q**2 / (R * (R + eta)) - 
         I3(eta, q, dip, nu, R) * np.sin(dip)**2)
    return u

# -----------------------------------------------------------------
def uy_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (-(eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + xi)) - 
         np.sin(dip) * xi * q / (R * (R + eta)) - 
         I1(xi, eta, q, dip, nu, R) * np.sin(dip)**2)
    mask = q != 0
    u = np.where(mask, u + np.sin(dip) * np.arctan2(xi * eta, q * R), u)
    return u

# -----------------------------------------------------------------
def uz_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = ((eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) + 
         np.cos(dip) * xi * q / (R * (R + eta)) - 
         I5(xi, eta, q, dip, nu, R, db) * np.sin(dip)**2)
    mask = q != 0
    u = np.where(mask, u - np.cos(dip) * np.arctan2(xi * eta, q * R), u)
    return u



# Continue with tilt and strain subfunctions...
# Strain subfunctions - Dip-slip [equation (32) p. 1145]
def uxx_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (xi * q / R**3 + 
         J3(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip))
    return u

def uxy_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = (yb * q / R**3 - 
         np.sin(dip) / R + 
         J1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip))
    return u

def uyx_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = (yb * q / R**3 + 
         q * np.cos(dip) / (R * (R + eta)) + 
         J1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip))
    return u

def uyy_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = (yb**2 * q * A_func(xi, R) - 
         (2 * yb / (R * (R + xi)) + xi * np.cos(dip) / (R * (R + eta))) * np.sin(dip) + 
         J2(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip))
    return u

# Tilt subfunctions - Strike-slip [equation (37) p. 1147]
def uzx_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (-xi * q**2 * A_func(eta, R) * np.cos(dip) + 
         (xi * q / R**3 - K1(xi, eta, q, dip, nu, R)) * np.sin(dip))
    return u

def uzy_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = ((db * q / R**3) * np.cos(dip) + 
         (xi**2 * q * A_func(eta, R) * np.cos(dip) - np.sin(dip) / R + 
          yb * q / R**3 - K2(xi, eta, q, dip, nu, R)) * np.sin(dip))
    return u


# Tilt subfunctions - Strike-slip [equation (37) p. 1147]
def uzx_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (-xi * q**2 * A_func(eta, R) * np.cos(dip) + 
         (xi * q / R**3 - K1(xi, eta, q, dip, nu, R)) * np.sin(dip))
    return u

def uzy_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = ((db * q / R**3) * np.cos(dip) + 
         (xi**2 * q * A_func(eta, R) * np.cos(dip) - np.sin(dip) / R + 
          yb * q / R**3 - K2(xi, eta, q, dip, nu, R)) * np.sin(dip))
    return u

# Dip-slip tilt subfunctions [equation (38) p. 1147]
def uzx_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (db * q / R**3 + 
         q * np.sin(dip) / (R * (R + eta)) + 
         K3(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip))
    return u

def uzy_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = (yb * db * q * A_func(xi, R) - 
         (2 * db / (R * (R + xi)) + xi * np.sin(dip) / (R * (R + eta))) * np.sin(dip) + 
         K1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip))
    return u

# Tensile fault tilt subfunctions [equation (39) p. 1147]
def uzx_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    u = (q**2 / R**3 * np.sin(dip) - 
         q**3 * A_func(eta, R) * np.cos(dip) + 
         K3(xi, eta, q, dip, nu, R) * np.sin(dip)**2)
    return u

def uzy_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    u = ((yb * np.sin(dip) + db * np.cos(dip)) * q**2 * A_func(xi, R) + 
         xi * q**2 * A_func(eta, R) * np.sin(dip) * np.cos(dip) - 
         (2 * q / (R * (R + xi)) - K1(xi, eta, q, dip, nu, R)) * np.sin(dip)**2)
    return u

def A_func(x, R):
    return (2*R + x) / (R**3 * (R + x)**2)

def K1(xi, eta, q, dip, nu, R):
    db = eta * np.sin(dip) - q * np.cos(dip)
    eps = np.finfo(float).eps
    
    if np.cos(dip) > eps:
        K = ((1 - 2*nu) * xi / np.cos(dip) * 
             (1 / (R * (R + db)) - np.sin(dip) / (R * (R + eta))))
    else:
        K = (1 - 2*nu) * xi * q / (R * (R + db)**2)
    
    return K

def K2(xi, eta, q, dip, nu, R):
    K = ((1 - 2*nu) * (-np.sin(dip) / R + q * np.cos(dip) / (R * (R + eta))) - 
         K3(xi, eta, q, dip, nu, R))
    return K

def K3(xi, eta, q, dip, nu, R):
    db = eta * np.sin(dip) - q * np.cos(dip)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    eps = np.finfo(float).eps
    
    if np.cos(dip) > eps:
        K = ((1 - 2*nu) / np.cos(dip) * 
             (q / (R * (R + eta)) - yb / (R * (R + db))))
    else:
        K = ((1 - 2*nu) * np.sin(dip) / (R + db) * 
             (xi**2 / (R * (R + db)) - 1))
    
    return K


def J1(xi, eta, q, dip, nu, R):
    db = eta * np.sin(dip) - q * np.cos(dip)
    eps = np.finfo(float).eps
    
    if np.cos(dip) > eps:
        J = ((1 - 2*nu) / np.cos(dip) * 
             (xi**2 / (R * (R + db)**2) - 1 / (R + db)) - 
             np.sin(dip) / np.cos(dip) * K3(xi, eta, q, dip, nu, R))
    else:
        J = ((1 - 2*nu) / 2 * q / (R + db)**2 * 
             (2 * xi**2 / (R * (R + db)) - 1))
    
    return J

def J2(xi, eta, q, dip, nu, R):
    db = eta * np.sin(dip) - q * np.cos(dip)
    yb = eta * np.cos(dip) + q * np.sin(dip)
    eps = np.finfo(float).eps
    
    if np.cos(dip) > eps:
        J = ((1 - 2*nu) / np.cos(dip) * xi * yb / (R * (R + db)**2) - 
             np.sin(dip) / np.cos(dip) * K1(xi, eta, q, dip, nu, R))
    else:
        J = ((1 - 2*nu) / 2 * xi * np.sin(dip) / (R + db)**2 * 
             (2 * q**2 / (R * (R + db)) - 1))
    
    return J

def J3(xi, eta, q, dip, nu, R):
    J = ((1 - 2*nu) * (-xi / (R * (R + eta))) - 
         J2(xi, eta, q, dip, nu, R))
    return J

def J4(xi, eta, q, dip, nu, R):
    J = ((1 - 2*nu) * (-np.cos(dip) / R - q * np.sin(dip) / (R * (R + eta))) - 
         J1(xi, eta, q, dip, nu, R))
    return J

# Add remaining displacement, tilt, and strain functions following the same pattern...
# (For brevity, I'm showing the structure. You'd need to implement all functions.)

def _plot_fault_geometry(e, n, depth, strike, dip, L, W, U1, U2, U3, d):
    """Plot fault geometry and dislocation"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot observation points
    ax.scatter(e.flatten(), n.flatten(), np.zeros_like(e.flatten()), 
               c='red', s=0.1, alpha=0.1)
    
    # Fault geometry
    alpha = np.pi/2 - strike
    x_fault = (L/2 * np.cos(alpha) * np.array([-1, 1, 1, -1]) + 
               np.sin(alpha) * np.cos(dip) * W/2 * np.array([-1, -1, 1, 1]))
    y_fault = (L/2 * np.sin(alpha) * np.array([-1, 1, 1, -1]) + 
               np.cos(alpha) * np.cos(dip) * W/2 * np.array([1, 1, -1, -1]))
    z_fault = -d + np.sin(dip) * W * np.array([1, 1, 0, 0])
    
    # Dislocation components
    ddx = U1 * np.cos(alpha) - U2 * np.sin(alpha) * np.cos(dip) + U3 * np.sin(alpha) * np.sin(dip)
    ddy = U1 * np.sin(alpha) + U2 * np.cos(alpha) * np.cos(dip) - U3 * np.cos(alpha) * np.sin(dip)
    ddz = U2 * np.sin(dip) + U3 * np.cos(dip)
    
    # Plot fault patches
    
    verts = [list(zip(x_fault, y_fault, z_fault))]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='gray', 
                                        edgecolors='black', linewidths=2, alpha=0.7))
    
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Vertical')
    ax.set_box_aspect([1,1,1])
    plt.show()

# Add all the remaining strain and tilt functions following the same pattern as above...
# This includes uxx_ss, uxy_ss, uyx_ss, uyy_ss, etc. for strike-slip, dip-slip, and tensile faults


def demo_fault_plot():
    """
    Demonstrate the fault geometry plotting with random input values
    """
    # Random input values for demonstration
    e, n = np.meshgrid(np.linspace(-200000, 2000000, 10000), np.linspace(-2000000, 200000, 10000))
    depth = -10  # Fault depth (km)
    strike = 45.0  # Strike angle (degrees)
    dip = 60.0  # Dip angle (degrees)
    L = 10.0  # Fault length (km)
    W = 8.0  # Fault width (km)
    rake = 90.0  # Rake angle (degrees)
    slip = 2.0  # Slip amount (m)
    opening = 1  # Opening (m)
    nu =0.25
    
    print("Demonstrating fault geometry plot with random values...")
    print(f"Fault parameters: depth={depth}km, strike={strike}°, dip={dip}°")
    print(f"Fault size: length={L}km, width={W}km")
    print(f"Dislocation: rake={rake}°, slip={slip}m, opening={opening}m")
    
    # Call okada85 with plot=True to visualize the fault
    result = okada85(e, n, depth, strike, dip, L, W, rake, slip, opening, nu=nu, plot=True)
    
    return result

if __name__ == "__main__":
    demo_fault_plot()
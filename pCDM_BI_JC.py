import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.interpolate import griddata
import pCDM_fast as pCDM_fast

import matplotlib.pyplot as plt
import os 
import pandas as pd
import llh2local as llh
import local2llh as l2llh

#### In this current version posative is towards the satellite ####
#### This follows GBIS conventions #####

"""
Bayesian inference for pCDM source parameters using MCMC with spatially correlated noise.
    
Author: John Condon
Date of edit: December 2024
"""

def convert_lat_long_2_xy(lat, lon, lat0, lon0):
    ll = [lon.flatten(), lat.flatten()]
    ll = np.array(ll, dtype=float)
    xy = llh.llh2local(ll, np.array([lon0, lat0], dtype=float))
    x = xy[0,:].reshape(lat.shape)
    y = xy[1,:].reshape(lat.shape)
    return xy

def simulated_annealing_optimization(u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                                   C_inv, C_logdet, los_e, los_n, los_u,starting_params=None,
                                   SA_iterations=10000, initial_temp=10.0, 
                                   cooling_rate=0.95, min_temp=0.01,
                                   step_sizes=None, bounds=None):
    """
    Use simulated annealing to find good initial parameter estimates.
    
    Parameters:
    -----------
    u_los_obs : array
        Observed LOS displacements
    X_obs, Y_obs : array
        Observation coordinates
    incidence_angle, heading : float
        Satellite geometry parameters
    C_inv : array
        Inverse of noise covariance matrix
    C_logdet : float
        Log determinant of noise covariance matrix
    los_e, los_n, los_u : float
        Line-of-sight unit vector components
    n_iterations : int
        Number of SA iterations
    initial_temp : float
        Initial temperature
    cooling_rate : float
        Temperature cooling rate (0 < rate < 1)
    min_temp : float
        Minimum temperature (stopping criterion)
    step_sizes : dict
        Step sizes for each parameter
    bounds : dict
        Parameter bounds
        
    Returns:
    --------
    best_params : dict
        Best parameter values found
    best_energy : float
        Best energy (negative log likelihood)
    energy_trace : list
        Energy evolution during SA
    temperature_trace : list
        Temperature evolution during SA
    """
    

    
    def energy_function(params):
        """Calculate energy (negative log likelihood)"""
        try:
            # Check bounds first
            for key, (lower, upper) in bounds.items():
                if not (lower <= params[key] <= upper):
                    return np.inf
                
            dvs = [params['DVx'], params['DVy'], params['DVz']]
            signs = [np.sign(dv) for dv in dvs if dv != 0]
            
            if len(set(signs)) > 1:
                return np.inf
                

            
            # Debug: Print parameters being tested
            if np.random.rand() < 0.01:  # Print 10% of the time
                print(f"  Testing params: X0={params['X0']:.3f}, Y0={params['Y0']:.3f}, depth={params['depth']:.3f}, "
                      f"DVx={params['DVx']:.5f}, DVy={params['DVy']:.5f}, DVz={params['DVz']:.5f}, ")

            # Forward model - ensure inputs are proper types
            X_obs_arr = np.asarray(X_obs, dtype=float)
            Y_obs_arr = np.asarray(Y_obs, dtype=float)
            
            ue, un, uv = pCDM_fast.pCDM(
                X_obs_arr, Y_obs_arr, 
                float(params['X0']), float(params['Y0']), float(params['depth']),
                float(params['omegaX']), float(params['omegaY']), float(params['omegaZ']),
                float(params['DVx']), float(params['DVy']), float(params['DVz']), 
                0.25
            )
            
            # Ensure outputs are arrays
            ue = np.asarray(ue, dtype=float)
            un = np.asarray(un, dtype=float)  
            uv = np.asarray(uv, dtype=float)
            
            # Check for NaN or inf in forward model output
            if np.any(~np.isfinite(ue)) or np.any(~np.isfinite(un)) or np.any(~np.isfinite(uv)):
                return np.inf
            
            # Convert to line-of-sight
            u_los_pred = -(ue * los_e + un * los_n + uv * los_u)
            u_los_pred = np.asarray(u_los_pred, dtype=float)
            
            # Check for NaN or inf in LOS prediction
            if np.any(~np.isfinite(u_los_pred)):
                return np.inf
            
            # Calculate residuals
            residuals = np.asarray(u_los_obs, dtype=float) - u_los_pred
            
            # Check residuals shape and finite values
            if residuals.shape != u_los_obs.shape:
                print(f"Shape mismatch: residuals {residuals.shape} vs observed {u_los_obs.shape}")
                return np.inf
                
            if np.any(~np.isfinite(residuals)):
                return np.inf
            
            # Calculate negative log likelihood (energy to minimize)
            quad_form = residuals.T @ C_inv @ residuals
            if not np.isfinite(quad_form):
                return np.inf
                
            neg_log_lik = 0.5 * (quad_form + C_logdet + len(residuals) * np.log(2 * np.pi))
            
            if not np.isfinite(neg_log_lik):
                return np.inf
            
            return neg_log_lik
            
        except Exception as e:
            if np.random.rand() < 0.001:  # Print occasional errors for debugging
                print(f"  Error in energy_function: {type(e).__name__}: {e}")
            return np.inf
    
    # Initialize with random parameters within bounds
    current_params = {}
    for key, (lower, upper) in bounds.items():
        current_params[key] = np.random.uniform(lower, upper)
    # Use provided starting parameters if available
    if starting_params is not None:
        current_params = starting_params.copy()
        print(f"Using provided starting parameters:")
        for key, val in current_params.items():
            print(f"  {key:10s}: {val:8.4f}")
    else:
        for key, (lower, upper) in bounds.items():
            current_params[key] = np.random.uniform(lower, upper)
        print(f"Generated random starting parameters:")
        for key, val in current_params.items():
            print(f"  {key:10s}: {val:8.4f}")

 

    
    current_energy = energy_function(current_params)
    
    # Track best solution
    best_params = current_params.copy()
    best_energy = current_energy
    
    # Storage for diagnostics
    energy_trace = []
    temperature_trace = []
    acceptance_count = 0
    
    temperature = initial_temp
    param_names = list(current_params.keys())
    
    print(f"\nStarting simulated annealing optimization...")
    print(f"Initial temperature: {initial_temp}, Cooling rate: {cooling_rate}")
    print(f"Initial energy: {current_energy:.3f}")
    print(f"SA bounds: ")
    for key, (lower, upper) in bounds.items():
        print(f"  {key:10s}: [{lower:8.2f}, {upper:8.2f}]")
    
    for i in range(SA_iterations):
        # Propose new parameters by perturbing one parameter
        proposed_params = current_params.copy()
        param_to_change = np.random.choice(param_names)
        
        # Generate proposal with temperature-dependent step size
        step_scale = np.sqrt(temperature / initial_temp)  # Scale step size with temperature
        proposed_params[param_to_change] += np.random.normal(0, step_sizes[param_to_change] * step_scale)
        
        # Calculate energy of proposed state
        proposed_energy = energy_function(proposed_params)
        
        # Accept or reject based on Metropolis criterion
        energy_diff = proposed_energy - current_energy
        
        if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff / temperature):
            # Accept proposal
            current_params = proposed_params
            current_energy = proposed_energy
            acceptance_count += 1
            
            # Update best solution if improved
            if current_energy < best_energy:
                best_params = current_params.copy()
                best_energy = current_energy
        
        # Store diagnostics
        energy_trace.append(current_energy)
        temperature_trace.append(temperature)
        
        # Cool down temperature
        temperature *= cooling_rate
        
        # Progress updates
        if (i + 1) % (SA_iterations // 10) == 0:
            acceptance_rate = acceptance_count / (i + 1)
            print(f"SA iteration {i+1:5d}/{SA_iterations}: "
                  f"T={temperature:.4f}, E={current_energy:.3f}, "
                  f"Best E={best_energy:.3f}, Accept rate={acceptance_rate:.3f}")
        
        # Stop if temperature is too low
        if temperature < min_temp:
            print(f"Stopping SA: temperature {temperature:.6f} below minimum {min_temp}")
            break
    
    final_acceptance_rate = acceptance_count / min(i + 1, SA_iterations)
    print(f"SA completed. Final acceptance rate: {final_acceptance_rate:.3f}")
    print(f"Best energy found: {best_energy:.6f}")
    print(f"Best parameters found:")
    for key, val in best_params.items():
        print(f"  {key:10s}: {val:8.4f}")
    
    return best_params, best_energy, energy_trace, temperature_trace

def plot_sa_diagnostics(energy_trace, temperature_trace, figure_folder=None):
    """
    Plot simulated annealing diagnostics.
    
    Parameters:
    -----------
    energy_trace : list
        Energy evolution during SA
    temperature_trace : list
        Temperature evolution during SA
    figure_folder : str, optional
        Folder to save figures
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    iterations = np.arange(len(energy_trace))
    
    # Plot energy evolution
    ax1.plot(iterations, energy_trace, 'b-', alpha=0.7, linewidth=1,label='Energy (Negative Log Likelihood)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy (Negative Log Likelihood)')
    ax1.set_title('Simulated Annealing: Energy Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Add running minimum
    running_min = np.minimum.accumulate(energy_trace)
    ax1.plot(iterations, running_min, 'r-', linewidth=2, alpha=0.8, label='Running minimum')
    ax1.legend()
    
    # Plot temperature evolution
    ax2.semilogy(iterations, temperature_trace, 'r-', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Temperature (log scale)')
    ax2.set_title('Simulated Annealing: Temperature Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/SA_diagnostics.png", dpi=300, bbox_inches='tight')


def estimate_noise_covariance(X_obs, Y_obs, u_los_obs, sill=None, nugget=None, range_param=None):
    """
    Estimate noise covariance matrix using variogram parameters.
    
    Parameters:
    -----------
    X_obs, Y_obs : array_like
        Observation coordinates
    u_los_obs : array_like
        Observed line-of-sight displacements
    sill : float, optional
        Variogram sill (total variance). If None, estimated from data.
    nugget : float, optional
        Variogram nugget (measurement error variance). If None, estimated.
    range_param : float, optional
        Variogram range (correlation length). If None, estimated.
        
    Returns:
    --------
    C : ndarray
        Covariance matrix
    """
    n = len(u_los_obs)
    
    # Calculate distances between all observation points
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt((X_obs[i] - X_obs[j])**2 + (Y_obs[i] - Y_obs[j])**2)
    
    # Estimate variogram parameters if not provided
    if sill is None:
        sill = np.var(u_los_obs)
    
    if nugget is None:
        nugget = sill * 0.01  # Assume 1% nugget effect
    
    if range_param is None:
        # Estimate range as a fraction of the maximum distance
        max_dist = np.max(distances)
        range_param = max_dist / 3.0
    
    # Construct covariance matrix using exponential model
    # C(h) = nugget + (sill - nugget) * exp(-h/range)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                C[i, j] = sill
            else:
                h = distances[i, j]
                C[i, j] = (sill - nugget) * np.exp(-h / range_param)
    
    return C, sill, nugget, range_param

def bayesian_inference_pCDM_with_noise(u_los_obs, X_obs, Y_obs, incidence_angle, heading, 
                                        n_iterations=10000, proposal_std=None,
                                        sill=None, nugget=None, range_param=None,
                                        adaptive_interval=1000, target_acceptance=0.23,
                                        initial_params=None, priors=None, max_step_sizes=None,use_sa_init=True,figure_folder=None):
    """
    Bayesian inference for pCDM source parameters using Metropolis-Hastings MCMC
    with spatially correlated noise model and adaptive proposal scaling.
    
    Parameters:
    -----------
    u_los_obs : array_like
        Observed line-of-sight displacements
    X_obs, Y_obs : array_like
        Observation coordinates
    incidence_angle : float
        Satellite incidence angle in degrees
    heading : float
        Satellite heading in degrees
    n_iterations : int
        Number of MCMC iterations
    proposal_std : dict
        Standard deviations for proposal distribution
    sill : float, optional
        Variogram sill parameter
    nugget : float, optional
        Variogram nugget parameter
    range_param : float, optional
        Variogram range parameter
    adaptive_interval : int
        Number of iterations between adaptation steps
    target_acceptance : float
        Target acceptance rate (0.23 is optimal for multivariate problems)
    initial_params : dict, optional
        Initial parameter values. If None, uses default values.
    priors : dict, optional
        Prior bounds for each parameter. If None, uses default bounds.
    max_step_sizes : dict, optional
        Maximum allowed step sizes for adaptive scaling. If None, uses defaults.
        
    Returns:
    --------
    samples : dict
        MCMC samples for each parameter
    log_likelihood_trace : array
        Log-likelihood values at each iteration
    """
    
    # Convert to numpy arrays
    u_los_obs = np.array(u_los_obs).flatten()
    X_obs = np.array(X_obs).flatten()
    Y_obs = np.array(Y_obs).flatten()
    
    # Estimate noise covariance matrix
    C, sill, nugget, range_param = estimate_noise_covariance(X_obs, Y_obs, u_los_obs, sill, nugget, range_param)
    
    # Compute inverse and determinant for likelihood calculation
    try:
        C_inv = np.linalg.inv(C)
        C_logdet = np.linalg.slogdet(C)[1]
    except np.linalg.LinAlgError:
        C += np.eye(len(C)) * 1e-8 * np.trace(C) / len(C)
        C_inv = np.linalg.inv(C)
        C_logdet = np.linalg.slogdet(C)[1]
    
    # Convert angles to radians for LOS calculation
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    
    # Line-of-sight unit vector components
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)

    
    # Initialize adaptive proposal tracking
    param_names = list(proposal_std.keys())
    proposal_std_adaptive = proposal_std.copy()
    acceptance_counts = {param: 0 for param in param_names}
    proposal_counts = {param: 0 for param in param_names}

    # If using simulated annealing for initialization
    if use_sa_init:
        print("\nUsing simulated annealing for initial parameter estimation...")
        sa_step_sizes = {key: val * 2.0 for key, val in proposal_std.items()}  # Larger step sizes for SA
        sa_bounds = priors  # Use prior bounds for SA
        best_params, best_energy, energy_trace, temp_trace = simulated_annealing_optimization(
            u_los_obs, X_obs, Y_obs, incidence_angle, heading,
            C_inv, C_logdet, los_e, los_n, los_u,
            SA_iterations=int(n_iterations*2), initial_temp=10.0, cooling_rate=0.95, min_temp=0.01,
            step_sizes=sa_step_sizes, bounds=sa_bounds, starting_params=initial_params
        )
        # print(energy_trace  )
        initial_params = best_params.copy()
        print("Simulated annealing completed. Using best parameters as MCMC initial state.")
        plot_sa_diagnostics(energy_trace, temp_trace)
    
    def log_prior(params):
        """Calculate log prior probability"""
        for key, (lower, upper) in priors.items():
            if not (lower <= params[key] <= upper):
                return -np.inf
            
        #       # # Additional constraint: all DV values must have same sign
        # dvs = [params['DVx'], params['DVy'], params['DVz']]
        # signs = [np.sign(dv) for dv in dvs if dv != 0]

        # if len(set(signs)) > 1:
        #     return -np.inf
        
        return 0.0
    
    def log_likelihood(params):
        """Calculate log likelihood with correlated noise"""
        try:
            # Forward model
            ue, un, uv = pCDM_fast.pCDM(X_obs, Y_obs, params['X0'], params['Y0'], params['depth'],
                                params['omegaX'], params['omegaY'], params['omegaZ'],
                                params['DVx'], params['DVy'], params['DVz'], 0.25)
            
            # Convert to line-of-sight
            u_los_pred = -(ue * los_e + un * los_n + uv * los_u)
            
            # Calculate likelihood with correlated noise
            residuals = u_los_obs - u_los_pred
            rms_value = np.sqrt(np.mean(residuals**2))
            
            # Multivariate Gaussian likelihood
            log_lik = -0.5 * (residuals.T @ C_inv @ residuals + C_logdet + len(residuals) * np.log(2 * np.pi))
            
            return log_lik, rms_value
            
        except:
            return -np.inf, np.nan
    
    def log_posterior(params):
        """Calculate log posterior probability"""
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf, np.nan, -np.inf
        log_lik, residual_rms = log_likelihood(params)
        return lp + log_lik, residual_rms, log_lik
    
    def adapt_proposal_scales(acceptance_counts, proposal_counts, proposal_std_adaptive, 
                             target_acceptance, max_step_sizes):
        """Adapt proposal standard deviations based on acceptance rates"""
        for param in param_names:
            if proposal_counts[param] > 0:
                acceptance_rate = acceptance_counts[param] / proposal_counts[param]
                
                # Calculate adaptation factor (similar to MATLAB version)
                if acceptance_rate > target_acceptance:
                    # Too many acceptances, increase step size
                    adaptation_factor = np.exp((acceptance_rate - target_acceptance) / target_acceptance * 2)
                else:
                    # Too few acceptances, decrease step size
                    adaptation_factor = np.exp((acceptance_rate - target_acceptance) / (1 - target_acceptance) * 2)
                
                # Apply adaptation with bounds
                new_std = proposal_std_adaptive[param] * adaptation_factor
                proposal_std_adaptive[param] = min(new_std, max_step_sizes[param])
                
                # Prevent step size from becoming too small
                proposal_std_adaptive[param] = max(proposal_std_adaptive[param], 
                                                 proposal_std[param] * 1e-5)
        
        # Reset counters
        acceptance_counts = {param: 0 for param in param_names}
        proposal_counts = {param: 0 for param in param_names}

        return acceptance_counts, proposal_counts, proposal_std_adaptive

    # Initialize with provided or default parameters
    current_params = initial_params.copy()
    
    # Storage for samples
    samples = {key: [] for key in current_params.keys()}
    log_likelihood_trace = []
    residuals_evolution = []
    proposal_std_evolution = {key: [] for key in param_names}
    acceptance_rate_evolution = {key: [] for key in param_names}
    
    current_log_post, residual_rms, log_like_current = log_posterior(current_params)
    
    print("Starting MCMC sampling with correlated noise model and adaptive proposals...")
    print(f"Noise parameters - Sill: {sill:.6f}, Nugget: {nugget:.6f}, Range: {range_param:.3f}")
    print(f"Target acceptance rate: {target_acceptance:.2f}")
    print(f"Adaptation interval: {adaptive_interval} iterations")
    print(f"Initial parameters:")
    for key, val in current_params.items():
        print(f"  {key}: {val:.6f}")
    print(f"Initial proposal standard deviations (learning rates):")
    for key, val in proposal_std.items():
        print(f"  {key}: {val:.6f}")
    
    for i in range(n_iterations):
        # Propose new parameters
        proposed_params = current_params.copy()
        
        # Randomly select parameter to update
        param_to_update = np.random.choice(param_names)
        proposed_params[param_to_update] += np.random.normal(0, proposal_std_adaptive[param_to_update])
        
        # Update proposal count
        proposal_counts[param_to_update] += 1
        
        # Calculate acceptance probability
        proposed_log_post, proposed_residual_rms, log_like_prop = log_posterior(proposed_params)
        alpha = min(1, np.exp(proposed_log_post - current_log_post))
        
        # Accept or reject
        if np.random.rand() < alpha:
            current_params = proposed_params
            current_log_post = proposed_log_post
            residual_rms = proposed_residual_rms
            log_like_current = log_like_prop
            acceptance_counts[param_to_update] += 1
        
        # Store samples
        for key in current_params.keys():
            samples[key].append(current_params[key])
        
        residuals_evolution.append(residual_rms)
        log_likelihood_trace.append(log_like_current)
        
        # Store current proposal standard deviations
        for key in param_names:
            proposal_std_evolution[key].append(proposal_std_adaptive[key])
        
        # Adaptive step size adjustment
        if (i + 1) % adaptive_interval == 0 and i > 0:
            # Calculate current acceptance rates
            current_acceptance_rates = {}
            for param in param_names:
                if proposal_counts[param] > 0:
                    current_acceptance_rates[param] = acceptance_counts[param] / proposal_counts[param]
                else:
                    current_acceptance_rates[param] = 0.0
            
            # Store acceptance rates
            for key in param_names:
                acceptance_rate_evolution[key].append(current_acceptance_rates[key])
            
            # Adapt proposal scales
            acceptance_counts, proposal_counts, proposal_std_adaptive = adapt_proposal_scales(
                acceptance_counts, proposal_counts, proposal_std_adaptive, 
                target_acceptance, max_step_sizes)
            
            # # Progress update with adaptation info
            # overall_acceptance = sum(acceptance_counts.values()) / max(1, sum(proposal_counts.values()))
            # print(f"Iteration {i+1}/{n_iterations}")
            # print(f"  Adapted proposal scales:")
            # for param in param_names:
            #     print(f"    {param}: {proposal_std_adaptive[param]:.6f} "
            #           f"(acceptance: {current_acceptance_rates[param]:.3f})")

        # Regular progress update
        # elif (i + 1) % 1000 == 0:
            # Calculate current most likely parameters
            if len(samples[param_names[0]]) > 0 and (i + 1) % int(n_iterations*0.1) == 0:
                current_samples = {param: samples[param][-min(int(n_iterations*0.1), len(samples[param])):] for param in param_names}
                current_means = {param: np.mean(vals) for param, vals in current_samples.items()}
                print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {i+1}/{n_iterations} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"  Adapted proposal scales:")
                for param in param_names:
                    print(f"    {param}: {proposal_std_adaptive[param]:.6f} "
                        f"(acceptance: {current_acceptance_rates[param]:.3f})")

                print(f"  Current most likely parameters (last {min(int(n_iterations*0.1), len(samples[param_names[0]]))} samples):")
                for param in param_names:
                    print(f"    {param}: {current_means[param]:.6f}")
                print(f"  Current RMS: {residual_rms:.6f}")
                print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    # # Final acceptance rate
    # final_acceptance = sum(acceptance_counts.values()) / max(1, sum(proposal_counts.values()))
    # print(f"MCMC completed. Final overall acceptance rate: {final_acceptance:.3f}")
    
    # Print final proposal standard deviations
    print("\nFinal proposal standard deviations:")
    for param in param_names:
        print(f"{param:8s}: {proposal_std_adaptive[param]:.6f}")
    # Save results to CSV

    # Get burn-in point
    burn_in = int(n_iterations * 0.2)
    samples_burned = {key: np.array(val[burn_in:]) for key, val in samples.items()}

    # Create DataFrame with samples
    df_samples = pd.DataFrame(samples_burned)

    # Add additional columns
    df_samples['iteration'] = range(burn_in, len(samples[list(samples.keys())[0]]))
    df_samples['log_likelihood'] = log_likelihood_trace[burn_in:]
    df_samples['rms_residual'] = residuals_evolution[burn_in:]

    # Save to CSV
    csv_filename = f"mcmc_samples_n{n_iterations}_accept{target_acceptance}.csv"
    if figure_folder is not None:
        csv_filename = f"{figure_folder}/{csv_filename}"

    df_samples.to_csv(csv_filename, index=False)
    print(f"MCMC samples saved to: {csv_filename}")

    # Also save summary statistics
    summary_stats = []
    for param in samples_burned.keys():
        mean_val = np.mean(samples_burned[param])
        std_val = np.std(samples_burned[param])
        q025 = np.percentile(samples_burned[param], 2.5)
        q975 = np.percentile(samples_burned[param], 97.5)
        
        summary_stats.append({
            'parameter': param,
            'mean': mean_val,
            'std': std_val,
            'q025': q025,
            'q975': q975
        })
    # Calculate most likely parameters (mean of posterior samples)
    optimal_params = {}
    for param in samples_burned.keys():
        optimal_params[param] = np.mean(samples_burned[param])

    # Also calculate the MAP (maximum a posteriori) estimate
    # Find the sample with highest likelihood
    best_idx = np.argmax(log_likelihood_trace[burn_in:])
    map_params = {}
    for param in samples_burned.keys():
        map_params[param] = samples_burned[param][best_idx]

    # Add optimal parameters to summary stats
    for param in optimal_params.keys():
        summary_stats.append({
            'parameter': f'{param}_optimal_mean',
            'mean': optimal_params[param],
            'std': 0.0,
            'q025': optimal_params[param],
            'q975': optimal_params[param]
        })
        
        summary_stats.append({
            'parameter': f'{param}_MAP',
            'max_likelihood': map_params[param],
            'std': 0.0,
            'q025': map_params[param],
            'q975': map_params[param]
        })

    print("\nOptimal Model Parameters (Posterior Mean):")
    print("-" * 50)
    for param, value in optimal_params.items():
        print(f"{param:8s}: {value:8.4f}")

    print("\nOptimal Model Parameters MAP (Maximum A Posteriori) Parameters:")
    print("-" * 50)
    for param, value in map_params.items():
        print(f"{param:8s}: {value:8.4f}")

    df_summary = pd.DataFrame(summary_stats)
    summary_filename = f"mcmc_summary_n{n_iterations}_accept{target_acceptance}.csv"
    if figure_folder is not None:
        summary_filename = f"{figure_folder}/{summary_filename}"

    df_summary.to_csv(summary_filename, index=False)
    print(f"Summary statistics saved to: {summary_filename}")

    # Save final output to text file
    output_filename = f"inference_results_n{n_iterations}_accept{target_acceptance}.txt"
    if figure_folder is not None:
        output_filename = f"{figure_folder}/{output_filename}"

    with open(output_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BAYESIAN INFERENCE RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Write final acceptance rate
        final_acceptance = sum(acceptance_counts.values()) / max(1, sum(proposal_counts.values()))
        f.write(f"MCMC completed. Final overall acceptance rate: {final_acceptance:.3f}\n\n")
        
        # Write final proposal standard deviations
        f.write("Final proposal standard deviations:\n")
        for param in param_names:
            f.write(f"{param:8s}: {proposal_std_adaptive[param]:.6f}\n")
        f.write("\n")
        
        # Get burn-in point and calculate burned samples
        burn_in = int(n_iterations * 0.2)
        samples_burned = {key: np.array(val[burn_in:]) for key, val in samples.items()}
        
        # Calculate most likely parameters (mean of posterior samples)
        optimal_params = {}
        for param in samples_burned.keys():
            optimal_params[param] = np.mean(samples_burned[param])

        # Calculate the MAP (maximum a posteriori) estimate
        best_idx = np.argmax(log_likelihood_trace[burn_in:])
        map_params = {}
        for param in samples_burned.keys():
            map_params[param] = samples_burned[param][best_idx]

        # Write optimal parameters
        f.write("Optimal Model Parameters (Posterior Mean):\n")
        f.write("-" * 50 + "\n")
        for param, value in optimal_params.items():
            f.write(f"{param:8s}: {value:8.4f}\n")
        f.write("\n")
        
        f.write("Maximum A Posteriori (MAP) Parameters:\n")
        f.write("-" * 50 + "\n")
        for param, value in map_params.items():
            f.write(f"{param:8s}: {value:8.4f}\n")
        f.write("\n")
        
        # Write posterior summary statistics
        f.write("Posterior Summary Statistics:\n")
        f.write("-" * 50 + "\n")
        for param in samples_burned.keys():
            mean_val = np.mean(samples_burned[param])
            std_val = np.std(samples_burned[param])
            q025 = np.percentile(samples_burned[param], 2.5)
            q975 = np.percentile(samples_burned[param], 97.5)
            f.write(f"{param:8s}: {mean_val:8.4f} ± {std_val:6.4f} [{q025:8.4f}, {q975:8.4f}]\n")
        f.write("\n")
        
        # Write convergence diagnostics if samples are long enough
        if len(samples_burned[list(samples_burned.keys())[0]]) > 100:
            f.write("Convergence Diagnostics (post burn-in):\n")
            f.write("-" * 60 + "\n")
            for param in samples_burned.keys():
                post_burnin = samples_burned[param]
                
                # Split into first and second half for comparison
                mid_point = len(post_burnin) // 2
                first_half_mean = np.mean(post_burnin[:mid_point])
                second_half_mean = np.mean(post_burnin[mid_point:])
                
                # Simple convergence metric: difference between halves
                convergence_diff = abs(first_half_mean - second_half_mean)
                overall_std = np.std(post_burnin)
                
                f.write(f"{param:8s}: 1st half mean={first_half_mean:8.4f}, "
                       f"2nd half mean={second_half_mean:8.4f}, "
                       f"diff={convergence_diff:8.4f} ({convergence_diff/overall_std:.2f}σ)\n")
            f.write("\n")
        
        # Write final RMS residual
        if len(residuals_evolution) > 0:
            final_rms = residuals_evolution[-1]
            f.write(f"Final RMS residual: {final_rms:.6f}\n")
            
            # Calculate mean RMS for post burn-in period
            post_burnin_rms = residuals_evolution[burn_in:]
            if len(post_burnin_rms) > 0:
                mean_rms = np.mean(post_burnin_rms)
                std_rms = np.std(post_burnin_rms)
                f.write(f"Post burn-in RMS: {mean_rms:.6f} ± {std_rms:.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")

    print(f"Inference results saved to: {output_filename}")
    
    return (samples, log_likelihood_trace, residuals_evolution, 
            proposal_std_evolution, acceptance_rate_evolution)



def plot_adaptation_diagnostics(proposal_std_evolution, acceptance_rate_evolution, 
                               adaptive_interval, target_acceptance=0.23, figure_folder=None):
    """
    Plot diagnostics for adaptive MCMC including proposal scale evolution 
    and acceptance rate evolution.
    
    Parameters:
    -----------
    proposal_std_evolution : dict
        Evolution of proposal standard deviations for each parameter
    acceptance_rate_evolution : dict
        Evolution of acceptance rates for each parameter
    adaptive_interval : int
        Interval between adaptations
    target_acceptance : float
        Target acceptance rate
    figure_folder : str, optional
        Folder to save figures
    """
    param_names = list(proposal_std_evolution.keys())
    n_params = len(param_names)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Plot proposal standard deviation evolution
    fig_height = max(8, 2.5 * n_rows)
    plt.figure(figsize=(15, fig_height))
    
    for i, param in enumerate(param_names):
        plt.subplot(n_rows, n_cols, i+1)
        iterations = np.arange(len(proposal_std_evolution[param]))
        plt.plot(iterations, proposal_std_evolution[param], color='b', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel(f'{param} Proposal Std')
        plt.title(f'{param} Proposal Scale Evolution')
        plt.grid(True, alpha=0.3)
        
        # Mark adaptation points
        adaptation_points = np.arange(adaptive_interval, len(iterations), adaptive_interval)
        for ap in adaptation_points:
            if ap < len(iterations):
                plt.axvline(ap, color='r', linestyle='--', alpha=0.1)
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/proposal_scale_evolution.png", dpi=300, bbox_inches='tight')
    
    # Plot acceptance rate evolution
    plt.figure(figsize=(15, fig_height))
    
    for i, param in enumerate(param_names):
        plt.subplot(n_rows, n_cols, i+1)
        if len(acceptance_rate_evolution[param]) > 0:
            adaptation_iterations = np.arange(adaptive_interval, 
                                            adaptive_interval * (len(acceptance_rate_evolution[param]) + 1), 
                                            adaptive_interval)
            plt.plot(adaptation_iterations[:len(acceptance_rate_evolution[param])], 
                    acceptance_rate_evolution[param], 'go-', alpha=0.7, markersize=4)
            plt.axhline(target_acceptance, color='r', linestyle='--', alpha=0.8, 
                       label=f'Target ({target_acceptance:.2f})')
        
        plt.xlabel('Iteration')
        plt.ylabel(f'{param} Acceptance Rate')
        plt.title(f'{param} Acceptance Rate Evolution')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        if i == 0:  # Add legend only to first subplot
            plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/acceptance_rate_evolution.png", dpi=300, bbox_inches='tight')

def plot_inference_results(samples, log_likelihood_trace, rms_evolution, burn_in=2000, 
                          u_los_obs=None, X_obs=None, Y_obs=None, 
                          incidence_angle=None, heading=None, figure_folder=None,
                          proposal_std_evolution=None, acceptance_rate_evolution=None,
                          adaptive_interval=None, target_acceptance=0.23):
    """
    Plot MCMC results including trace plots, posterior distributions,
    and comparison between initial and optimal models.
    """
    # Remove burn-in samples
    samples_burned = {key: np.array(val[burn_in:]) for key, val in samples.items()}
    log_lik_burned = np.array(log_likelihood_trace[burn_in:])
    
    # Plot log-likelihood trace
    plt.figure(figsize=(12, 8))
    plt.plot(log_likelihood_trace)
    plt.axvline(burn_in, color='r', linestyle='--', alpha=0.7, label='Burn-in')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood Trace')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/log_likelihood_trace.png", dpi=300)
    
    # Plot parameter traces and histograms
    all_params = ['X0', 'Y0', 'depth', 'DVx', 'DVy', 'DVz', 'omegaX', 'omegaY', 'omegaZ']
    n_params = len(all_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig_height = max(8, 2.5 * n_rows)
    plt.figure(figsize=(12, fig_height))
    
    for i, param in enumerate(all_params):
        plt.subplot(n_rows, n_cols, i+1)
        plt.hist(samples_burned[param], bins=50, density=True, alpha=0.7, color='skyblue')
        plt.xlabel(param)
        plt.ylabel('Density')
        plt.title(f'{param} Posterior')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(samples_burned[param])
        std_val = np.std(samples_burned[param])
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.8)
        # Also add maximum likelihood line
        # Find the sample with highest likelihood after burn-in
        burn_in_offset = burn_in
        best_idx = np.argmax(log_likelihood_trace[burn_in_offset:])
        map_value = samples_burned[param][best_idx]
        plt.axvline(map_value, color='green', linestyle=':', alpha=0.8, linewidth=2)
        plt.text(0.02, 0.98, f'μ={mean_val:.4f}\nσ={std_val:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/MCMC_traces_posteriors.png", dpi=300)
    
    # Plot adaptation diagnostics if available
    if (proposal_std_evolution is not None and acceptance_rate_evolution is not None 
        and adaptive_interval is not None):
        plot_adaptation_diagnostics(proposal_std_evolution, acceptance_rate_evolution,
                                   adaptive_interval, target_acceptance, figure_folder)
    
    # Plot other diagnostics if observation data is provided
    if all(x is not None for x in [u_los_obs, X_obs, Y_obs, incidence_angle, heading]):
        plot_model_comparison(samples, u_los_obs, X_obs, Y_obs, 
                             incidence_angle, heading, log_likelihood_trace, burn_in, figure_folder=figure_folder)
        plot_rms_evolution(rms_evolution, figure_folder=figure_folder)
        plot_parameter_convergence(samples, burn_in, figure_folder=figure_folder)
    
    # Print summary statistics
    print("\nPosterior Summary Statistics:")
    print("-" * 50)
    for param in samples_burned.keys():
        mean_val = np.mean(samples_burned[param])
        std_val = np.std(samples_burned[param])
        q025 = np.percentile(samples_burned[param], 2.5)
        q975 = np.percentile(samples_burned[param], 97.5)
        print(f"{param:8s}: {mean_val:8.4f} ± {std_val:6.4f} [{q025:8.4f}, {q975:8.4f}]")

def plot_rms_evolution(rms_evolution, figure_folder=None):
    """
    Plot RMS residual as a function of MCMC iteration.
    """

    # Plot RMS evolution
    plt.figure(figsize=(12, 8))
    
    # Top subplot: RMS evolution
    # plt.subplot(2, 1, 1)
    plt.plot(rms_evolution, 'b-', alpha=0.7, linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('RMS Residual')
    plt.title('RMS Residual Evolution')
    plt.grid(True, alpha=0.3)
    
    # Add running mean
    window_size = min(500, len(rms_evolution) // 10)
    if window_size > 1:
        running_mean = np.convolve(rms_evolution, np.ones(window_size)/window_size, mode='valid')
        # Create x-axis that matches the length of running_mean
        x_running = np.arange(window_size//2, window_size//2 + len(running_mean))
        plt.plot(x_running, running_mean, 
                'r-', linewidth=2, label=f'Running mean ({window_size} iterations)')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/RMS_evolution.png", dpi=300)
    
    return rms_evolution

def plot_parameter_convergence(samples, burn_in=2000, figure_folder=None):
    """
    Plot the convergence of each parameter over MCMC iterations.
    
    Parameters:
    -----------
    samples : dict
        MCMC samples for each parameter
    burn_in : int
        Number of burn-in iterations to mark on plot
    figure_folder : str, optional
        Folder to save figures
    """
    all_params = ['X0', 'Y0', 'depth', 'DVx', 'DVy', 'DVz', 'omegaX', 'omegaY', 'omegaZ']
    
    # Calculate grid dimensions
    n_params = len(all_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Create figure
    fig_height = max(8, 2.5 * n_rows)
    plt.figure(figsize=(15, fig_height))
    
    for i, param in enumerate(all_params):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Plot trace
        iterations = np.arange(len(samples[param]))
        plt.plot(iterations, samples[param], alpha=0.7, color='blue', linewidth=0.5)
        
        # Mark burn-in period
        if burn_in < len(samples[param]):
            plt.axvline(burn_in, color='red', linestyle='--', alpha=0.8, 
                        label='Burn-in end' if i == 0 else "")
        
        # Calculate and plot running mean for convergence assessment
        window_size = min(500, len(samples[param]) // 10)
        if window_size > 1:
            running_mean = np.convolve(samples[param], np.ones(window_size)/window_size, mode='valid')
            x_running = np.arange(window_size//2, window_size//2 + len(running_mean))
            plt.plot(x_running, running_mean, 'orange', linewidth=2, alpha=0.8)
        
        plt.xlabel('Iteration')
        plt.ylabel(param)
        plt.title(f'{param} Convergence')
        plt.grid(True, alpha=0.3)
        
        # Add final value text
        if len(samples[param]) > burn_in:
            final_mean = np.mean(samples[param][burn_in:])
            final_std = np.std(samples[param][burn_in:])
            plt.text(0.02, 0.98, f'Final: {final_mean:.4f}±{final_std:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend only to first subplot to avoid clutter
    if burn_in < len(samples[all_params[0]]):
        plt.subplot(n_rows, n_cols, 1)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/parameter_convergence.png", dpi=300, bbox_inches='tight')
    
    # Calculate and print convergence diagnostics
    print("\nConvergence Diagnostics (post burn-in):")
    print("-" * 60)
    for param in all_params:
        if len(samples[param]) > burn_in:
            post_burnin = np.array(samples[param][burn_in:])
            
            # Split into first and second half for comparison
            mid_point = len(post_burnin) // 2
            first_half_mean = np.mean(post_burnin[:mid_point])
            second_half_mean = np.mean(post_burnin[mid_point:])
            
            # Simple convergence metric: difference between halves
            convergence_diff = abs(first_half_mean - second_half_mean)
            overall_std = np.std(post_burnin)
            
            print(f"{param:8s}: 1st half mean={first_half_mean:8.4f}, "
                    f"2nd half mean={second_half_mean:8.4f}, "
                    f"diff={convergence_diff:8.4f} ({convergence_diff/overall_std:.2f}σ)")
            


def plot_model_comparison(samples, u_los_obs, X_obs, Y_obs, 
                         incidence_angle, heading, log_likelihood_trace, burn_in=2000, figure_folder=None):
    """
    Plot comparison between initial model, optimal model, and residuals.
    """
    # Get initial parameters (first sample)
    initial_params = {key: val[0] for key, val in samples.items()}
    
    # Get optimal parameters (mean of post-burn-in samples)
    samples_burned = {key: np.array(val[burn_in:]) for key, val in samples.items()}
    optimal_params = {key: np.mean(vals) for key, vals in samples_burned.items()}
    # Also calculate the MAP (maximum a posteriori) estimate using log likelihood
    # Find the sample with highest likelihood
    best_idx = np.argmax(log_likelihood_trace[burn_in:])
    map_params = {key: samples_burned[key][best_idx] for key in samples_burned.keys()}

    print(f"\nMaximum Likelihood Parameters:")
    for param, value in map_params.items():
        print(f"  {param:10s}: {value:8.4f}")

    # Use MAP parameters as optimal parameters for model calculation
    optimal_params = map_params
    # Convert angles to radians for LOS calculation
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    
    # Line-of-sight unit vector components
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)
    
    # Calculate initial model
    ue_init, un_init, uv_init = pCDM_fast.pCDM(X_obs, Y_obs, initial_params['X0'], initial_params['Y0'], 
                                     initial_params['depth'], initial_params['omegaX'], 
                                     initial_params['omegaY'], initial_params['omegaZ'],
                                     initial_params['DVx'], initial_params['DVy'], 
                                     initial_params['DVz'], 0.25)
    u_los_init = -(ue_init * los_e + un_init * los_n + uv_init * los_u)
    
    # Calculate optimal model
    ue_opt, un_opt, uv_opt = pCDM_fast.pCDM(X_obs, Y_obs, optimal_params['X0'], optimal_params['Y0'], 
                                  optimal_params['depth'], optimal_params['omegaX'], 
                                  optimal_params['omegaY'], optimal_params['omegaZ'],
                                  optimal_params['DVx'], optimal_params['DVy'], 
                                  optimal_params['DVz'], 0.25)
    u_los_opt = -(ue_opt * los_e + un_opt * los_n + uv_opt * los_u)
    
    # Calculate residuals
    residual_init = u_los_obs - u_los_init
    residual_opt = u_los_obs - u_los_opt
    
    # Create regular grid for interpolation if data is scattered
    if len(np.unique(X_obs)) > 1 and len(np.unique(Y_obs)) > 1:
        # Create regular grid
        x_min, x_max = np.min(X_obs), np.max(X_obs)
        y_min, y_max = np.min(Y_obs), np.max(Y_obs)
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate data to regular grid
        
        u_obs_grid = griddata((X_obs, Y_obs), u_los_obs, (Xi, Yi), method='cubic')
        u_init_grid = griddata((X_obs, Y_obs), u_los_init, (Xi, Yi), method='cubic')
        u_opt_grid = griddata((X_obs, Y_obs), u_los_opt, (Xi, Yi), method='cubic')
        res_init_grid = griddata((X_obs, Y_obs), residual_init, (Xi, Yi), method='cubic')
        res_opt_grid = griddata((X_obs, Y_obs), residual_opt, (Xi, Yi), method='cubic')
        
        X_plot, Y_plot = Xi, Yi
        u_obs_plot = u_obs_grid
        u_init_plot = u_init_grid
        u_opt_plot = u_opt_grid
        res_init_plot = res_init_grid
        res_opt_plot = res_opt_grid
    else:
        # Assume data is already on regular grid
        try:
            grid_shape = (int(np.sqrt(len(X_obs))), int(np.sqrt(len(X_obs))))
            X_plot = X_obs.reshape(grid_shape)
            Y_plot = Y_obs.reshape(grid_shape)
            u_obs_plot = u_los_obs.reshape(grid_shape)
            u_init_plot = u_los_init.reshape(grid_shape)
            u_opt_plot = u_los_opt.reshape(grid_shape)
            res_init_plot = residual_init.reshape(grid_shape)
            res_opt_plot = residual_opt.reshape(grid_shape)
        except:
            print("Could not reshape data for plotting. Using scatter plots instead.")
            X_plot, Y_plot = X_obs, Y_obs
            u_obs_plot = u_los_obs
            u_init_plot = u_los_init
            u_opt_plot = u_los_opt
            res_init_plot = residual_init
            res_opt_plot = residual_opt
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # for ax in axes:
    #     ax.set_aspect('equal')
    
    # Determine common color scale for observed and optimal
    vmin = np.nanmin([u_obs_plot, u_opt_plot])
    vmax = np.nanmax([u_obs_plot, u_opt_plot])
    
    # Residual color scale
    res_vmax = np.nanmax(np.abs(res_opt_plot))
    res_vmin = -res_vmax
    
    if X_plot.ndim == 2:
        # Contour plots with consistent color scale
        im1 = axes[0].contourf(X_plot, Y_plot, u_obs_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title('Observed Data')
        
        im2 = axes[1].contourf(X_plot, Y_plot, u_opt_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title('Optimal Model')
        
        # Use the same color scale for residuals as the data for direct comparison
        im3 = axes[2].contourf(X_plot, Y_plot, res_opt_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2].set_title('Optimal Residual')
        
        # Add a single colorbar across the bottom representing all three plots
        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        plt.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='Displacement (m)')
    else:
        # Scatter plots
        im1 = axes[0].scatter(X_plot, Y_plot, c=u_obs_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title('Observed Data')
        
        im2 = axes[1].scatter(X_plot, Y_plot, c=u_opt_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title('Optimal Model')
        
        # Bottom row: Residual
        im3 = axes[2].scatter(X_plot, Y_plot, c=res_opt_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2].set_title('Optimal Residual')
    
   
    

    # RMS comparison in the bottom right subplot
    rms_init = np.sqrt(np.nanmean(residual_init**2))
    rms_opt = np.sqrt(np.nanmean(residual_opt**2))
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/Model_Comparison.png", dpi=300)

  



def genereate_synthetic_data(true_params, grid_size=100, noise_level=0.05):
    # Create grid based on grid_size parameter
    x_range = np.linspace(-5, 5, grid_size)
    y_range = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    print(f"Generated grid with {grid_size}x{grid_size} points.")
    
    # True parameters
    true_params = {
        'X0': 0.5, 'Y0': -0.3, 'depth': 1.2,
        'DVx': -0.002, 'DVy': -0.0018, 'DVz': -0.0015,
        'omegaX': 10.0, 'omegaY': -30.0, 'omegaZ': -10.0
    }
    
    # Generate true displacements
    ue_true, un_true, uv_true = pCDM_fast.pCDM(X_flat, Y_flat, true_params['X0'], true_params['Y0'], 
                                        true_params['depth'], true_params['omegaX'], 
                                        true_params['omegaY'], true_params['omegaZ'],
                                        true_params['DVx'], true_params['DVy'], 
                                        true_params['DVz'], 0.25)
    
    # Convert to LOS
    incidence_angle = 39.4
    heading = -169.9
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)
    
    u_los_true = -(ue_true * los_e + un_true * los_n + uv_true * los_u)
    
    # Add noise
    noise_std = np.std(u_los_true) * noise_level  # 5% noise
    u_los_obs = u_los_true + np.random.normal(0, noise_std, len(u_los_true))
    
    # Generate spatially correlated noise
    n_obs = len(u_los_obs)
    distances = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            distances[i, j] = np.sqrt((X_flat[i] - X_flat[j])**2 + (Y_flat[i] - Y_flat[j])**2)
    
    # Noise parameters
    noise_sill = (noise_std)**2  # Total variance
    noise_nugget = noise_sill * 0.01  # 1% nugget effect
    noise_range = np.max(distances) / 4.0  # Correlation length
    
    # Construct noise covariance matrix using exponential model
    C_noise = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            if i == j:
                C_noise[i, j] = noise_sill
            else:
                h = distances[i, j]
                C_noise[i, j] = (noise_sill - noise_nugget) * np.exp(-h / noise_range)
    
    # Generate spatially correlated noise
    spatially_correlated_noise = np.random.multivariate_normal(np.zeros(n_obs), C_noise)
    
    # Add spatially correlated noise instead of independent noise
    u_los_obs = u_los_true + spatially_correlated_noise

    return u_los_obs, X_flat, Y_flat, incidence_angle, heading, noise_sill, noise_nugget, noise_range

def run_baysian_inference(u_los_obs, X_obs, Y_obs, incidence_angle, heading, 
                         n_iterations=10000, sill=None, nugget=None, range_param=None,
                         initial_params=None, priors=None, proposal_std=None, 
                         max_step_sizes=None, adaptive_interval=1000, 
                         target_acceptance=0.23, figure_folder=None,use_sa_init=False):
    """
    Run Bayesian inference and plot results.
    
    Parameters:
    -----------
    u_los_obs : array_like
        Observed line-of-sight displacements
    X_obs, Y_obs : array_like
        Observation coordinates
    incidence_angle : float
        Satellite incidence angle in degrees
    heading : float
        Satellite heading in degrees
    n_iterations : int
        Number of MCMC iterations
    sill : float, optional
        Variogram sill parameter. If None, estimated from data.
    nugget : float, optional
        Variogram nugget parameter. If None, estimated as 1% of sill.
    range_param : float, optional
        Variogram range parameter. If None, estimated as 1/3 of max distance.
    initial_params : dict, optional
        Initial parameter values. If None, uses default values.
    priors : dict, optional
        Prior bounds for each parameter. If None, uses default bounds.
    proposal_std : dict, optional
        Initial proposal standard deviations (learning rates). If None, uses defaults.
    max_step_sizes : dict, optional
        Maximum allowed step sizes for adaptive scaling. If None, uses defaults.
    adaptive_interval : int
        Number of iterations between adaptation steps
    target_acceptance : float
        Target acceptance rate
    figure_folder : str, optional
        Folder to save figures
    """
    # Create figure folder if it doesn't exist
    if figure_folder is not None and not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    
 
    
    # convert from Phase to Displacement (m)
    u_los_obs = -u_los_obs*(0.0555/(4*np.pi))
    
   
    # Estimate noise parameters if not provided
    if sill is None:
        sill = np.var(u_los_obs)
    
    if nugget is None:
        nugget = sill * 0.01
        
    if range_param is None:
        # Calculate maximum distance between observation points
        n_obs = len(X_obs)
        max_dist = 0
        for i in range(n_obs):
            for j in range(i+1, n_obs):
                dist = np.sqrt((X_obs[i] - X_obs[j])**2 + (Y_obs[i] - Y_obs[j])**2)
                if dist > max_dist:
                    max_dist = dist
        range_param = max_dist / 3.0
    
    print("=" * 80)
    print("BAYESIAN INFERENCE CONFIGURATION")
    print("=" * 80)
    
    print(f"\nNoise Model Parameters:")
    print(f"  Sill:         {sill:.6f}")
    print(f"  Nugget:       {nugget:.6f}")
    print(f"  Range:        {range_param:.3f}")
    
    print(f"\nInitial Parameters:")
    for key, val in initial_params.items():
        print(f"  {key:10s}: {val:8.4f}")
    
    print(f"\nPrior Bounds:")
    for key, (lower, upper) in priors.items():
        print(f"  {key:10s}: [{lower:8.4f}, {upper:8.4f}]")
    
    print(f"\nInitial Proposal Standard Deviations (Learning Rates):")
    for key, val in proposal_std.items():
        print(f"  {key:10s}: {val:8.6f}")
    
    print(f"\nMaximum Step Sizes:")
    for key, val in max_step_sizes.items():
        print(f"  {key:10s}: {val:8.4f}")
    
    print(f"\nAdaptive MCMC Settings:")
    print(f"  Adaptation interval:   {adaptive_interval:5d} iterations")
    print(f"  Target acceptance:     {target_acceptance:.3f}")
    print(f"  Total iterations:      {n_iterations:5d}")
    
    print("=" * 80)


    # Run inference with custom parameters
    samples, log_lik_trace, rms_evolution, proposal_std_evolution, acceptance_rate_evolution = bayesian_inference_pCDM_with_noise(
        u_los_obs, X_obs, Y_obs, incidence_angle, heading, 
        n_iterations=int(n_iterations),
        sill=sill, nugget=nugget, range_param=range_param,
        initial_params=initial_params,
        priors=priors,
        proposal_std=proposal_std,
        max_step_sizes=max_step_sizes,
        adaptive_interval=adaptive_interval,
        target_acceptance=target_acceptance,
        use_sa_init=use_sa_init,
        figure_folder=figure_folder)
    
    # Plot results
    plot_inference_results(samples, log_lik_trace, rms_evolution, burn_in=int(n_iterations*0.2), 
                          X_obs=X_obs, Y_obs=Y_obs, u_los_obs=u_los_obs, 
                          incidence_angle=incidence_angle, heading=heading, figure_folder="figure_test",
                          proposal_std_evolution=proposal_std_evolution, 
                          acceptance_rate_evolution=acceptance_rate_evolution,
                          adaptive_interval=adaptive_interval, target_acceptance=target_acceptance)
    
   
    
    return samples, log_lik_trace, rms_evolution
def sythetic_test():
    # Default parameters
    default_initial = {
        'X0': 0,
        'Y0': 0,
        'depth': 5000,
        'DVx': 7e7,
        'DVy': 7e7,
        'DVz': 7e7,
        'omegaX': 0,
        'omegaY': 0,
        'omegaZ': 0
    }
    
    default_priors = {
        'X0': (-15000,15000),
        'Y0': (-15000, 15000),
        'depth': (100, 35000),
        'DVx': (1e2, 1e9),
        'DVy': (1e2, 1e9),
        'DVz': (-1e9, 1e9),
        'omegaX': (-45, 45),
        'omegaY': (-45, 45),
        'omegaZ': (-45, 45)
    }
    
    default_learning_rates = {
        'X0': 100,
        'Y0': 100,
        'depth': 100,
        'DVx': 1e4,
        'DVy': 1e4,
        'DVz': 1e4,
        'omegaX': 1,
        'omegaY': 1,
        'omegaZ': 1
    }

    max_step_sizes = {
            'X0': 100000.0,
            'Y0': 100000.0,
            'depth': 10000.0,
            'DVx': 1e7,
            'DVy': 1e7,
            'DVz': 1e7,
            'omegaX': 20,
            'omegaY': 20,
            'omegaZ': 20
        }
    
    # Generate synthetic data
    u_los_obs, X_obs, Y_obs, incidence_angle, heading, noise_sill, noise_nugget, noise_range = genereate_synthetic_data(
        true_params=None, grid_size=50, noise_level=0.05)
    
    print(f"Generated synthetic data shapes:")
    print(f"  u_los_obs: {u_los_obs.shape}")
    print(f"  X_obs: {X_obs.shape}")
    print(f"  Y_obs: {Y_obs.shape}")
    print(f"  incidence_angle: {incidence_angle}")
    print(f"  heading: {heading}")
    print(f"  noise parameters: sill={noise_sill:.6f}, nugget={noise_nugget:.6f}, range={noise_range:.3f}")

    # Run Bayesian inference with synthetic data and default settings
    samples, log_lik_trace, rms_evolution = run_baysian_inference(
        u_los_obs=u_los_obs,
        X_obs=X_obs,
        Y_obs=Y_obs,
        incidence_angle=incidence_angle,
        heading=heading,
        n_iterations=int(1e4),
        sill=noise_sill,
        nugget=noise_nugget,
        range_param=noise_range,
        initial_params=default_initial,
        priors=default_priors,
        proposal_std=default_learning_rates,
        max_step_sizes=max_step_sizes,
        adaptive_interval=1000,
        target_acceptance=0.23,
        figure_folder="figure_test_synth",
        use_sa_init=True)
    

if __name__ == "__main__":

##### Values to Edit #########
    # Example with custom parameters
    custom_initial = {
        'X0': 0,
        'Y0': 0,
        'depth': 5000,
        'DVx': 7e7,
        'DVy': 7e7,
        'DVz': 7e7,
        'omegaX': 0,
        'omegaY': 0,
        'omegaZ': 0
    }
    #Input Priors here
    custom_priors = {
        'X0': (-15000,15000),
        'Y0': (-15000, 15000),
        'depth': (100, 35000),
        'DVx': (1e2, 1e9),
        'DVy': (1e2, 1e9),
        'DVz': (-1e9, 1e9),
        'omegaX': (-45, 45),
        'omegaY': (-45, 45),
        'omegaZ': (-45, 45)
    }
    # Intial Learning Rates Here
    custom_learning_rates = {
        'X0': 100,
        'Y0': 100,
        'depth': 100,
        'DVx': 1e4,
        'DVy': 1e4,
        'DVz': 1e4,
        'omegaX': 1,
        'omegaY': 1,
        'omegaZ': 1
    }
    # Max Step Sizes Here
    max_step_sizes = {
            'X0': 100000.0,
            'Y0': 100000.0,
            'depth': 10000.0,
            'DVx': 1e7,
            'DVy': 1e7,
            'DVz': 1e7,
            'omegaX': 20,
            'omegaY': 20,
            'omegaZ': 20
        }
       
    # Load data from .npy file
    data = np.load('benji_test.npy', allow_pickle=True)


    sill = 6.1951e-05
    range_param = 15737.072
    nugget = 3.311e-06
    referencePoint = [-67.83951712, -21.77505660]
    ###################################################################

    # Extract data from the loaded object
    data_dict = data.item()
    print(data_dict.keys())
    u_los_obs = np.array(data_dict['Phase']).flatten()

    Lon = np.array(data_dict['Lon']).flatten()
    Lat = np.array(data_dict['Lat']).flatten()
   
    X_obs, Y_obs = convert_lat_long_2_xy(Lon, Lat, referencePoint[0], referencePoint[1])
    print(f"Converted Lon/Lat to X/Y with reference point {referencePoint}")
    print(f"X_obs range: {X_obs.min():.3f} to {X_obs.max():.3f}")
    print(f"Y_obs range: {Y_obs.min():.3f} to {Y_obs.max():.3f}")
    print(len(X_obs), len(Y_obs), len(u_los_obs))
    incidence_angle = np.mean(np.array(data_dict['Inc']))
    heading = np.mean(np.array(data_dict['Heading']))

    print(f"Loaded data shapes:")
    print(f"  u_los_obs: {u_los_obs.shape}")
    print(f"  X_obs: {X_obs.shape}")
    print(f"  Y_obs: {Y_obs.shape}")
    print(f"  incidence_angle: {incidence_angle}")
    print(f"  heading: {heading}")
  

    # Run Bayesian inference with synthetic data and custom settings
    samples, log_lik_trace, rms_evolution = run_baysian_inference(
        u_los_obs=u_los_obs, 
        X_obs=X_obs, 
        Y_obs=Y_obs, 
        incidence_angle=incidence_angle, 
        heading=heading,
        n_iterations=int(1e4),
        sill=sill,
        nugget=nugget,
        range_param=range_param,
        initial_params=custom_initial,
        priors=custom_priors,
        proposal_std=custom_learning_rates,
        max_step_sizes=max_step_sizes,
        adaptive_interval=1000,
        target_acceptance=0.23,
        figure_folder="figure_test",
        use_sa_init=True)
    
  
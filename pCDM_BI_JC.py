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
import pCDM_BI_plotting_funcs as pCDM_BI_plotting_funcs
import pCDM_BI_simulated_annealing as pCDM_BI_simulated_annealing
import pickle
from datetime import datetime

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
                                            initial_params=None, priors=None, max_step_sizes=None,
                                            use_sa_init=True, figure_folder=None, model_type='pCDM'):
        """
        Bayesian inference for various source models using Metropolis-Hastings MCMC
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
        model_type : str
            Type of forward model to use ('pCDM', 'Mogi', 'Yang', etc.)
            
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

        # Define forward models
        def forward_model(params, model_type):
            """
            Generic forward model dispatcher
            """
            if model_type == 'pCDM':
                ue, un, uv = pCDM_fast.pCDM(X_obs, Y_obs, params['X0'], params['Y0'], params['depth'],
                                    params['omegaX'], params['omegaY'], params['omegaZ'],
                                    params['DVx'], params['DVy'], params['DVz'], 0.25)
            
            elif model_type == 'Mogi':
                # Mogi point source model
                # Expected parameters: X0, Y0, depth, DV
                # ue, un, uv = mogi_model(X_obs, Y_obs, params['X0'], params['Y0'], 
                #                        params['depth'], params['DV'])
                pass # Placeholder to avoid error
            
            elif model_type == 'Yang':
                # Yang finite spherical source model
                # Expected parameters: X0, Y0, depth, DV, radius
                # ue, un, uv = yang_model(X_obs, Y_obs, params['X0'], params['Y0'], 
                #                        params['depth'], params['DV'], params['radius'])
                pass  # Placeholder to avoid error
            
            elif model_type == 'McTigue':
                # McTigue prolate spheroid model
                # Expected parameters: X0, Y0, depth, DV, a, c, strike, dip, plunge
                # ue, un, uv = mctigue_model(X_obs, Y_obs, params['X0'], params['Y0'], 
                #                           params['depth'], params['DV'], params['a'], 
                #                           params['c'], params['strike'], params['dip'], 
                #                           params['plunge'])
                pass  # Placeholder to avoid error
            
            elif model_type == 'Okada':
                # Okada rectangular dislocation model
                # Expected parameters: X0, Y0, depth, length, width, strike, dip, rake, slip
                # ue, un, uv = okada_model(X_obs, Y_obs, params['X0'], params['Y0'], 
                #                         params['depth'], params['length'], params['width'],
                #                         params['strike'], params['dip'], params['rake'], 
                #                         params['slip'])
                pass  # Placeholder to avoid error
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return ue, un, uv

     
        
        # Initialize adaptive proposal tracking
        param_names = list(proposal_std.keys())
        proposal_std_adaptive = proposal_std.copy()
        acceptance_counts = {param: 0 for param in param_names}
        proposal_counts = {param: 0 for param in param_names}

        # If using simulated annealing for initialization only works for pCDM needs changing 
        if use_sa_init:
            print(f"\nUsing simulated annealing for initial {model_type} parameter estimation...")
            sa_step_sizes = {key: val * 2.0 for key, val in proposal_std.items()}
            sa_bounds = priors
            best_params, best_energy, energy_trace, temp_trace = pCDM_BI_simulated_annealing.simulated_annealing_optimization(
                u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                C_inv, C_logdet, los_e, los_n, los_u,
                SA_iterations=int(n_iterations*2), initial_temp=10.0, cooling_rate=0.95, min_temp=0.01,
                step_sizes=sa_step_sizes, bounds=sa_bounds, starting_params=initial_params,
                model_type=model_type
            )
            initial_params = best_params.copy()
            print(f"Simulated annealing completed for {model_type}. Using best parameters as MCMC initial state.")
            pCDM_BI_plotting_funcs.plot_sa_diagnostics(energy_trace, temp_trace)
        
        def log_prior(params):
            """Calculate log prior probability"""
            for key, (lower, upper) in priors.items():
                if not (lower <= params[key] <= upper):
                    return -np.inf
            
            # Model-specific constraints
            if model_type == 'pCDM':
                # Additional constraint: all DV values must have same sign (if desired) not needed as caught by the model itself 
                # dvs = [params['DVx'], params['DVy'], params['DVz']]
                # signs = [np.sign(dv) for dv in dvs if dv != 0]
                # if len(set(signs)) > 1:
                #     return -np.inf
                pass
            
            elif model_type in ['Mogi', 'Yang']: # examples area for contraits 
                # Volume change should be reasonable
                if 'DV' in params and abs(params['DV']) > 1e10:
                    return -np.inf
            
            elif model_type == 'McTigue': # examples area for contraits 
                # Semi-axes should be positive and reasonable
                if 'a' in params and 'c' in params:
                    if params['a'] <= 0 or params['c'] <= 0:
                        return -np.inf
                    if params['a'] > 10000 or params['c'] > 10000:  # reasonable bounds
                        return -np.inf
            
            elif model_type == 'Okada': # examples area for contraits 
                # Fault dimensions should be positive
                if 'length' in params and 'width' in params:
                    if params['length'] <= 0 or params['width'] <= 0:
                        return -np.inf
            
            return 0.0
        
        def log_likelihood(params):
            """Calculate log likelihood with correlated noise"""
            try:
                # Forward model
                ue, un, uv = forward_model(params, model_type)
                
                # Convert to line-of-sight
                u_los_pred = -(ue * los_e + un * los_n + uv * los_u)
                
                # Calculate likelihood with correlated noise
                residuals = u_los_obs - u_los_pred
                rms_value = np.sqrt(np.mean(residuals**2))
                
                # Optimized multivariate Gaussian likelihood using einsum for faster matrix operations
                log_lik = -0.5 * (np.einsum('i,ij,j->', residuals, C_inv, residuals) + C_logdet + len(residuals) * np.log(2 * np.pi))
                
                return log_lik, rms_value
                
            except Exception as e:
                print(f"Forward model error: {e}")
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
                    
                    # Calculate adaptation factor
                    if acceptance_rate > target_acceptance:
                        adaptation_factor = np.exp((acceptance_rate - target_acceptance) / target_acceptance * 2)
                    else:
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

        # Initialize with provided parameters
        current_params = initial_params.copy()
        
        # Storage for samples
        samples = {key: [] for key in current_params.keys()}
        log_likelihood_trace = []
        residuals_evolution = []
        proposal_std_evolution = {key: [] for key in param_names}
        acceptance_rate_evolution = {key: [] for key in param_names}
        
        current_log_post, residual_rms, log_like_current = log_posterior(current_params)
        
        print(f"Starting MCMC sampling with {model_type} model and correlated noise...")
        print(f"Model parameters: {list(param_names)}")
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
                
                # Print diagnostics
                if len(samples[param_names[0]]) > 0 and (i + 1) % int(n_iterations*0.1) == 0:
                    current_samples = {param: samples[param][-min(int(n_iterations*0.1), len(samples[param])):] for param in param_names}
                    current_means = {param: np.mean(vals) for param, vals in current_samples.items()}
                    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {i+1}/{n_iterations} ({model_type}) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"  Adapted proposal scales:")
                    for param in param_names:
                        print(f"    {param}: {proposal_std_adaptive[param]:.6f} "
                            f"(acceptance: {current_acceptance_rates[param]:.3f})")

                    print(f"  Current most likely parameters (last {min(int(n_iterations*0.1), len(samples[param_names[0]]))} samples):")
                    for param in param_names:
                        print(f"    {param}: {current_means[param]:.6f}")
                    print(f"  Current RMS: {residual_rms:.6f}")
                    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
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
        csv_filename = f"mcmc_samples_{model_type}_n{n_iterations}_accept{target_acceptance}.csv"
        if figure_folder is not None:
            csv_filename = f"{figure_folder}/{csv_filename}"

        df_samples.to_csv(csv_filename, index=False)
        print(f"MCMC samples saved to: {csv_filename}")

        # Calculate optimal parameters and save summary
        summary_stats = []
        optimal_params = {}
        map_params = {}
        
        for param in samples_burned.keys():
            optimal_params[param] = np.mean(samples_burned[param])
            
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

        # Calculate MAP estimate
        best_idx = np.argmax(log_likelihood_trace[burn_in:])
        for param in samples_burned.keys():
            map_params[param] = samples_burned[param][best_idx]

        print(f"\n{model_type} Model - Optimal Parameters (Posterior Mean):")
        print("-" * 50)
        for param, value in optimal_params.items():
            print(f"{param:8s}: {value:8.4f}")

        print(f"\n{model_type} Model - MAP (Maximum A Posteriori) Parameters:")
        print("-" * 50)
        for param, value in map_params.items():
            print(f"{param:8s}: {value:8.4f}")

        # Save summary and results
        df_summary = pd.DataFrame(summary_stats)
        summary_filename = f"mcmc_summary_{model_type}_n{n_iterations}_accept{target_acceptance}.csv"
        if figure_folder is not None:
            summary_filename = f"{figure_folder}/{summary_filename}"

        df_summary.to_csv(summary_filename, index=False)
        print(f"Summary statistics saved to: {summary_filename}")

        # Save detailed output
        output_filename = f"inference_results_{model_type}_n{n_iterations}_accept{target_acceptance}.txt"
        if figure_folder is not None:
            output_filename = f"{figure_folder}/{output_filename}"

        with open(output_filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BAYESIAN INFERENCE RESULTS - {model_type} MODEL\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Model Parameters: {list(param_names)}\n\n")
            
            # Write final acceptance rate
            final_acceptance = sum(acceptance_counts.values()) / max(1, sum(proposal_counts.values()))
            f.write(f"MCMC completed. Final overall acceptance rate: {final_acceptance:.3f}\n\n")
            
            # Write final proposal standard deviations
            f.write("Final proposal standard deviations:\n")
            for param in param_names:
                f.write(f"{param:8s}: {proposal_std_adaptive[param]:.6f}\n")
            f.write("\n")
            
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
            
            # Write final RMS residual
            if len(residuals_evolution) > 0:
                final_rms = residuals_evolution[-1]
                f.write(f"Final RMS residual: {final_rms:.6f}\n")
                
                post_burnin_rms = residuals_evolution[burn_in:]
                if len(post_burnin_rms) > 0:
                    mean_rms = np.mean(post_burnin_rms)
                    std_rms = np.std(post_burnin_rms)
                    f.write(f"Post burn-in RMS: {mean_rms:.6f} ± {std_rms:.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")

        print(f"Inference results saved to: {output_filename}")
        
        return (samples, log_likelihood_trace, residuals_evolution, 
                proposal_std_evolution, acceptance_rate_evolution)

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

def save_inference_state(samples, log_lik_trace, rms_evolution, proposal_std_evolution, 
                        acceptance_rate_evolution, X_obs, Y_obs, u_los_obs, 
                        incidence_angle, heading, inference_params, figure_folder=None):
    """
    Save complete inference state to pickle file for later regeneration.
    
    Parameters:
    -----------
    samples : dict
        MCMC samples for each parameter
    log_lik_trace : array
        Log-likelihood trace
    rms_evolution : array
        RMS residual evolution
    proposal_std_evolution : dict
        Evolution of proposal standard deviations
    acceptance_rate_evolution : dict
        Evolution of acceptance rates
    X_obs, Y_obs : array_like
        Observation coordinates
    u_los_obs : array_like
        Observed line-of-sight displacements
    incidence_angle : float
        Satellite incidence angle
    heading : float
        Satellite heading
    inference_params : dict
        All inference parameters (initial params, priors, etc.)
    figure_folder : str, optional
        Folder to save the pickle file
    """
    
    # Create comprehensive state dictionary
    inference_state = {
        # Core results
        'samples': samples,
        'log_likelihood_trace': log_lik_trace,
        'rms_evolution': rms_evolution,
        'proposal_std_evolution': proposal_std_evolution,
        'acceptance_rate_evolution': acceptance_rate_evolution,
        
        # Input data
        'X_obs': X_obs,
        'Y_obs': Y_obs,
        'u_los_obs': u_los_obs,
        'incidence_angle': incidence_angle,
        'heading': heading,
        
        # All inference parameters
        'inference_params': inference_params,
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'n_iterations': len(log_lik_trace),
        'burn_in': int(len(log_lik_trace) * 0.2)
    }
    
    # Generate filename
    n_iterations = len(log_lik_trace)
    target_acceptance = inference_params.get('target_acceptance', 0.23)
    pickle_filename = f"bayesian_inference_state_n{n_iterations}_accept{target_acceptance}.pkl"
    
    if figure_folder is not None:
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)
        pickle_filename = os.path.join(figure_folder, pickle_filename)
    
    # Save to pickle
    with open(pickle_filename, 'wb') as f:
        pickle.dump(inference_state, f)
    
    print(f"Complete inference state saved to: {pickle_filename}")
    print(f"File size: {os.path.getsize(pickle_filename) / (1024*1024):.2f} MB")
    
    return pickle_filename

def load_inference_state(pickle_filename):
    """
    Load inference state from pickle file.
    
    Parameters:
    -----------
    pickle_filename : str
        Path to the pickle file
        
    Returns:
    --------
    inference_state : dict
        Complete inference state dictionary
    """
    with open(pickle_filename, 'rb') as f:
        inference_state = pickle.load(f)
    
    print(f"Loaded inference state from: {pickle_filename}")
    print(f"Timestamp: {inference_state.get('timestamp', 'Unknown')}")
    print(f"Number of iterations: {inference_state.get('n_iterations', 'Unknown')}")
    print(f"Burn-in period: {inference_state.get('burn_in', 'Unknown')}")
    
    return inference_state

def regenerate_plots_from_state(pickle_filename, new_figure_folder=None):
    """
    Regenerate all plots from saved inference state.
    
    Parameters:
    -----------
    pickle_filename : str
        Path to the pickle file containing inference state
    new_figure_folder : str, optional
        New folder to save regenerated plots. If None, uses timestamp-based folder.
    """
    # Load the inference state
    inference_state = load_inference_state(pickle_filename)
    
    # Extract data
    samples = inference_state['samples']
    log_lik_trace = inference_state['log_likelihood_trace']
    rms_evolution = inference_state['rms_evolution']
    proposal_std_evolution = inference_state['proposal_std_evolution']
    acceptance_rate_evolution = inference_state['acceptance_rate_evolution']
    X_obs = inference_state['X_obs']
    Y_obs = inference_state['Y_obs']
    u_los_obs = inference_state['u_los_obs']
    incidence_angle = inference_state['incidence_angle']
    heading = inference_state['heading']
    inference_params = inference_state['inference_params']
    burn_in = inference_state['burn_in']
    
    # Create figure folder if not specified
    if new_figure_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_figure_folder = f"regenerated_plots_{timestamp}"
    
    if not os.path.exists(new_figure_folder):
        os.makedirs(new_figure_folder)
    
    print(f"Regenerating plots in folder: {new_figure_folder}")
    
    # Regenerate plots using the plotting function
    pCDM_BI_plotting_funcs.plot_inference_results(
        samples,
        log_lik_trace,
        rms_evolution,
        burn_in=burn_in,
        X_obs=X_obs,
        Y_obs=Y_obs,
        u_los_obs=u_los_obs,
        incidence_angle=incidence_angle,
        heading=heading,
        figure_folder=new_figure_folder,
        proposal_std_evolution=proposal_std_evolution,
        acceptance_rate_evolution=acceptance_rate_evolution,
        adaptive_interval=inference_params.get('adaptive_interval', 1000),
        target_acceptance=inference_params.get('target_acceptance', 0.23)
    )

   
    print(f"All plots regenerated in: {new_figure_folder}")

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

    # Store all inference parameters for saving
    inference_params = {
        'n_iterations': n_iterations,
        'sill': sill,
        'nugget': nugget,
        'range_param': range_param,
        'initial_params': initial_params,
        'priors': priors,
        'proposal_std': proposal_std,
        'max_step_sizes': max_step_sizes,
        'adaptive_interval': adaptive_interval,
        'target_acceptance': target_acceptance,
        'use_sa_init': use_sa_init
    }

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
    
    # Save complete inference state to pickle
    pickle_filename = save_inference_state(
        samples=samples,
        log_lik_trace=log_lik_trace,
        rms_evolution=rms_evolution,
        proposal_std_evolution=proposal_std_evolution,
        acceptance_rate_evolution=acceptance_rate_evolution,
        X_obs=X_obs,
        Y_obs=Y_obs,
        u_los_obs=u_los_obs,
        incidence_angle=incidence_angle,
        heading=heading,
        inference_params=inference_params,
        figure_folder=figure_folder
    )
    
    # Plot results
    pCDM_BI_plotting_funcs.plot_inference_results(samples, log_lik_trace, rms_evolution, burn_in=int(n_iterations*0.2), 
                          X_obs=X_obs, Y_obs=Y_obs, u_los_obs=u_los_obs, 
                          incidence_angle=incidence_angle, heading=heading, figure_folder=figure_folder,
                          proposal_std_evolution=proposal_std_evolution, 
                          acceptance_rate_evolution=acceptance_rate_evolution,
                          adaptive_interval=adaptive_interval, target_acceptance=target_acceptance)
    
    return samples, log_lik_trace, rms_evolution, pickle_filename


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
       
    # # Load data from .npy file
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
    samples, log_lik_trace, rms_evolution,pickle_filename = run_baysian_inference(
        u_los_obs=u_los_obs, 
        X_obs=X_obs, 
        Y_obs=Y_obs, 
        incidence_angle=incidence_angle, 
        heading=heading,
        n_iterations=int(1e5),
        sill=sill,
        nugget=nugget,
        range_param=range_param,
        initial_params=custom_initial,
        priors=custom_priors,
        proposal_std=custom_learning_rates,
        max_step_sizes=max_step_sizes,
        adaptive_interval=1000,
        target_acceptance=0.23,
        figure_folder="figure_test_realdata_1e6",
        use_sa_init=True)
    
  
  #### Reload old inference state and regenerate plots example ####
    # pickle_file = "./figure_test/bayesian_inference_state_n10000_accept0.23.pkl"
    # regenerate_plots_from_state(pickle_file, new_figure_folder="regenerated_plots_test")
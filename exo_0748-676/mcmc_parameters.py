import numpy as np
import subprocess
from astropy import constants as const

# Directory for storing inlist, LOGS, and MCMC output files.
model_directory = 'basic_run'
mcmc_inlist = 'inlist'
median_inlist = '{}/inlist'.format(model_directory)
LOGS = '{}/LOGS'.format(model_directory)

source_name = 'EXO 0748-676'

corner_plot_title = source_name
corner_plot_file = '{}/corner-exo_0748-676.png'.format(model_directory)

best_fit_plot_title = source_name
best_fit_plot_file = '{}/cooling_curve-exo_0748-676.png'.format(model_directory)

autocorrelations_plot_title = 'Autocorrelations'
autocorrelations_plot_file = '{}/autocorrelations-exo_0748-676.png'.format(model_directory)

mcmc_output_file = '{}/output-exo_0748-676'.format(model_directory)

# Times of observations of the cooling curve in days after end of outburst.
observation_times = np.array([39.2, 62.4, 172.5, 194.0, 278.5, 299.3, 592.1, 650.2, 775.6, 1031.0, 1683.2, 1791.1, 3526.3])
# Effective temperature in eV measured at the observation times.
Teff_obs = np.array([129.6, 125.9, 122.7, 120.4, 119.7, 118.0, 117.2, 117.2, 116.2, 117.8, 111.9, 110.8, 114.3])
# Uncertainties of the temperature observations in eV.
Teff_obs_sigma = np.array([0.4, 0.2, 0.4, 0.2, 0.4, 0.1, 0.5, 0.2, 0.5, 0.4, 0.1, 0.4, 0.1])
# Boltzmann constant for converting temperature from eV to MK
k_B = const.k_B.to('eV/K').value
# Observed effective temperature in MK. Initial data is in eV and is converted.
Teff_obs *= 10**-6/k_B
# Error on effective temperature in MK. Initial data is in eV and is converted.
Teff_obs_sigma *= 10**-6/k_B

# Maximum number of MCMC walker steps and number of walkers.
max_n = 20000
nwalkers = 32
# The MCMC can be considered converged if the number of steps is autocorrelation_convergence_factor times the maximum autocorrelation time.
autocorrelation_convergence_factor = 100
samples_file = '{}/samples-exo_0748-676.h5'.format(model_directory)
parameter_distributions_array = '{}/parameter_distributions.npy'.format(model_directory)
autocorrelations_array = '{}/autocorrelations.npy'.format(model_directory)

# List of names of parameters to vary as they appear in the dStar inlist.
# Must also have an option as a command line argument (set in run.f file).
# Options include: ['core_mass', 'core_radius', 'core_temperature', 'Qimp', 'Q_heating_shallow', 'Q_heating_inner']
parameter_names = np.array(['core_temperature', 'Qimp', 'Q_heating_shallow', 'Q_heating_inner'])
# Command line options corresponding to each parameter.
# Options include: ['-M', '-R', '-T', '-Q', '-s', '-i']
parameter_options = np.array(['-T', '-Q', '-s', '-i'])

# Initial guess for parameter values.
parameter_values_initial = np.array([1.0e8, 4.0, 2.0, 1.5])

# Minumum and maximum values for each parameter.
# We use these values to set uniform priors bounded by a lower and upper limit.
# To have no prior, set the minimum values to -np.inf and the maximum values to np.inf
parameter_minimum_values = np.array([1.0e7, 0.0, 0.0, 0.0])
parameter_maximum_values = np.array([1.0e9, 40.0, 40.0, 40.0])

# The number of observations to compare the model to.
num_obs = Teff_obs.shape[0]

def log_prior_uniform(parameter_values, parameter_minimum_values, parameter_maximum_values):
    if np.any(parameter_values < parameter_minimum_values) or np.any(parameter_values > parameter_maximum_values):
        return -np.inf
    else:
        volume = np.prod(parameter_maximum_values - parameter_minimum_values)
        return np.log(1/volume)

def calculate_chi2(parameter_values):
    '''
    Calculates the reduced Chi^2 of a dStar model with respect to observed effective temperature (Teff_obs).

    Only takes the parameter_values array as an argument for convenient use with emcee. The number of observations (num_obs), inlist, LOGS directory, parameter options, 
    Teff_obs, and Teff_obs_sigma must all be assigned outside the function.

    Parameters
    ----------

    parameter_values: array-like'
        Input array of parameters for dStar model.
        Must modify the src/run.f file to accomodate command line arguments for each parameter.
        Parameters here are: [core_temperature, Qimp, Q_heating_shallow]

    Returns
    -------

    chi2: float
        The reduced Chi^2 of the dStar model Teff.

    '''
    # Run the models and calculate the log of the probability.
    
    # The number of degrees of freedom of the model.
    dof = num_obs - parameter_values.shape[0]

    # The list of arguments to be used in the command line to run the model
    arguments = ['./run_dStar', '-I', mcmc_inlist, '-L', LOGS]+[str(option) for parameter in zip(parameter_options, parameter_values) for option in parameter]
    
    # Try to run the dStar model which calculates and outputs the Teff.
    # If the model can't run or returns some sort of error, we will assume that this set of parameters is invalid and return np.inf.
    try:
        Teff = np.array(subprocess.run(arguments, capture_output=True, text=True).stdout.split()[-num_obs:], dtype=float)
        
        chi2 = np.sum(((Teff - Teff_obs)/Teff_obs_sigma)**2)/dof
        
        return chi2
    except:
        return np.inf

def log_probability(parameter_values):
    '''
    Calculates the log of the probability of a dStar model for an MCMC.
    
    The log of the probability is based on the chi^2 error assuming a Gaussian probability distribution: probability = exp(-chi2/2). I ignore leading coefficients since 
    the MCMC only uses ratios of probabilities so they would cancel anyway.

    Uses uniform priors for all model parameters with given minimum values and maximum values.
    
    Only takes the parameter_values array as an argument for convenient use with emcee.
    '''
    
    chi2 = calculate_chi2(parameter_values)

    return log_prior_uniform(parameter_values, parameter_minimum_values, parameter_maximum_values) - chi2/2

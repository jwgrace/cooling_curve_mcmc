import numpy as np
import subprocess
from astropy import constants as const

# Directory for storing inlist, LOGS, and MCMC output files.
model_directory = 'basic_run'
inlist = 'inlist'
LOGS = '{}/LOGS'.format(model_directory)

source_name = 'KS 1731-260'

corner_plot_title = source_name
corner_plot_file = '{}/corner-ks_1731-260.png'.format(model_directory)

best_fit_plot_title = source_name
best_fit_plot_file = '{}/cooling_curve-ks_1731-260.png'.format(model_directory)

autocorrelations_plot_title = 'Autocorrelations - {}'.format(source_name)
autocorrelations_plot_file = '{}/autocorrelations-ks_1731-260.png'.format(model_directory)

mcmc_output_file = '{}/output-ks_1731-260'.format(model_directory)

# dStar will output Teff for every epoch (accretion and observation) excluding the starting time of the first epoch.
# For calculating a Chi^2 for the cooling curve we only want the observation epochs.
# This array will mask which of the Teff outputs we want to use for calculating the Chi^2.
is_observation_epoch = np.array([False, True, True, True, True, True, True, True, True])
# Times of observations of the cooling curve in days after end of outburst.
observation_times = np.array([65.1, 235.7, 751.6, 929.5, 1500.5, 1582.9, 3039.7, 5312.1])
# Effective temperature in eV measured at the observation times.
Teff_obs = np.array([104.6, 89.5, 76.4, 73.8, 71.7, 70.3, 64.5, 64.4])
# Uncertainties of the temperature observations in eV.
Teff_obs_sigma = np.array([1.3, 1.03, 1.8, 1.9, 1.4, 1.9, 1.8, 1.2])
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
samples_file = '{}/samples-ks_1731-260.h5'.format(model_directory)
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
    # We don't need history files for these runs so set write_interval_for_history to a large value so that it never writes.
    arguments = ['./run_dStar', '-I', inlist, '-H', '1000']+[str(option) for parameter in zip(parameter_options, parameter_values) for option in parameter]
    
    # Try to run the dStar model which calculates and outputs the Teff.
    # If the model can't run or returns some sort of error, we will assume that this set of parameters is invalid and return np.inf.
    try:
        Teff = np.array(subprocess.run(arguments, capture_output=True, text=True).stdout.split(), dtype=float)[is_observation_epoch]
        
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

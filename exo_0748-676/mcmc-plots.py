import numpy as np
import importlib
import os
import sys
import getopt
import subprocess
import emcee
import corner
from multiprocessing import Pool
import matplotlib.pyplot as plt

# Set path to dStar directory to import reader for reading history.data files
home_directory = os.getenv('HOME')    
ns_cooling_directory = home_directory + '/Documents/neutron_stars/cooling_curves'
sys.path.append(ns_cooling_directory)

import ns_cooling as ns
import ns_accretion as acc

argument_list = sys.argv[1:]
options = 'P'
long_options = ['parameters=']

mcmc_parameters_file = 'mcmc_parameters'

try:
    # Parsing argument
    arguments, values = getopt.getopt(argument_list, options, long_options)

    for argument, value in arguments:
        if argument in ('-P', '--parameters'):
            mcmc_variables_file = value

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    
# Import file containing all variables for the MCMC.
mcmc = importlib.import_module(mcmc_parameters_file)

reader = emcee.backends.HDFBackend(mcmc.samples_file)

tau = reader.get_autocorr_time()
# Set burnin time as a multiple of the maximum autocorrelation time.
burnin = int(2 * np.max(tau))
#thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True)

ndim = samples.shape[1]

# Create a corner plot of the parameters.

fig = corner.corner(samples, labels=mcmc.parameter_names)

fig.suptitle(mcmc.corner_plot_title, fontsize=18)
plt.tight_layout()
fig.savefig(mcmc.corner_plot_file, dpi=150, facecolor='white')

plt.clf()

# Array for storing the middle 68% values for each parameter.
parameters_distribution = np.zeros((ndim, 3))

for i in range(ndim):
    parameters_distribution[i] = np.percentile(samples[:,i], [16, 50, 84])

np.save('{}/parameters_distribution.npy'.format(mcmc.model_directory), parameters_distribution)

# Create a plot of the autocorrelation time as a function of number of iterations.

autocorrelations = np.load(mcmc.autocorrelations_array)

iterations = np.arange(mcmc.autocorrelation_convergence_factor, mcmc.max_n+mcmc.autocorrelation_convergence_factor, mcmc.autocorrelation_convergence_factor)

for i, autocorrelation in enumerate(autocorrelations.T):
    plt.plot(iterations, autocorrelation, label=mcmc.parameter_names[i])
    
plt.xlabel('iterations')
plt.ylabel('tau')
plt.title(mcmc.autocorrelations_plot_title)
plt.legend()
plt.tight_layout()
plt.savefig(mcmc.autocorrelations_plot_file, dpi=150, facecolor='white')

plt.clf()

# We will choose the median parameters to be the "best-fit" parameters for the purposes of plotting a cooling curve.
median_parameters = parameters_distribution[:,1]
# "1-sigma" (68%) +/- errors.
parameters_err_low = parameters_distribution[:,1] - parameters_distribution[:,0]
parameters_err_high = parameters_distribution[:,2] - parameters_distribution[:,1]

# Run a dStar model with the median parameters to produce a history file.
# Plot the cooling curve with the median parameters.

# Write new inlist for the median parameters.
# Only change is to set write_interval_for_history = 1 so that it produces a history file for plotting purposes.
ns.write_inlist(['write_interval_for_history'], [1], mcmc.median_inlist, base_inlist_file=mcmc.mcmc_inlist)

arguments = ['./run_dStar', '-I', mcmc.median_inlist, '-L', mcmc.LOGS]+[str(option) for parameter in zip(mcmc.parameter_options, median_parameters) for option in parameter]

# Run dStar with median parameters to save history file.
subprocess.run(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# Calculate the reduced Chi^2 with the median parameters.
median_chi2 = mcmc.calculate_chi2(median_parameters)

t, lg_L, lg_Teff, Teff_inf = ns.load_LOGS_data(mcmc.LOGS)

# Convert from K to MK.
Teff_model = 1e-6*Teff_inf

fig = plt.figure()
fig.set_dpi(150)
fig.patch.set_facecolor('white')

plt.errorbar(mcmc.observation_times, mcmc.Teff_obs, yerr=mcmc.Teff_obs_sigma, fmt='k.')
plt.semilogx(t[t>0], Teff_model[t>0])
plt.xlabel('Time After Outburst (Days)')
plt.ylabel('Teff (MK)')
plt.title(mcmc.best_fit_plot_title)
plt.tight_layout()
plt.savefig(mcmc.best_fit_plot_file, dpi=150, facecolor='white')

# Append parameter information to the MCMC output file.

# Determine maximum parameter name length to align columns in output file.
max_name_len = 0

for parameter_name in mcmc.parameter_names:
    if len(parameter_name) > max_name_len:
        max_name_len = len(parameter_name)

f = open(mcmc.mcmc_output_file, 'a')
f.write('Parameter Distributions (Middle 68%):\n')

for i, parameter_name in enumerate(mcmc.parameter_names):    
    # core_temperature value will be an exponential. Other parameters will just be floats.
    if parameter_name == 'core_temperature':
        f.write('{:<{width}}:\t{:.02e}\t-{:.02e}\t+{:.02e}\n'.format(parameter_name, median_parameters[i], parameters_err_low[i], parameters_err_high[i], 
                                                                     width=max_name_len))
    else:
        f.write('{:<{width}}:\t{:.02f}\t\t-{:.02f}\t\t+{:.02f}\n'.format(parameter_name, median_parameters[i], parameters_err_low[i], parameters_err_high[i], 
                                                                         width=max_name_len))
        
f.write('\nMedian Parameters Chi^2:\t' + '{:.02f}\n'.format(median_chi2))
f.close()

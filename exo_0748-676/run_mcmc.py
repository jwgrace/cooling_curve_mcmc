import numpy as np
import importlib
import sys
import getopt
import subprocess
import emcee
from multiprocessing import Pool
import tracemalloc
import time

tracemalloc.start()

argument_list = sys.argv[1:]
options = 'P'
long_options = ['parameters=']

mcmc_parameters_file = 'mcmc_parameters'

try:
    # Parsing argument
    arguments, values = getopt.getopt(argument_list, options, long_options)

    for argument, value in arguments:
        if argument in ('-P', '--parameters'):
            mcmc_parameters_file = value

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    
# Import file containing all the mcmc variables.
mcmc = importlib.import_module(mcmc_parameters_file)

parameter_options = mcmc.parameter_options
Teff_obs = mcmc.Teff_obs
Teff_obs_sigma = mcmc.Teff_obs_sigma
parameter_minimum_values = mcmc.parameter_minimum_values
parameter_maximum_values = mcmc.parameter_maximum_values

# The number of dimensions in parameter space.
ndim = mcmc.parameter_values_initial.shape[0]

# Array to track autocorrelation times throughout the MCMC.
autocorrelations = np.full((mcmc.max_n//mcmc.autocorrelation_convergence_factor, ndim), np.nan)

# 2D array of initial positions of the walkers.
# Center the walkers on the initial guess and randomly vary each parameter by up to 10%.
pos = mcmc.parameter_values_initial + .1*mcmc.parameter_values_initial*np.random.randn(mcmc.nwalkers, ndim)

backend = emcee.backends.HDFBackend(mcmc.samples_file)
backend.reset(mcmc.nwalkers, ndim)

i = 0

start = time.time()

with Pool() as pool:
    sampler = emcee.EnsembleSampler(mcmc.nwalkers, ndim, mcmc.log_probability, pool=pool, backend=backend)
    #sampler.run_mcmc(pos, max_steps, progress=True)

    # Will use this for testing convergence.
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=mcmc.max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % mcmc.autocorrelation_convergence_factor:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        
        # Save the autocorrelations for this iteration.
        autocorrelations[i] = tau
        np.save(mcmc.autocorrelations_array, autocorrelations)
        i += 1

        # Convergence conditions:
        # 1. The number of MCMC iterations has exceeded the autocorrelation convergence factor times the greatest autocorrelation time.
        # 2. The fractional change in the autocorrelation is less than a tolerance of .01.
        converged = np.all(tau * mcmc.autocorrelation_convergence_factor < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
        
end = time.time()

# Convert the runtime from seconds to hours, minutes, seconds for easier readability.
runtime = end - start
runtime_hours = int(runtime//3600)
runtime_minutes = int(runtime//60 - 60*runtime_hours)
runtime_seconds = int(runtime%60)

autocorrelations[i] = tau
np.save(mcmc.autocorrelations_array, autocorrelations)

current_memory, peak_memory = tracemalloc.get_traced_memory()

tracemalloc.stop()

# Save MCMC walker and runtime output to file.
f = open(mcmc.mcmc_output_file, 'w')

f.write('{}\n\n'.format(mcmc.source_name))
f.write('Number of walkers:\t{}\n'.format(mcmc.nwalkers))
f.write('Number of iterations:\t{}\n'.format(sampler.iteration))
f.write('Runtime:\t{}:{}:{}\n\n'.format(runtime_hours, runtime_minutes, runtime_seconds))

f.close()

from numba import jit
import numpy as np
from scipy import interpolate
from scipy import integrate
import os
import sys
import subprocess
from glob import glob
import matplotlib.pyplot as plt
import time

# Set path to dStar directory to import reader for reading history.data files
home_directory = os.getenv('HOME')
dStar_directory = home_directory + '/dStar/'
sys.path.append(dStar_directory + 'tools/')

import reader
    
ns_cooling_directory = home_directory + '/Documents/neutron_stars/cooling_curves'
default_inlist = ns_cooling_directory + '/inlist-basic'
    
default_inlist = ns_cooling_directory + '/inlist-basic'

def write_inlist(parameter_names, parameter_values, new_inlist_file, base_inlist_file=default_inlist):

    # Strings in the inlist must be written with quotes.
    # Check each parameter value in the list to see if it is a string. If it is, add quote marks.
    for parameter_value in parameter_values:
        if type(parameter_value) == str:
            parameter_value = "'{}'".format(parameter_value)
    
    # Array to indicate which parameters still need to be written to the inlist.
    # After a parameter assignment has been written in the inlist, its need_to_write value will be marked False.
    # This will allow us to skip parameters that have already been written when searching for a match to a parameter assignment in each line.
    # It will also tell us if we reach the end of the document and still have some parameters to write.
    need_to_write = np.ones_like(parameter_names, dtype=bool)
    
    base_inlist = open(base_inlist_file, 'r')
    new_inlist = open(new_inlist_file, 'w')
    
    # Read the base inlist line-by-line.
    # For each line, check if it is a parameter assignment.
    # If it is, re-write the line with the new parameter value.
    # If it is not, copy the line from the base inlist.
    for line in base_inlist.readlines():
    
        # The inlist ends with '/'
        # So we check if we have reached the end of the document and determine whether we have written all the parameter assignments or not.
        # If any assignments remain to be written, write them.
        # After writing all assignments, end the document with '/'
        if line == '/\n':
            if np.any(need_to_write):
                
                new_inlist.write('\n')
                
                for i, parameter_name in enumerate(parameter_names):
                    if need_to_write[i]:
                        new_inlist.write("    {} = {}\n".format(parameter_name, parameter_values[i]))
                    else:
                        continue

                # Mark the end of the inlist with '/'
                new_inlist.write('/')
                break
            
            else:
                new_inlist.write('/')
                break
        
        else:
            # Marks whether or not this line was one in which we re-wrote a parameter assignment.
            # This is to indicate whether we need to copy the line of the base inlist or not.
            parameter_line = False

            # Loop through all parameters that still need to be written.
            for i, parameter_name in enumerate(parameter_names):
                if not need_to_write[i]:
                    continue
                # A parameter assignment line will begin with the parameter name, followed by the assignment value.
                elif parameter_name == line.split('=')[0].strip():
                    new_inlist.write("    {} = {}\n".format(parameter_name, parameter_values[i]))
                    need_to_write[i] = False
                    parameter_line = True
                    break
                else:
                    continue
                    
            # If it is not one of the lines to change, then copy it to each new inlist.
            if not parameter_line:
                new_inlist.write(line)
                continue
    
    base_inlist.close()
    new_inlist.close()
    
    return None
    
def load_LOGS_data(LOGS_directory):
    '''
    Load a dStar history.data file and read the times, luminosities, and effective temperatures of the models.
    '''
    f = reader.dStarReader('{}/history.data'.format(LOGS_directory))

    # time after outburts in days
    t = f.data.time
    # log(L) [ergs]
    lg_L = f.data.lg_Lsurf
    # log(T_eff) [K]
    lg_Teff = f.data.lg_Teff
    
    # constants in cgs
    c = 3e10
    G = 6.67e-8
    M_sun = 1.989e33

    M = M_sun*f.header.total_mass
    R = f.header.total_radius*1e5
    redshift_factor = (1 - 2*G*M/(R*c**2))**(-.5)
    
    Teff_inf = 10**(lg_Teff)/redshift_factor
    
    return t, lg_L, lg_Teff, Teff_inf

def dStar_args_list(parameter_options, parameter_values):
    
    arguments = ['./run_dStar']+[str(option) for parameter in zip(parameter_options, parameter_values) for option in parameter]
    
    return arguments
    
def write_epoch_datafile(t_epoch, accretion_rates, epoch_datafile_name, epoch_datafile_base_name='epoch_datafile_base'):
    '''
    Write the epoch datafile to be read by dStar, specified in the dStar inlist.
    
    Copies a base datafile with the necessary headers and then writes the start
    times and corresponding average accretion rates of each epoch, line-by-line.
    
    Parameters
    ----------
    t_epoch : array
        Array of the start times of each accretion epoch.
    accretion_rates : array
        Array of the average accretion rates corresponding to the epochs 
        specified by t_epoch.
    epoch_datafile_name : string
        The name of the epoch datafile to be saved.
    epoch_datafile_base_name : string, optional
        The name of the base epoch datafile.
        This contains the necessary headers which will be copied to 
        epoch_datafile_name.
    '''
    # Base file to copy to accretion history file. Includes necessary headers.
    epoch_datafile_base = open(epoch_datafile_base_name, 'r')
    # Accretion history file to write to.
    epoch_datafile = open(epoch_datafile_name, 'w')

    # Copy lines from base file to accretion history file.
    for line in epoch_datafile_base.readlines():
        epoch_datafile.write(line)

    epoch_datafile_base.close()

    # Write times and accretion rates to accretion history file.
    for i, t in enumerate(t_epoch):
        epoch_datafile.write('{:8.4f}\t{:.4e}\n'.format(t, accretion_rates[i]))

    epoch_datafile.close()

def interpolate_epochs(accretion_epochs, observation_epochs, t_acc, accretion_rates):
    '''
    Interpolates and integrates over accretion rates to give average accretion 
    rates for desired epochs to use in an epoch datafile for dStar.
    
    To calculate the average accretion rates for each epoch, we linearly 
    interpolate the observed accretion rates then integrate over each epoch.
    
    Returns start times for all epochs (accretion, start of quiescence, and 
    observation times) and accretion rates corresponding to those epochs.
    
    Parameters
    ----------
    accretion_epochs : array
        The start times of the accretion epochs over which the accretion rates 
        will be averaged.
    observation_epochs : array
        The desired times to take observations. These times don't contribute to
        the accretion history, but are necessary for ensuring that dStar 
        outputs temperature information to compare to observations.
    t_acc : array
        Times at which the accretion rate was measured.
    accretion_rates : array
        Accretion rates corresponding to the times in t_acc
    
    Returns
    -------
    t_epochs : array
        The start times of all epochs (accretion, start of quiescence, and 
        observation)
    Mdot : array
        The accretion rates corresponding to each epoch. The only non-zero
        values are for the accretion epochs.
    '''
    
    Mdot_interp = interpolate.interp1d(t_acc, accretion_rates)
    
    num_accretion_epochs = accretion_epochs.shape[0]
    num_observation_epochs = observation_epochs.shape[0]
    
    # The total number of epochs will include all accretion epochs, t = 0.0, then all observation times.
    t_epochs = np.zeros(num_accretion_epochs + num_observation_epochs + 1)
    t_epochs[0:num_accretion_epochs] = accretion_epochs
    t_epochs[-num_observation_epochs:] = observation_epochs

    dt = t_epochs[1:] - t_epochs[:-1]
    
    Mdot = np.zeros_like(t_epochs)

    # Calculate the average accretion rate for each accretion epoch by integrating over the interpolated accretion function.
    for i in range(num_accretion_epochs):
        Mdot[i] = integrate.quad(Mdot_interp, t_epochs[i], t_epochs[i+1])[0]/dt[i]
        
    return t_epochs, Mdot

def set_up_accretion_history_files(accretion_epochs, observation_epochs, t_acc, accretion_rates, base_directory):
    '''
    Create the necessary epoch datafile, inlist, and LOGS directory for a dStar
    model with the desired accretion and observation epochs and observational 
    accretion history data.
    '''
    # The nuber of accretion epochs for the model.
    num_accretion_epochs = accretion_epochs.shape[0]
    
    # Directories within the base directories where the epoch datafile, inlist, and LOGS directory will be stored.
    epoch_datafile_directories = '{}/epoch_datafiles'.format(base_directory)
    inlist_directories = '{}/inlists'.format(base_directory)
    LOGS_directories = '{}/LOGS_directories'.format(base_directory)
    
    # If any of these directories don't exist, create them.
    if not os.path.exists(epoch_datafile_directories):
        os.mkdir(epoch_datafile_directories)
    if not os.path.exists(inlist_directories):
        os.mkdir(inlist_directories)
    if not os.path.exists(LOGS_directories):
        os.mkdir(LOGS_directories)
    
    # Arrays of all epochs (accretion and observation) and the corresponding accretion rates.
    t, Mdot = interpolate_epochs(accretion_epochs, observation_epochs, t_acc, accretion_rates)

    # Files for saving the accretion history, the inlist, and the LOGS output.
    # The files and LOGS directory for this model are indicated by the number of epochs.
    epoch_datafile = '{}/epoch_datafile-{:04d}'.format(epoch_datafile_directories, num_accretion_epochs)
    inlist_file = '{}/inlist-{:04d}'.format(inlist_directories, num_accretion_epochs)
    LOGS_directory = '{}/LOGS-{:04d}'.format(LOGS_directories, num_accretion_epochs)

    # Create the LOGS directory if it doesn't already exist.
    if not os.path.exists(LOGS_directory):
        os.mkdir(LOGS_directory)

    # Write the epoch datafile file.
    write_epoch_datafile(t, Mdot, epoch_datafile)

    # Write the inlist file.
    write_inlist(inlist_file, epoch_datafile)
    
def time_models(base_directory):
    '''
    Run and time every model within a given directory of models.
    '''
    # Sort the inlists and their corresponding LOGS directories from fewest epochs to most epochs.
    inlists = sorted(glob('{}/inlists/*'.format(base_directory)))
    LOGS = sorted(glob('{}/LOGS_directories/*'.format(base_directory)))

    # Arrays for storing the number of epochs and the runtime of each model.
    num_accretion_epochs = np.zeros(len(inlists))
    runtimes = np.zeros(len(inlists))

    for i, inlist in enumerate(inlists):

        # Read the number at the end of the inlist name to determine the number of accretion epochs.
        num_accretion_epochs[i] = int(inlist[-4:])

        # Run and time the model.
        start = time.time()
        subprocess.run(['./run_dStar', '-I', inlist, '-L', LOGS[i]], capture_output=True, text=True)
        end = time.time()

        runtimes[i] = end - start
        
    return num_accretion_epochs, runtimes

def create_and_time_models(epoch_spacing_func, num_accretion_epochs_batch, t_acc, accretion_rates, observation_epochs, base_directory, **kwargs):
    '''
    Create a batch of dStar models with differing numbers of accretion epochs and run and time them all.
    '''
    # Directory to store all accretion history files.
    epoch_datafile_directories = '{}/epoch_datafiles'.format(base_directory)
    # Directory to store all inlists.
    inlist_directories = '{}/inlists'.format(base_directory)
    # Directory to store all LOGS directories.
    LOGS_directories = '{}/LOGS_directories'.format(base_directory)

    # If these directories do not already exist, create them.
    if not os.path.exists(base_directory):
        os.mkdir(base_directory)
    if not os.path.exists(epoch_datafile_directories):
        os.mkdir(epoch_datafile_directories)
    if not os.path.exists(inlist_directories):
        os.mkdir(inlist_directories)
    if not os.path.exists(LOGS_directories):
        os.mkdir(LOGS_directories)
        
    # Write all accretion history and inlist files and create LOGS directories.

    total_accretion_time = -t_acc[0]
    
    # Loop through the num_accretion_epochs array.
    for i, num_accretion_epochs in enumerate(num_accretion_epochs_batch):

        # Array of accretion epochs. Each element marks the time of the beginning of the corresponding epoch.
        accretion_epochs = epoch_spacing_func(total_accretion_time, num_accretion_epochs, **kwargs)

        if np.any(accretion_epochs == None):
            break

        else:
            # Write the accretion history and inlist files and create a LOGS directory.
            set_up_accretion_history_files(accretion_epochs, observation_epochs, t_acc, accretion_rates, base_directory)
            
    # Time the accretion models.
    num_accretion_epochs, runtimes = time_models(base_directory)
    
    return num_accretion_epochs, runtimes

def constant_epochs(total_accretion_time, num_accretion_epochs, dt_tolerance=1.):
    
    accretion_epochs = np.linspace(-total_accretion_time, 0., num_accretion_epochs, endpoint=False)
    
    if -accretion_epochs[-1] < dt_tolerance:
        return None
    
    else:
        return accretion_epochs
    
def linear_epochs(total_accretion_time, num_intervals, dt_tolerance=1.):
    
    dt0 = 2*total_accretion_time/(num_intervals*(num_intervals + 1))
    
    if dt0 < dt_tolerance:
        print('Shortest epoch is too short.')
        return None
    
    epoch_lengths = dt0*np.arange(1, num_intervals + 1)[::-1]
    
    accretion_epochs = np.zeros(num_intervals)
    
    accretion_epochs[0] = -total_accretion_time
    
    for i, dt in enumerate(epoch_lengths[:-1]):
        accretion_epochs[i+1] = accretion_epochs[i] + dt
        
    return accretion_epochs

def logarithmic_epochs(total_accretion_time, num_intervals, log_base=2, dt_tolerance=1.):
    dt0 = total_accretion_time/np.sum(log_base**np.arange(num_intervals))

    if dt0 < dt_tolerance:
        print('Shortest epoch is too short.')
        return None
    
    accretion_epochs = np.zeros(num_intervals)

    epoch_lengths = dt0*np.logspace(0., num_intervals-1, num_intervals, base=log_base)[::-1]

    accretion_epochs[0] = -total_accretion_time

    for i, dt in enumerate(epoch_lengths[:-1]):
        accretion_epochs[i+1] = accretion_epochs[i] + dt
        
    return accretion_epochs

def random_epochs(total_accretion_time, num_intervals, dt_tolerance=1., num_epoch_failure_tolerance=100):
    
    valid_epochs = False
    num_epoch_failures = 0
    
    while not valid_epochs:
        accretion_epochs = np.sort(total_accretion_time*(np.random.rand(num_intervals) - 1))
        
        dt = np.zeros(num_intervals)
        
        dt[:-1] = accretion_epochs[1:] - accretion_epochs[:-1]
        dt[-1] = -accretion_epochs[-1]
        
        if dt.min() < dt_tolerance:
            num_epoch_failures += 1
            
            if num_epoch_failures >= num_epoch_failure_tolerance:
                print('Failed to create valid epochs.')
                return None
            else:
                continue
        else:
            valid_epochs = True
            
    return accretion_epochs

def first_point_error(LOGS, t_first, Teff_first_high_res):
    '''
    Calculates the error in the temperature of the cooling curve at the first observation point relative to the high resolution model.
    '''
    t, lg_L, lg_Teff, Teff_inf = load_LOGS_data(LOGS)

    Teff_first = Teff_inf[t >= t_first][0]*1e-6

    error = np.abs((Teff_first - Teff_first_high_res))
    
    return error

def errors_batch(base_directory, t_first, Teff_first_high_res):
    
    LOGS_directories = sorted(glob('{}/LOGS_directories/*'.format(base_directory)))

    num_accretion_epochs = np.zeros(len(LOGS_directories))
    errors = np.zeros_like(num_accretion_epochs)

    for i, LOGS in enumerate(LOGS_directories):

        num_accretion_epochs[i] = int(LOGS[-4:])

        errors[i] = first_point_error(LOGS, t_first, Teff_first_high_res)
        
    return num_accretion_epochs, errors

@jit(nopython=True)
def filter_Teff(Teff_full, t_model, t_obs):
    
    Teff_filtered = np.zeros_like(t_obs)
    
    for i in range(len(t_obs)):
        for j in range(len(t_model)):
            if t_model[j] == t_obs[i]:
                Teff_filtered[i] = Teff_full[j]
                break
                
    return Teff_filtered

def calculate_chi2(LOGS, t_obs, Teff_obs, Teff_obs_sigma):
    '''
    Calculates the chi^2 error in the temperature of the cooling curve at the observation times.
    '''
    t, lg_L, lg_Teff, Teff_inf = load_LOGS_data(LOGS)

    Teff_model = filter_Teff(Teff_inf, t, t_obs)        

    chi2 = np.sum(((Teff_model - Teff_obs)/Teff_obs_sigma)**2)
    
    return chi2

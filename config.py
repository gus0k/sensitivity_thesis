import os
import socket
from os.path import expanduser

hostname = socket.gethostname()

home = expanduser("~")
SIMULATION_PARAMETERS = home + '/Simulations/sensitivity'

if not os.path.isdir(SIMULATION_PARAMETERS):
    os.makedirs(SIMULATION_PARAMETERS)
if hostname == 'guso-inspiron':
    DATAPATH = home + '/github/coop_extension_code/load_profiles/'
    CPLEX_PATH = None 
elif hostname == 'lame23':
    DATAPATH = home + '/coop_extension_code/coop_extension_code/load_profiles/'
    CPLEX_PATH = '/home/infres/dkiedanski/Cplex/cplex/bin/x86-64_linux/cplex'

DATA = DATAPATH + 'home_data_2012-13.csv'
DATA_SOLAR = DATAPATH + 'home_data_2012-13_rand_03.csv'
DATA_FORCAST = DATAPATH + 'home_data_2012-13_forcast.csv'
DATA_SOLAR_FORCAST = DATAPATH + 'home_data_2012-13_rand_03_forcast.csv'





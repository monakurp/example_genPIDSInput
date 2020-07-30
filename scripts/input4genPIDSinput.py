import numpy as np
import scipy.ndimage.measurements as snm

np.set_printoptions(precision=3, suppress=False)

#%% Give information

dx = 2.0  # grid spacing in x-direction
dy = 2.0  # grid spacing in y-direction

nx = 20  # number of grid points in the x-direction
ny = 20  # number of grid points in the y-direction

ncat = 4  # number of emission categories

output_path = '../input/'

#%% Create maps:

topoR = np.zeros([ny,nx])-9999
oroR = np.zeros([ny,nx])
idR = np.zeros([ny,nx])
saR = np.zeros([ny,nx,ncat])

# Topography: create buildings
topoR[0:4,0:4] = 10
topoR[16:20,0:4] = 20
topoR[16:20,16:20] = 30
topoR[0:4,16:20] = 40

# Individual building IDs:
Rmod = np.copy(topoR)
Rmod[Rmod>0] = 1
Rmod[Rmod<=0] = 0
labeled_array, num_features = snm.label(Rmod)
labeled_array[labeled_array==0] = -9999
idR = labeled_array.astype(int)
del Rmod, labeled_array, num_features

# Source areas for aerosols:
saR[:,8:12,0] = 1 # traffic: street
saR[:,8:12,1] = 1 # road dust: street
saR[1:3,1:3,2] = 1 # wood combustion: chimneys
saR[17:19,1:3,2] = 1 # wood combustion: chimneys
saR[10,10,3] = 1 # other: random point

# Source areas for gases: only traffic exhaust and wood combustion
sgR = saR[:,:,[0,2]]

#%% Create emission_time_factors for aerosols

nhoursyear = 24 * 365
nmonthdayhour = 91

wood_burning_activity = np.sin(np.linspace(0,2*np.pi,24)+2.5)
wood_burning_activity = wood_burning_activity - np.min(wood_burning_activity)
wood_burning_activity = wood_burning_activity / np.sum(wood_burning_activity)

diurnal_cycle_weekday = [[0.009, 0.004, 0.004, 0.009, 0.029, 0.039, 0.056, 0.053, 0.051, 0.051,
                         0.052, 0.055, 0.059, 0.061, 0.064, 0.067, 0.069, 0.069, 0.049, 0.039,
                         0.039, 0.029, 0.024, 0.019], # traffic combustion
                         [0.009, 0.004, 0.004, 0.009, 0.029, 0.039, 0.056, 0.053, 0.051, 0.051,
                         0.052, 0.055, 0.059, 0.061, 0.064, 0.067, 0.069, 0.069, 0.049, 0.039,
                         0.039, 0.029, 0.024, 0.019], # road dust
                         [0.065, 0.055, 0.045, 0.033, 0.023, 0.014, 0.006, 0.002, 0.000, 0.001,
                          0.005, 0.012, 0.021, 0.031, 0.042, 0.053, 0.063, 0.071, 0.078, 0.081,
                          0.081, 0.078, 0.073, 0.065], # wood combustion (heating)
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.5, 0., 0.,
                          0., 0., 0., 0., 0., 0.]] # other

diurnal_cycle_sat = [[0.038, 0.025, 0.023, 0.024, 0.023, 0.020, 0.015, 0.015, 0.020, 0.028, 0.036,
                     0.047, 0.054, 0.061, 0.062, 0.063, 0.064, 0.062, 0.058, 0.062, 0.053, 0.049,
                     0.051, 0.047], # traffic combustion
                     [0.038, 0.025, 0.023, 0.024, 0.023, 0.020, 0.015, 0.015, 0.020, 0.028, 0.036,
                     0.047, 0.054, 0.061, 0.062, 0.063, 0.064, 0.062, 0.058, 0.062, 0.053, 0.049,
                     0.051, 0.047], # road dust
                     [0.065, 0.055, 0.045, 0.033, 0.023, 0.014, 0.006, 0.002, 0.000, 0.001, 0.005,
                      0.012, 0.021, 0.031, 0.042, 0.053, 0.063, 0.071, 0.078, 0.081, 0.081, 0.078,
                      0.073, 0.065], # wood combustion (heating)
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.5, 0., 0., 0.,
                      0., 0., 0., 0., 0.]] # other

diurnal_cycle_sun = [[0.029, 0.022, 0.02 , 0.02 , 0.02 , 0.018, 0.012, 0.015, 0.024, 0.031, 0.039,
                     0.044, 0.052, 0.063, 0.066, 0.072, 0.062, 0.065, 0.065, 0.066, 0.06 , 0.058,
                     0.041, 0.036], # traffic combustion
                     [0.029, 0.022, 0.02 , 0.02 , 0.02 , 0.018, 0.012, 0.015, 0.024, 0.031, 0.039,
                     0.044, 0.052, 0.063, 0.066, 0.072, 0.062, 0.065, 0.065, 0.066, 0.06 , 0.058,
                     0.041, 0.036], # road dust
                     [0.065, 0.055, 0.045, 0.033, 0.023, 0.014, 0.006, 0.002, 0.000, 0.001, 0.005,
                      0.012, 0.021, 0.031, 0.042, 0.053, 0.063, 0.071, 0.078, 0.081, 0.081, 0.078,
                      0.073, 0.065], # wood combustion (heating)
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.5, 0., 0., 0.,
                      0., 0., 0., 0., 0.]] # other

weekly_cycle = [[0.16, 0.16, 0.16, 0.16, 0.16, 0.12, 0.08], # traffic combustion
                [0.16, 0.16, 0.16, 0.16, 0.16, 0.12, 0.08], # road dust
                [0.13, 0.13, 0.13, 0.13, 0.13, 0.17, 0.18], # wood combustion (heating)
                [0.20, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00]] # other

monthly_cycle = [[0.09, 0.09, 0.09, 0.09, 0.09, 0.06, 0.06, 0.07, 0.09, 0.09, 0.09, 0.09], # traffic combustion
                 [0.10, 0.15, 0.18, 0.20, 0.15, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.10], # road dust
                 [0.20, 0.20, 0.15, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.10, 0.20], # wood combustion (heating)
                 [0.10, 0.10, 0.10, 0.10, 0.10, 0.00, 0.00, 0.10, 0.10, 0.10, 0.10, 0.10]] # other

# emission time factor (etf) lod=1: scaling factors based on the month, day and hour
etf_lod1 = np.zeros([ncat, nmonthdayhour])
for n in range(ncat):
  etf_lod1[n,0:12] = monthly_cycle[n]
  etf_lod1[n,12:19] = weekly_cycle[n]
  etf_lod1[n,19:43] = diurnal_cycle_weekday[n]
  etf_lod1[n,43:67] = diurnal_cycle_sat[n]
  etf_lod1[n,67:91] = diurnal_cycle_sun[n]

# emission time factor (etf) lod=1: scaling factors for each hour of the year
etf_lod2 = np.zeros([ncat, nhoursyear])
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

for n in range(ncat):
  i = 0
  dow = 0 # day of week

  for mm in range(12):

    for dd in range(days_in_month[mm]):

      if dow>6:
        dow = 0
      if dow<5:
        diurnal_cycle = diurnal_cycle_weekday[n]
      elif dow==5:
        diurnal_cycle = diurnal_cycle_sat[n]
      else:
        diurnal_cycle = diurnal_cycle_sun[n]

      for hh in range(24):
        etf_lod2[n,i] = monthly_cycle[n][mm] * weekly_cycle[n][dow] * diurnal_cycle[hh]
        i += 1
      dow += 1
  etf_lod2[n,:] = etf_lod2[n,:] / np.sum(etf_lod2[n,:])

# For gases: only traffic exhaust and wood combustion
gases_etf_lod1 = etf_lod1[[0,2],:]
gases_etf_lod2 = etf_lod2[[0,2],:]

#%% Save files:

topo = dict()
topo['R'] = topoR
topo['GlobOrig'] = [25497030.0, 6675781.0]
topo['GlobOrigBL'] = [25497030.0, 6675741.0]
topo['gridRot'] = 0.0
topo['dPx'] = np.array([dy, dx])

oro = dict()
oro['R'] = oroR
oro['GlobOrig'] = [25497030.0, 6675781.0]
oro['GlobOrigBL'] = [25497030.0, 6675741.0]
oro['gridRot'] = 0.0
oro['dPx'] = np.array([dy, dx])

buildID = dict()
buildID['R'] = idR
buildID['GlobOrig'] = [25497030.0, 6675781.0]
buildID['GlobOrigBL'] = [25497030.0, 6675741.0]
buildID['gridRot'] = 0.0
buildID['dPx'] = np.array([dy, dx])

source_aerosol = dict()
source_aerosol['R'] = saR
source_aerosol['GlobOrig'] = [25497030.0, 6675781.0]
source_aerosol['GlobOrigBL'] = [25497030.0, 6675741.0]
source_aerosol['gridRot'] = 0.0
source_aerosol['dPx'] = np.array([dy, dx])

source_gas = dict()
source_gas['R'] = sgR
source_gas['GlobOrig'] = [25497030.0, 6675781.0]
source_gas['GlobOrigBL'] = [25497030.0, 6675741.0]
source_gas['gridRot'] = 0.0
source_gas['dPx'] = np.array([dy, dx])



np.savez_compressed(output_path + 'topo.npz', **topo)
np.savez_compressed(output_path + 'oro.npz', **oro)
np.savez_compressed(output_path + 'building_id.npz', **buildID)
np.savez_compressed(output_path + 'aerosol_source_area.npz', **source_aerosol)
np.savez_compressed(output_path + 'gas_source_area.npz', **source_gas)

etfStr = '_emission_time_factors_'
np.savetxt(output_path + 'aerosol' + etfStr + 'lod1.csv', (etf_lod1), delimiter=',', fmt='%1.3e')
np.savetxt(output_path + 'aerosol' + etfStr + 'lod2.csv', (etf_lod2), delimiter=',', fmt='%1.3e')

np.savetxt(output_path + 'gas' + etfStr + 'lod1.csv', (gases_etf_lod1), delimiter=',', fmt='%1.3e')
np.savetxt(output_path + 'gas' + etfStr + 'lod2.csv', (gases_etf_lod2), delimiter=',', fmt='%1.3e')

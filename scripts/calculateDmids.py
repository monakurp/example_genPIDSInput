#!/usr/bin/env python3
import sys
import numpy as np
import os

'''
Description:

Author:
Mona Kurppa
mona.kurppa@helsinki.fi
Institute for Atmospheric and Earth System Research (INAR) / Physics
University of Helsinki
'''

#==========================================================#
# Input arguments:

# Number of size bins in the subrange 1 and 2:
nbin = [2, 8]

# Min & max diameters of subranges
reglim = [3.0E-9, 10.0E-9, 1.0E-6]

nbins = np.sum(nbin)

#==========================================================#
# Calculate the bin limits:
# Applies volume-ratio size distribution, i.e., the volume of particles in a size bin equals the
# volume of particles in the next smallest size bin multiplied by a constant volume ratio, ratio_d.

vlolim = np.zeros(nbins)
vhilim = np.zeros(nbins)
dmid   = np.zeros(nbins)
bin_limits = np.zeros(nbins)

# Subrange 1:
ratio_d = reglim[1] / reglim[0]
for b in range(nbin[0]):
  vlolim[b] = np.pi / 6.0 * (reglim[0] * ratio_d**(float(b) / nbin[0]))**3
  vhilim[b] = np.pi / 6.0 * (reglim[0] * ratio_d**(float(b+1) / nbin[0]))**3
  dmid[b] = np.sqrt((6.0 * vhilim[b] / np.pi)**(1.0 / 3.0) * (6.0 * vlolim[b] / np.pi)**(1.0 / 3.0))

# Subrange 1:
ratio_d = reglim[2] / reglim[1]
for b in np.arange(nbin[0], nbins, 1):
  c = b-nbin[0]
  vlolim[b] = np.pi / 6.0 * (reglim[1] * ratio_d**(float(c) / nbin[1]))**3
  vhilim[b] = np.pi / 6.0 * (reglim[1] * ratio_d**(float(c+1) / nbin[1]))**3
  dmid[b] = np.sqrt((6.0 * vhilim[b] / np.pi)**(1.0 / 3.0) * (6.0 * vlolim[b] / np.pi)**(1.0 / 3.0))

# Bin limits:
bin_limits = (6.0 * vlolim / np.pi)**(1.0 / 3.0)
bin_limits = np.append(bin_limits, reglim[-1])

#==========================================================#
# Print values on the screen

np.set_printoptions(precision=3, suppress=False)
print("----------------------------------------------------")
print("")
print("Number of bins per subrange: {}".format(nbin))
print("Min & max diameters of subranges [m]: " + str(reglim))
print("")
print("Dmid [m]: " + np.array2string(dmid, separator=', '))
print("Bin limits [m]: " + np.array2string(bin_limits, separator=', '))
print("")
print("----------------------------------------------------")

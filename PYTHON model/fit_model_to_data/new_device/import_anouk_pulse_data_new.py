import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import re

# iterate over files in directory
for root, dirs, files in os.walk('../../../raw_data/pulses/new_device/'):
    dirs.sort()
    for file in files:
        if file.endswith('processed.txt'):
            # read data
            data = np.loadtxt(os.path.join(root, file), delimiter='\t', skiprows=1, usecols=[1, 2])
            # read parameters
            setV, resetV, readV = re.findall(r'\d+', file)
            setV = int(setV)
            resetV = int(resetV)
            readV = int(readV)
            resetV *= -1
            readV = -1 * readV if readV == 500 else readV
            # plot data
            plt.plot(data[:, 0], readV / 1000 / data[:, 1], label=f'{resetV} V, {setV} V, {readV} mV')
            plt.yscale('log')

            print(f'RESET {resetV} V, SET {setV} V, READ {readV} mV: {file}')
    plt.legend()
    plt.show()

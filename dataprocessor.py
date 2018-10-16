from enum import Enum, auto
import csv
import numpy as np
from math import exp

from deformation import EngineeringStrain, Stretch, TrueStrain
from stress import EngineeringStress, TrueStress

def process_file(filename, delimiter=',', samples=-1,
                 deformation_quantity=EngineeringStrain(),
                 deformation_column=0,
                 stress_quantity=EngineeringStress(),
                 stress_column=1):
    """Processes the given .csv file containing measurement data.
    ----------
    Keyword arguments:
    filename -- The name of the input file.
    delimiter -- The string separating the columns. (default: ',')
    samples -- The number of measurement points to be read.
               If there are more points, the closest points will be thrown out,
               with respect to stretch. If all the points are to be kept,
               -1 should be passed. (default: -1)
    deformation_quantity -- The quantity of deformation.
                            (default: engineering strain)
    deformation_column -- The column index of deformation data. (default: 0)
    stress_quantity -- The quantity of stress.
                       (default: engineering stress)
    stress_column -- The column index of stress data. (default: 1)
    ----------
    Returns:
    A tuple of two lists (xdata, ydata) containing stretch and pressure data.
    """
    with open(filename, 'r') as csvfile:
        decimal = ','
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        xdata, ydata = [], []
        for row in csvreader:
            deformation = float(row[deformation_column].replace(decimal, '.'))
            stress = float(row[stress_column].replace(decimal, '.'))

            engineering_strain = deformation_quantity.get_engineering_strain(deformation)
            stretch = engineering_strain + 1
            engineering_stress = stress_quantity.get_engineering_stress(stress, stretch)

            xdata.append(engineering_strain)
            ydata.append(engineering_stress)
        
        # Throw out samples which are close to each other.
        if (samples > 2):
            while len(xdata) > samples:
                distances = [xdata[i + 2] - xdata[i] for i in range(len(xdata) - 2)]
                minindex = distances.index(min(distances)) + 1
                del xdata[minindex]
                del ydata[minindex]
        elif samples == 2:
            # Keep the first and the last samples.
            xdata = [xdata[0], xdata[-1]]
            ydata = [ydata[0], ydata[-1]]
        elif samples == 1:
            # Keep the last sample.
            xdata = [xdata[-1]]
            ydata = [ydata[-1]]

        return xdata, ydata

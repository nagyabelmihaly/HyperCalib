from enum import Enum, auto
import csv
import numpy as np

class Columns(Enum):
    STRETCH = auto()
    PRESSURE = auto()

def process_file(filename, delimiter=',', samples=-1, columns=[Columns.STRETCH, Columns.PRESSURE]):
    """Processes the given .csv file containing measurement data.
    ----------
    Keyword arguments:
    filename -- The name of the input file.
    delimiter -- The string separating the columns. (default: ',')
    samples -- The number of measurement points to be read.
               If there are more points, the closest points will be thrown out,
               with respect to stretch. If all the points are to be kept,
               -1 should be passed. (default: -1)
    columns -- A list of 'Columns' enum, specifying column order.
               (default: [stretch, pressure])
    ----------
    Returns:
    A tuple of two lists (xdata, ydata) containing stretch and pressure data.
    """
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        xdata, ydata = [], []
        for row in csvreader:
            xdata.append(float(row[columns.index(Columns.STRETCH)]))
            ydata.append(float(row[columns.index(Columns.PRESSURE)]))
        
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

from enum import Enum, auto
import csv
import numpy as np

class Columns(Enum):
    STRETCH = auto()
    PRESSURE = auto()

def process_file(filename, delimiter=',', samples=-1, columns=[Columns.STRETCH, Columns.PRESSURE]):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        xdata, ydata = [], []
        for row in csvreader:
            xdata.append(float(row[columns.index(Columns.STRETCH)]))
            ydata.append(float(row[columns.index(Columns.PRESSURE)]))
        if (samples > 0):
            while len(xdata) > samples:
                distances = [xdata[i + 2] - xdata[i] for i in range(len(xdata) - 3)]
                minindex = distances.index(min(distances)) + 1
                del xdata[minindex]
                del ydata[minindex]
        return np.asarray(xdata), np.asarray(ydata)

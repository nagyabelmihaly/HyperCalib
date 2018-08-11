import csv
import numpy as np

def generate_input(filename, size):
    def func(x):
        return x * x - 5
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        xdata = np.linspace(0, 10, size)
        ydata = func(xdata)

        for x, y in zip(xdata, ydata):
            writer.writerow({'x': x, 'y': y})

generate_input('data\\big5000.csv', 5000)
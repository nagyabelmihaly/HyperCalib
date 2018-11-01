from enum import Enum, auto
import csv
import numpy as np
from math import exp

from tkinter import messagebox

from utilities import *

from deformation import EngineeringStrain, Stretch, TrueStrain
from stress import EngineeringStress, TrueStress

class DataProcessor:
    def load_file(self, filename):
        with open(filename, 'r', newline='') as f:
            self.lines = [s.strip() for s in f.readlines()]

    def parse_csv(self, delimiter=';', decimalsep=','):
        csvreader = csv.reader(self.lines, delimiter=delimiter)
        self.raw = []
        for row in csvreader:
            self.raw.append([try_parse_float(s.replace(decimalsep, '.')) for s in row])
        self.max_column_count = max([len(row) for row in self.raw])
        self.min_column_count = min([len(row) for row in self.raw])

    def define_data(self,
                 deformation_quantity=Stretch,
                 deformation_column=1,
                 stress_quantity=TrueStress,
                 stress_column=2):
        if deformation_column < 1:
            raise ValueError('Deformation column index has to be positive.')
        if deformation_column > self.min_column_count:
            raise ValueError('Deformation column index ' + \
                str(deformation_column) + ' is out of range. ' + \
                'Number of full columns is ' + \
                str(self.min_column_count) + '.')
        if stress_column < 1:
            raise ValueError('Stress column index has to be positive.')
        if stress_column > self.min_column_count:
            raise ValueError('Stress column index ' + \
                str(stress_column) + ' is out of range. ' + \
                'Number of full columns is ' + \
                str(self.min_column_count) + '.')
        self.stretch, self.true_stress = [], []
        rowindex = 0
        for row in self.raw:
            rowindex += 1
            deformation = row[deformation_column - 1]
            stress = row[stress_column - 1]
            if not isinstance(deformation, float):
                raise ValueError('Deformation "' + deformation + \
                    '" is not a valid number in row ' + \
                    str(rowindex) + '.')
                return
            if not isinstance(stress, float):
                raise ValueError('Stress "' + stress + \
                    '" is not a valid number in row ' + \
                    str(rowindex) + '.')
                return
            stretch = deformation_quantity.to_stretch(deformation)
            true_stress = stress_quantity.to_true_stress(stress, stretch)
            self.stretch.append(stretch)
            self.true_stress.append(true_stress)
            combined = [pair for pair in zip(self.stretch, self.true_stress)]
            combined.sort(key=lambda pair : pair[0])
            self.stretch = [pair[0] for pair in combined]
            self.true_stress = [pair[1] for pair in combined]

    def limit_data(self, samples):
        self.stretch_limited, self.true_stress_limited = self.stretch.copy(), self.true_stress.copy()
        if (samples > 2):
            while len(self.stretch_limited) > samples:
                distances = [stretch_limited[i + 2] - stretch_limited[i] for i in range(len(stretch_limited) - 2)]
                minindex = distances.index(min(distances)) + 1
                del stretch_limited[minindex]
                del true_stress_limited[minindex]
        elif samples == 2:
            # Keep the first and the last samples.
            stretch_limited = [stretch_limited[0], stretch_limited[-1]]
            true_stress_limited = [true_stress_limited[0], true_stress_limited[-1]]
        elif samples == 1:
            # Keep the last sample.
            stretch_limited = [stretch_limited[-1]]
            true_stress_limited = [true_stress_limited[-1]]

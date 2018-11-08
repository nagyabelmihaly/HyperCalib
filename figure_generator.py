import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import messagebox
from scipy.optimize import minimize, BFGS
from datetime import datetime
from time import process_time
import threading

from utilities import *

from file_dialog import FileDialog
from fit_dialog import FitDialog
from plot_settings_dialog import PlotSettingsDialog
from create_report_dialog import CreateReportDialog

from dataprocessor import DataProcessor

from cod import COD
from deformation import EngineeringStrain, Stretch, TrueStrain
from stress import EngineeringStress, TrueStress

from rmsae import RMSAE
from rmsre import RMSRE
from weighted_error import WeightedError
from neo_hooke import NeoHooke
from ogden import Ogden
from mooney_rivlin import MooneyRivlin
from yeoh import Yeoh
from arruda_boyce import ArrudaBoyce
from trust_constr import TrustConstr
from pdf_generator import PdfGenerator
from tex_generator import TexGenerator

import csv

class FigureGenerator:
    def __init__(self):
        self.dp = DataProcessor()
        self.measurements = ['Treloar', '2012', '2015']
        self.separators = [[',', '.'], [';', ','], [';', ',']]
        self.quantities = [[Stretch, EngineeringStress],
                           [EngineeringStrain, EngineeringStress],
                           [EngineeringStrain, EngineeringStress]]

    def gen_fig(self, exp, defmodes, model, params, filename):
        exp -= 1
        measurement = self.measurements[exp]
        filenames = ['data/{}/ut.csv'.format(measurement),
                     'data/{}/et.csv'.format(measurement),
                     'data/{}/ps.csv'.format(measurement)]
        xdatas, ydatas = [None] * 3, [None] * 3
        for defmode in range(3):
            self.dp.load_file(filenames[defmode])
            self.dp.parse_csv(self.separators[exp][0], self.separators[exp][1])
            self.dp.define_data(self.quantities[exp][0], 1, self.quantities[exp][1], 2)
            xdatas[defmode], ydatas[defmode] = self.dp.stretch, self.dp.true_stress
        texgen = TexGenerator()
        texgen.xdatas = xdatas
        texgen.ydatas = ydatas
        texgen.model = model
        texgen.params = params
        texgen.plot_defmode = defmodes
        texgen.plot_error = COD
        texgen.plot_deformation = Stretch
        texgen.plot_stress = EngineeringStress
        texgen.filename = os.getcwd() + '\\Figures\\' + filename
        texgen.generate()

generator = FigureGenerator()

# MATLAB
generator.gen_fig(1, [True, False, False], Ogden(2), [0.2933, 7.963e-7, 2.087, 9.299], 'example1a')
generator.gen_fig(1, [True, False, False], Ogden(2), [0.1728, 4.0982e-10, 2.6132, 13.0608], 'example1b') # sqp
generator.gen_fig(2, [True, True, False], Ogden(3), [0.4205, 0.6451, 0.3893, 4.149, -2.922, 0.3594], 'example2a')
generator.gen_fig(2, [True, True, False], Ogden(3), [0.8355, 0.2874, 0.2872, -2.6510, 3.8546, 3.8546], 'example2b') # active set
generator.gen_fig(2, [True, True, True], ArrudaBoyce(), [2.302, 4.752], 'example3a')
generator.gen_fig(2, [True, True, True], ArrudaBoyce(), [2.3023, 4.7522], 'example3b') # sqp
generator.gen_fig(3, [True, True, True], ArrudaBoyce(), [0.4262, 3.705], 'example4a')
generator.gen_fig(3, [True, True, True], ArrudaBoyce(), [0.4262, 3.7050], 'example4b') # sqp

# ABAQUS
generator.gen_fig(3, [True, False, True], MooneyRivlin(), [0.1819, 0.06142], 'example5a')
generator.gen_fig(3, [True, False, True], MooneyRivlin(), [0.1868, 0.0478], 'example5b')
generator.gen_fig(3, [True, True, True], Yeoh(), [0.2383, -0.002552, 0.0002013], 'example6a')
generator.gen_fig(3, [True, True, True], Yeoh(), [0.2316, -0.001411, 0.0001703], 'example6b')
generator.gen_fig(1, [True, True, False], Ogden(3), [0.007412, 0.3944, 0.0008749, -2.212, 1.382, 5.725], 'example7a')
generator.gen_fig(1, [True, True, False], Ogden(3), [0.4, 1.435e-3, 6.448e-3, 1.288, 5.467, -2.283], 'example7b')
generator.gen_fig(1, [True, True, True], Ogden(3), [0.003921, 0.3727, 0.01467, 4.898, 1.304, -1.86], 'example8a')
generator.gen_fig(1, [True, True, True], Ogden(3), [0.3586, 1.007e-3, 2.709e-2, 1.514, 5.625, -1.56], 'example8b')

# ANSYS
generator.gen_fig(1, [False, False, True], NeoHooke(), [0.34], 'example9a')
generator.gen_fig(1, [False, False, True], NeoHooke(), [0.34], 'example9b')
generator.gen_fig(2, [False, True, True], Ogden(2), [0.6904, 0.6904, 2.657, 2.657], 'example10a')
generator.gen_fig(2, [False, True, True], Ogden(2), [8.478e-5, 8.478e-5, 8.7, 8.725], 'example10b')
generator.gen_fig(2, [True, False, True], Yeoh(), [0.6575, 0.03945, -0.0007198], 'example11a')
generator.gen_fig(2, [True, False, True], Yeoh(), [0.51992, 0.072646, -0.0017948], 'example11b')
generator.gen_fig(3, [True, True, True], ArrudaBoyce(), [0.4017, 3.609], 'example12a')
generator.gen_fig(3, [True, True, True], ArrudaBoyce(), [0.41434, -3.5765], 'example12b')
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

class GuiSupport:
    """Implements GUI functionality and provides a binding
    between front-end and back-end."""
    def Init(self, top, gui):
        """Initializes the window with the widgets inside."""
        self.w = gui
        self.top_level = top
        self.root = top

        # Set window icon.
        self.root.iconbitmap('hypercalib.ico')

        # Initialize data variables.
        self.xdatas = [None] * 3
        self.ydatas = [None] * 3
        self.plot_xdatas = [None] * 3
        self.plot_ydatas = [None] * 3

        # Initialize filename variables.
        self.filenames = [None] * 3

        # Initialize model and function variables.
        self.model = None

        # Initialize parameters variable.
        self.params = None

        # Initialize error values variable.
        self.error_values = None
        self.errors = [None] * 3
        self.fit_error = None

        # Initialize a Figure for plotting the model.
        fig = Figure(figsize=(5, 2), dpi=100)
        self.plt = fig.add_subplot(111)
        fig.set_tight_layout(True)
    
        # Initialize a canvas to draw the Figure.
        self.canvas = FigureCanvasTkAgg(fig, self.w.FramePlot)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Initialize navigation toolbar.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.w.FrameNavigationPlot)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initialize a Figure for plotting the error function.
        figErr = Figure(figsize=(5, 2), dpi=100)
        self.pltErr = figErr.add_subplot(111)
        figErr.set_tight_layout(True)
    
        # Initialize a canvas to draw the Figure.
        self.canvasErr = FigureCanvasTkAgg(figErr, self.w.FrameError)
        self.canvasErr.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Initialize navigation toolbar.
        self.toolbarErr = NavigationToolbar2Tk(self.canvasErr, self.w.FrameNavigationError)
        self.canvasErr._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initialize plotting utilities.
        self.titles = ['UT', 'ET', 'PS']
        self.data_colors = ['b', 'r', 'g']
        self.fit_colors = ['#8080ff', '#ff8080', '#80ff80']

        # Initialize plot configuration variables.
        self.plot_defmode = [False for i in range(3)]
        self.plot_error = COD
        self.plot_deformation = Stretch
        self.plot_stress = TrueStress

        # TODO: delete
        # Load data to speedup developement process.

        if False:
            measurements = ['Treloar', '2012', '2015']
            separators = [[',', '.'],
                          [';', ','],
                          [';', ',']]
            quantities = [[Stretch, EngineeringStress],
                          [EngineeringStrain, EngineeringStress],
                          [EngineeringStrain, EngineeringStress]]
            weightdata = [([1, 0, 0], 'UT'),
                       ([0, 1, 0], 'ET'),
                       ([0, 0, 1], 'PS'),
                       ([1, 1, 0], 'UTET'),
                       ([0, 1, 1], 'ETPS'),
                       ([1, 0, 1], 'PSUT'),
                       ([1, 1, 1], 'UTETPS')]
            models = [Ogden(1), Ogden(2), Ogden(3), NeoHooke(), MooneyRivlin(), Yeoh(), ArrudaBoyce()]
            error_functions = [RMSAE, RMSRE]
            for measurement, separator, quantity in zip(measurements, separators, quantities):
                dp = DataProcessor()
                self.filenames = ['data/{}/ut.csv'.format(measurement),
                                  'data/{}/et.csv'.format(measurement),
                                  'data/{}/ps.csv'.format(measurement)]
                for defmode in range(3):
                    dp.load_file(self.filenames[defmode])
                    dp.parse_csv(separator[0], separator[1])
                    dp.define_data(quantity[0], 1, quantity[1], 2)
                    self.xdatas[defmode], self.ydatas[defmode] = dp.stretch, dp.true_stress
                    self.plot_defmode[defmode] = True
                for model in models:
                    self.model = model
                    for error_function in error_functions:
                        self.params = None                        
                        errors = [error_function(model.getfunc(defmode), model.getjac(defmode),
                                                 model.gethess(defmode), self.xdatas[defmode],
                                                 self.ydatas[defmode]) for defmode in range(3)]
                        self.update_plot_settings()
                        for weights, weightname in weightdata:
                            self.weighted_error = WeightedError(errors, weights)

                            method = TrustConstr()
                            method.objfunc = self.weighted_error.objfunc
                            method.x0 = [1.0] * model.paramcount
                            method.constraint = model.constraint
                            if isinstance(model, ArrudaBoyce):
                                method.jac = '2-point'
                                method.calcJac = False
                            else:
                                method.jac = self.weighted_error.jac
                                method.calcJac = True
                            method.hess = BFGS()
                            method.calcHess = False

                            print("""Measurement: {}
Deformation type: {}
Hyperelastic model: {}
Error function: {}""".format(measurement,
                             weightname,
                             model.name,
                             error_function.name))
                            self.error_values = []
                            result = method.minimize(self.update_model)
                            print('Model has been fitted.')

                            pdfgen = PdfGenerator()
                            pdfgen.xdatas = self.xdatas
                            pdfgen.ydatas = self.ydatas
                            pdfgen.model = model
                            pdfgen.params = result.x
                            pdfgen.plot_defmode = self.plot_defmode
                            pdfgen.plot_error = error_function
                            pdfgen.plot_deformation = quantity[0]
                            pdfgen.plot_stress = quantity[1]
                            pdfgen.filenames = self.filenames
                            pdfgen.weights = weights
                            pdfgen.method = method
                            pdfgen.fit_error = error_function
                            pdfgen.filename = os.getcwd() + '/Reports/{}-{}-{}-{}'.format( \
                                measurement, weightname, model.name, error_function.shortname)
                        
                            pdfgen.generate()

        # Load Treloar data
        dp = DataProcessor()
        measurement = '2012'
        self.filenames = ['data/{}/ut.csv'.format(measurement),
                            'data/{}/et.csv'.format(measurement),
                            'data/{}/ps.csv'.format(measurement)]
        #model = Ogden(2)
        for defmode in range(3):
            dp.load_file(self.filenames[defmode])
            dp.parse_csv(';', ',')
            dp.define_data(EngineeringStrain, 1, EngineeringStress, 2)
            self.xdatas[defmode], self.ydatas[defmode] = dp.stretch, dp.true_stress
            self.plot_defmode[defmode] = True
            #self.errors[defmode] = RMSRE(model.getfunc(defmode), model.getjac(defmode),
            #                             model.gethess(defmode), self.xdatas[defmode],
            #                             self.ydatas[defmode])

        #weighted_error = WeightedError(self.errors, [1, 0, 0])
        #print('weighted error = ' + str(weighted_error.objfunc([1, 1, 1, 1])))
        self.w.ButtonFitModel['state'] = tk.NORMAL

        # Generate Tex output
        #texgen = TexGenerator()
        #texgen.xdatas = self.xdatas
        #texgen.ydatas = self.ydatas
        #texgen.model = Ogden(2)
        #texgen.params = [8.478e-5, 8.4781e-5, 8.7005, 8.7246]
        #texgen.plot_defmode = [False, True, True]
        #texgen.plot_error = RMSRE
        #texgen.plot_deformation = Stretch
        #texgen.plot_stress = EngineeringStress
        #texgen.filenames = self.filenames
        #texgen.filename = os.getcwd() + '\\example6b'
                        
        #texgen.generate()


        #rmsae = RMSAE(model.ut, model.ut_jac, model.ut_hess, self.xdatas[0], self.ydatas[0])
        #print('Abaqus RMSAE = ', rmsae.objfunc([6.5991E-3, 0.42654, 5.3619, -4.3558]))
        #print('HyperCalib RMSAE = ', rmsae.objfunc([0.04209, 0.3044, 4.253, -0.588]))
        #rmsre = RMSRE(model.ut, model.ut_jac, model.ut_hess, self.xdatas[0], self.ydatas[0])
        #print('Abaqus RMSRE = ', rmsre.objfunc([6.5991E-3, 0.42654, 5.3619, -4.3558]))
        #print('HyperCalib RMSRE = ', rmsre.objfunc([0.03719, 0.3292, 4.343, -0.4907]))

        #dp = DataProcessor()
        #dp.load_file('data/TRELOARUT.csv')
        #dp.parse_csv(',', '.')
        #dp.define_data(Stretch(), 1, EngineeringStress(), 2)
        #self.xdatas[0], self.ydatas[0] = dp.stretch, dp.true_stress
        #self.plot_defmode[0] = True
        #self.w.ButtonFitModel['state'] = tk.NORMAL
        
        self.update_plot_settings()

    def SetTkVar(self):      
        """Sets the TkInter variables referenced by the widgets."""
        pass

    def ButtonProcess_Click(self):
        FileDialog(self.file_loaded)

    def file_loaded(self, defmode, stretch, true_stress, filename):
        # Load read data into actual data variables.
        if defmode == 'UT':
            defmode_index = 0
        if defmode == 'ET':
            defmode_index = 1
        if defmode == 'PS':
            defmode_index = 2
        self.xdatas[defmode_index] = stretch
        self.ydatas[defmode_index] = true_stress

        # Enable this deformation mode in GUI.
        self.plot_defmode[defmode_index] = True
        self.update_plot_settings()

        # Enable model fitting.
        self.w.ButtonFitModel['state'] = tk.NORMAL

        # Save filename.
        self.filenames[defmode_index] = filename

    def ButtonFitModel_Click(self):
        FitDialog(self.xdatas, self.ydatas, self.start_fit)

    def name(self, i):
        if i == 0:
            return 'UT'
        if i == 1:
            return 'ET'
        if i == 2:
            return 'PS'

    def start_fit(self, model, error_function, errors, weights, weighted_error, method):
        # Disable buttons.
        self.w.ButtonProcess['state'] = tk.DISABLED
        self.w.ButtonFitModel['state'] = tk.DISABLED
        self.w.ButtonCreateReport['state'] = tk.DISABLED

        self.model = model
        self.weights = weights
        self.weighted_error = weighted_error
        self.method = method
        self.fit_error = error_function

        self.functions = [model.ut, model.et, model.ps]

        defcount = sum(weights[i] > 0 for i in range(3))
        if defcount == 1:
            defindex = weights.index(next(w for w in weights if w != 0))
            deformation = self.name(defindex)
        else:
            deformation = '; '.join([str(weights[i]) + 'x' + self.name(i) for i in range(3) if weights[i] != 0.0])
        self.print('Starting curve fitting...')
        self.print("""Deformation type: {0}
Hyperelastic model: {1}
Error function: {2}
Optimization method: {3}
{4}""".format(deformation,
              model.name,
              error_function.name,
              method.name,
              method.print_params()), False, 5)
        
        thread = threading.Thread(target=self.fit_model)
        thread.setDaemon(True)
        thread.start()

    def ButtonPlotSettings_Click(self):
        loaded_defmode = [self.xdatas[i] is not None for i in range(3)]
        PlotSettingsDialog(self.update_plot_settings, loaded_defmode, self.plot_defmode,
                           self.plot_error, self.plot_deformation, self.plot_stress)

    def ButtonCreateReport_Click(self):
        CreateReportDialog(self.xdatas, self.ydatas, self.model, 
                           self.params, self.plot_defmode, self.plot_error,
                           self.plot_deformation, self.plot_stress,
                           self.filenames, self.weights, self.method,
                           self.fit_error)

    def fit_model(self):
        start_time = process_time()

        self.error_values = []
        result = self.method.minimize(self.update_model)

        self.update_model(result.x, result)
        self.update_plot_settings()
        end_time = process_time()
        elapsed_time = end_time - start_time
        self.print('Model has been fitted.')
        self.print("""Error: {0:.4g}
Elapsed time: {1:.4f} s
{2}""".format(result.fun, elapsed_time, result.print), False, 5)

        # Enable buttons.
        self.w.ButtonProcess['state'] = tk.NORMAL
        self.w.ButtonFitModel['state'] = tk.NORMAL
        self.w.ButtonCreateReport['state'] = tk.NORMAL

    def update_model(self, xk, state=None):
        self.params = xk
        err_val = self.weighted_error.objfunc(xk)
        self.error_values.append(err_val)
        if state is not None and state.niter % 100 == 0:
            self.update_plot_settings()

    def update_plot_settings(self, plot_defmode=None, error=None, deformation_quantity=None, stress_quantity=None):
        if plot_defmode is None: plot_defmode = self.plot_defmode
        if error is None: error = self.plot_error
        if deformation_quantity is None: deformation_quantity = self.plot_deformation
        if stress_quantity is None: stress_quantity = self.plot_stress

        self.plot_defmode = plot_defmode

        # Refresh plotted values and error instances if a change is present.
        changed = False
        for defmode in range(3):
            changed |= self.xdatas[defmode] is not None and self.plot_xdatas[defmode] is None
            changed |= self.model is not None and self.errors[defmode] is None
        changed |= self.plot_deformation != deformation_quantity
        changed |= self.plot_stress != stress_quantity
        changed |= self.plot_error != error
        
        if changed:
            for defmode in range(3):
                if self.xdatas[defmode] is None:
                    continue
                self.plot_xdatas[defmode] = np.zeros(len(self.xdatas[defmode]))
                self.plot_ydatas[defmode] = np.zeros(len(self.ydatas[defmode]))
                for i in range(len(self.xdatas[defmode])):
                    stretch = self.xdatas[defmode][i]
                    true_stress = self.ydatas[defmode][i]
                    self.plot_xdatas[defmode][i] = deformation_quantity.from_stretch(stretch)
                    self.plot_ydatas[defmode][i] = stress_quantity.from_true_stress(true_stress, stretch)
                if self.model is None:
                    self.errors[defmode] = None
                else:
                    self.errors[defmode] = error(self.model.getfunc(defmode),
                                                 self.model.getjac(defmode),
                                                 self.model.gethess(defmode),
                                                 self.xdatas[defmode],
                                                 self.ydatas[defmode])

        # Update self variables.
        self.plot_error = error
        self.plot_deformation = deformation_quantity
        self.plot_stress = stress_quantity  

        # Replot the figure.
        self.plot()

    def plot(self):
        self.plt.clear()
        is_empty = True

        for defmode in range(3):
            # Plot read data as points.
            if self.xdatas[defmode] is None or not self.plot_defmode[defmode]:
                continue
            
            xdata = self.plot_xdatas[defmode]
            ydata = self.plot_ydatas[defmode]

            label = self.titles[defmode]
            if self.errors[defmode] is not None and self.params is not None:
                label += ' - {} = {:.4g}'.format(self.plot_error.shortname, self.errors[defmode].objfunc(self.params))
            self.plt.plot(xdata, ydata, marker='o', linestyle='',
                          color=self.data_colors[defmode], label=label)
            is_empty = False
        
            # Plot fitted models as curves.
            if self.params is None:
                continue
            xlin = np.linspace(min(xdata), max(xdata))
            y = np.zeros(len(xlin))
            for i in range(len(xlin)):
                stretch = self.plot_deformation.to_stretch(xlin[i])
                y[i] = self.plot_stress.from_true_stress(self.model.getfunc(defmode)(stretch, *self.params), stretch)
            self.plt.plot(xlin, y, marker='', linestyle='-', color=self.fit_colors[defmode])

        if self.params is None:
            # Clear Text widget.
            set_text(self.w.TextState, '')
            pass
        else:
            # Write parameters to Text widget.
            set_text(self.w.TextState, '\n'.join(['{} = {:.4g}' \
                .format(self.model.paramnames[i], self.params[i]) \
                for i in range(self.model.paramcount)]))
            pass

        if not is_empty:        
            self.plt.set_xlabel(self.plot_deformation.name)
            self.plt.set_ylabel(self.plot_stress.name)
            self.plt.legend()
            self.canvas.draw()
            self.toolbar.update()
           
        # Plot error function.
        self.pltErr.clear()
        if self.error_values is not None:
            self.pltErr.plot(range(len(self.error_values)), self.error_values, '-k')
            self.pltErr.set_xlabel('iterations')
            self.pltErr.set_ylabel('error (logarithmic)')
            self.pltErr.set_yscale('log')
            self.canvasErr.draw()
            self.toolbarErr.update()    

    def print(self, string, show_time=True, indent=0):
        """Prints the given text to the logger
        with a time stamp and a new line at end."""
        text = self.w.TextLog
        if show_time:
            beginning = datetime.now().strftime("[%H:%M:%S] ") + ' ' * indent
        else:
            beginning = ' ' * (18 + indent)
        line_sep = '\n' + ' ' * (18 + indent)
        string = line_sep.join(string.split('\n'))
        add_text(text, beginning + string + '\n')
        text.see(tk.END)

# gui.py bindings.
global gui_support
gui_support = GuiSupport()

def init(top, gui, *args, **kwargs):
    gui_support.Init(top, gui)

def set_Tk_var():
    global dummy
    dummy = tk.StringVar()
    gui_support.SetTkVar()

def ButtonProcess_Click():
    gui_support.ButtonProcess_Click()

def ButtonFitModel_Click():
    gui_support.ButtonFitModel_Click()

def ButtonPlotSettings_Click():
    gui_support.ButtonPlotSettings_Click()

def ButtonCreateReport_Click():
    gui_support.ButtonCreateReport_Click()
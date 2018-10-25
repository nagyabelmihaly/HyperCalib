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

from dataprocessor import DataProcessor
from deformation import EngineeringStrain, Stretch, TrueStrain
from stress import EngineeringStress, TrueStress

class GuiSupport:
    """Implements GUI functionality and provides a binding
    between front-end and back-end."""
    def Init(self, top, gui):
        """Initializes the window with the widgets inside."""
        self.w = gui
        self.top_level = top
        self.root = top

        # Enter full-screen mode by default.
        top.state('zoomed')

        # Initialize data variables.
        self.xdatas = [None] * 3
        self.ydatas = [None] * 3

        # Initialize parameters variable.
        self.params = None

        # Initialize error values variable.
        self.error_values = None
        self.errors = [None] * 3

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

        # Initialize widget references.
        self.plot_checkbuttons = [self.w.CheckbuttonPlotUT, self.w.CheckbuttonPlotET, self.w.CheckbuttonPlotPS]

        # TODO: delete
        # Load data to speedup developement process.
        dp = DataProcessor()
        dp.load_file('data/ut.csv')
        dp.parse_csv()
        dp.define_data(EngineeringStrain(), 1, EngineeringStress(), 2)
        self.xdatas[0], self.ydatas[0] = dp.stretch, dp.true_stress
        self.isPlot[0].set(True)
        dp.load_file('data/et.csv')
        dp.parse_csv()
        dp.define_data(EngineeringStrain(), 1, EngineeringStress(), 2)
        self.xdatas[1], self.ydatas[1] = dp.stretch, dp.true_stress
        self.isPlot[1].set(True)
        dp.load_file('data/planar.csv')
        dp.parse_csv()
        dp.define_data(EngineeringStrain(), 1, EngineeringStress(), 2)
        self.xdatas[2], self.ydatas[2] = dp.stretch, dp.true_stress
        self.isPlot[2].set(True)
        self.w.ButtonFitModel['state'] = tk.NORMAL
        for defmode_index in range(3):
            self.plot_checkbuttons[defmode_index]['state'] = tk.NORMAL
        self.plot()

        #dp = DataProcessor()
        #dp.load_file('data/TRELOARUT.csv')
        #dp.parse_csv(',', '.')
        #dp.define_data(Stretch(), 1, EngineeringStress(), 2)
        #self.xdatas[0], self.ydatas[0] = dp.stretch, dp.true_stress
        #self.isPlot[0].set(True)
        #self.w.ButtonFitModel['state'] = tk.NORMAL
        #self.plot_checkbuttons[0]['state'] = tk.NORMAL
        #self.plot()

    def SetTkVar(self):      
        """Sets the TkInter variables referenced by the widgets."""
        self.isPlot = [tk.BooleanVar() for i in range(3)]

    def is_fit_changed(self):
        self.plot()

    def ButtonProcess_Click(self):
        FileDialog(self.file_loaded)

    def file_loaded(self, defmode, stretch, true_stress):
        if defmode == 'UT':
            defmode_index = 0
        if defmode == 'ET':
            defmode_index = 1
        if defmode == 'PS':
            defmode_index = 2
        self.xdatas[defmode_index] = stretch
        self.ydatas[defmode_index] = true_stress

        # Enable this deformation mode in GUI.f
        self.plot_checkbuttons[defmode_index]['state'] = tk.NORMAL
        self.isPlot[defmode_index].set(True)
        self.plot()

        # Enable model fitting.
        self.w.ButtonFitModel['state'] = tk.NORMAL

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
        self.model = model
        self.error_function = error_function
        self.errors = errors
        self.weights = weights
        self.weighted_error = weighted_error
        self.method = method

        self.functions = [model.ut, model.et, model.ps]

        defcount = sum(weights[i] > 0 for i in range(3))
        if defcount == 1:
            defindex = weights.index(next(w for w in weights if w != 0))
            deformation = self.name(defindex)
        else:
            deformation = '; '.join([str(weights[i]) + 'x' + self.name(i) for i in range(3) if weights[i] != 0.0])
        self.print("""Starting curve fitting...
Deformation type: {0}
Hyperelastic model: {1}
Error function: {2}
Optimization method: {3}
{4}""".format(deformation,
              model.name,
              error_function.name,
              method.name,
              method.print_params()))
        
        thread = threading.Thread(target=self.fit_model)
        thread.setDaemon(True)
        thread.start()

    def fit_model(self):
        start_time = process_time()

        self.error_values = []
        result = self.method.minimize(self.update_model)

        self.update_model(result.x, result)
        self.plot()
        end_time = process_time()
        elapsed_time = end_time - start_time
        self.print("""Model has been fitted.
Error: {0:.4g}
Elapsed time: {1:.4f} s
{2}""".format(result.fun, elapsed_time, result.print))

    def update_model(self, xk, state=None):
        self.params = xk
        err_val = self.weighted_error.objfunc(xk)
        self.error_values.append(err_val)
        if state is not None and state.niter % 100 == 0:
            self.plot()

    def plot(self):
        self.plt.clear()
        is_empty = True

        for defmode in range(3):
            # Plot read data as points.
            if self.xdatas[defmode] is None or not self.isPlot[defmode].get():
                continue
            xdata = self.xdatas[defmode]
            ydata = self.ydatas[defmode]
            label = self.titles[defmode]
            if self.errors[defmode] is not None and self.params is not None:
                label += ' - {}={:.4g}'.format(self.error_function.shortname, self.errors[defmode].objfunc(self.params))
            self.plt.plot(xdata, ydata, marker='o', linestyle='', color=self.data_colors[defmode], label=label)
            is_empty = False
        
            # Plot fitted models as curves.
            if self.params is None:
                continue
            xlin = np.linspace(min(xdata), max(xdata))
            self.plt.plot(xlin, self.functions[defmode](xlin, *self.params),
                          marker='', linestyle='-', color=self.fit_colors[defmode])    

        if self.params is None:
            # Clear Text widget.
            set_text(self.w.TextState, '')
            pass
        else:
            # Write parameters and weighted error to Text widget.
            set_text(self.w.TextState, '\n'.join(['{} = {:.4g}' \
                .format(self.model.paramnames[i], self.params[i]) \
                for i in range(self.model.paramcount)]) + \
                '\n\n{} = {:.4f}'.format(self.error_function.name,
                                         self.error_function.sign * self.weighted_error.objfunc(self.params)))
            pass

        if not is_empty:        
            self.plt.set_xlabel('stretch')
            self.plt.set_ylabel('true stress')
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

    def print(self, string):
        """Prints the given text to the logger
        with a time stamp and a new line at end."""
        text = self.w.TextLog
        time = datetime.now().strftime("[%H:%M:%S] ")
        line_sep = '\n' + ' ' * 18
        string = line_sep.join(string.split('\n'))
        add_text(text, time + string + '\n')
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
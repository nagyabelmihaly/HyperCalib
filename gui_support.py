import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib import style
style.use('ggplot')
import tkinter as tk
from tkinter import messagebox
import threading
from scipy.optimize import minimize, BFGS

import dataprocessor
from errorfuncs import MSE, MSRE, WeightedError

from ogden1 import Ogden1
from ogden2 import Ogden2
from ogden3 import Ogden3
from neo_hooke import NeoHooke
from mooney_rivlin import MooneyRivlin

class GuiSupport:

    def Init(self, top, gui):
        self.w = gui
        self.top_level = top
        self.root = top

        # Initialize models.
        self.models = [Ogden1(), Ogden2(), Ogden3(), NeoHooke(), MooneyRivlin()]
        self.model = None

        # Initialize errors.
        self.error_functions = [MSE, MSRE]
        self.error_function = None
        self.errors = [None] * 3
        self.weighted_error = None
    
        # Initialize list of available models.
        self.w.TComboboxModel['values'] = [m.name for m in self.models]
        self.w.TComboboxModel.bind('<<ComboboxSelected>>', self.ComboboxModelSelected)
        self.w.TComboboxModel.current(0)

        # Initialize list of available error functions.
        self.w.TComboboxError['values'] = [e.name for e in self.error_functions]
        self.w.TComboboxError.bind('<<ComboboxSelected>>', self.ComboboxErrorSelected)
        self.w.TComboboxError.current(0)

        # Initialize functions and derivatives.
        self.ComboboxModelSelected(None)

        # Initialize error functions.
        self.ComboboxErrorSelected(None)

        # Initialize plotting utilities.
        self.titles = ['UT', 'ET', 'PS']
        self.data_markers = ['bo', 'ro', 'go']
        self.fit_markers = ['b-', 'r-', 'g-']

        # Initialize widget references. 
        self.fit_checkbuttons = [self.w.CheckbuttonFitUT, self.w.CheckbuttonFitET, self.w.CheckbuttonFitPS]
        self.weight_entries = [self.w.EntryUT, self.w.EntryET, self.w.EntryPS]
        self.plot_checkbuttons = [self.w.CheckbuttonPlotUT, self.w.CheckbuttonPlotET, self.w.CheckbuttonPlotPS]

        # Initialize default texts.
        self.set_entry(self.w.EntryFilename, 'data/TRELOARUT.csv')
        self.set_entry(self.w.EntryDelimiter, ',')

        # Initialize a Figure for plotting.
        fig = Figure(figsize=(5, 2), dpi=100)
        self.plt = fig.add_subplot(111)
        fig.set_tight_layout(True)
    
        # Initialize a canvas to draw the Figure.
        self.canvas = FigureCanvasTkAgg(fig, self.w.FramePlot)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Initialize navigation toolbar.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.w.FrameNavigation)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add animation to plot interactively.
        self.ani = animation.FuncAnimation(fig, self.plot, interval=500)

        # Initialize data variables.
        self.xdatas = [None] * 3
        self.ydatas = [None] * 3

        # Initialize parameters variable.
        self.params = None

    def SetTkVar(self):        
        self.radiovar = tk.IntVar()
        self.isFit = [tk.BooleanVar() for i in range(3)]
        self.isPlot = [tk.BooleanVar() for i in range(3)]

    def ButtonProcess_Click(self):
        # Read file parsing parameters from GUI.
        filename = self.w.EntryFilename.get()
        samplesString = self.w.EntrySamples.get()
        if self.is_empty_or_whitespace(samplesString):
            samples = -1
        else:
            try:
                samples = int(samplesString)
            except ValueError:
                messagebox.showerror('Error', 'Invalid number of samples.')
                return
        delimiter = self.w.EntryDelimiter.get().strip()
        if len(delimiter) != 1:
            messagebox.showerror('Error', 'Delimiter must be a 1-character string.')

        # Start a background thread to process file.
        thread_dataproc = threading.Thread(target=self.process_data, args=(filename, delimiter, samples))
        thread_dataproc.setDaemon(True)
        thread_dataproc.start()
        
    def ComboboxModelSelected(self, eventObj):
        current_model = self.w.TComboboxModel.current()
        if self.model == self.models[current_model]:
            return
        self.params = None
        self.model = self.models[current_model]
        self.functions = [self.model.ut, self.model.et, self.model.ps]
        self.jacobians = [self.model.ut_jac, self.model.et_jac, self.model.ps_jac]
        self.hessians = [self.model.ut_hess, self.model.et_hess, self.model.ps_hess]

    def ComboboxErrorSelected(self, evetObj):
        current_error_function = self.w.TComboboxError.current()
        if self.error_function == self.error_functions[current_error_function]:
            return
        self.params = None
        self.error_function = self.error_functions[current_error_function]

    def ButtonFitModel_Click(self):
        # Start a background thread to fit model parameters.
        thread_fitmodel = threading.Thread(target=self.fit_model)
        thread_fitmodel.setDaemon(True)
        thread_fitmodel.start()

    def process_data(self, filename, delimiter, samples):
        defmode = self.radiovar.get()
        try:
            self.xdatas[defmode], self.ydatas[defmode] = dataprocessor.process_file(filename, delimiter=delimiter, samples=samples)
        except FileNotFoundError:
            messagebox.showerror('Error', 'File not found.')
            return
        except ValueError:
            messagebox.showerror('Error', 'Invalid file format.')
            return
    
        # Enable this deformation mode in GUI.
        self.fit_checkbuttons[defmode].config(state=tk.NORMAL)
        self.isFit[defmode].set(True)
        self.weight_entries[defmode].config(state=tk.NORMAL)
        self.set_entry(self.weight_entries[defmode], '1')
        self.w.ButtonFitModel.config(state=tk.NORMAL)
        self.plot_checkbuttons[defmode].config(state=tk.NORMAL)
        self.isPlot[defmode].set(True)

    def fit_model(self):
        try:
            weights = [float(self.weight_entries[i].get().strip()) \
                if self.xdatas[i] is not None and self.isFit[i].get() \
                else 0.0 for i in range(3)]
        except ValueError:
            messagebox.showerror('Error', 'Weights must be real numbers.')
            return
        self.errors = [self.error_function(self.functions[i], self.jacobians[i], self.hessians[i],
                           self.xdatas[i], self.ydatas[i]) \
            if self.xdatas[i] is not None else None for i in range(3)]
        self.weighted_error = WeightedError(self.errors, weights)
        minimize(self.weighted_error.objfunc, [1.0] * self.model.paramcount,
                 method='trust-constr',
                 constraints=self.model.constraint(),
                 jac="2-point", hess=BFGS(),
                 callback=self.update_model,
                 options={'maxiter':10000, 'disp': True})

    def update_model(self, xk, state):
        self.params = xk

    def plot(self, i):
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
                label += ' - error={:.4f}'.format(self.errors[defmode].objfunc(self.params))
            self.plt.plot(xdata, ydata, self.data_markers[defmode], label=label)
            is_empty = False
        
            # Plot fitted models as curves.
            if self.params is None:
                continue
            xlin = np.linspace(min(xdata), max(xdata))
            self.plt.plot(xlin, self.functions[defmode](xlin, *self.params), self.fit_markers[defmode])    

        if self.params is None:
            # Clear Text widget.
            self.set_text(self.w.TextParameters, '')
        else:
            # Write parameters and weighted error to Text widget.
            self.set_text(self.w.TextParameters, '\n'.join(['{} = {}' \
                .format(self.model.paramnames[i], self.params[i]) \
                for i in range(self.model.paramcount)]) + \
                '\n\nerror = {}'.format(self.weighted_error.objfunc(self.params)))

        if not is_empty:        
            self.plt.set_xlabel('stretch')
            self.plt.set_ylabel('pressure')
            self.plt.legend()
            self.canvas.draw()
            self.toolbar.update()

    def set_entry(self, entry, value):
        entry.delete(0, tk.END)
        entry.insert(tk.END, value)

    def set_text(self, text, value):
        text.config(state=tk.NORMAL)
        text.delete(1.0, tk.END)
        text.insert(tk.END, value)
        text.config(state=tk.DISABLED)

    def is_empty_or_whitespace(self, string):
        if not string:
            return True    
        return string.isspace()

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
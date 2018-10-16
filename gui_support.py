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
from datetime import datetime
from time import process_time

import dataprocessor
from deformation import EngineeringStrain, Stretch, TrueStrain
from stress import EngineeringStress, TrueStress

from mse import MSE
from msre import MSRE
from weighted_error import WeightedError

from ogden1 import Ogden1
from ogden2 import Ogden2
from ogden3 import Ogden3
from neo_hooke import NeoHooke
from mooney_rivlin import MooneyRivlin

from trust_constr import TrustConstr
from cobyla import Cobyla
from slsqp import Slsqp

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

        # Initialize deformation and stress quantities.
        self.deformation_quantities = [EngineeringStrain(), Stretch(), TrueStrain()]
        self.stress_quantities = [EngineeringStress(), TrueStress()]

        # Initialize models.
        self.models = [Ogden1(), Ogden2(), Ogden3(), NeoHooke(), MooneyRivlin()]
        self.model = None

        # Initialize errors.
        self.error_functions = [MSE, MSRE]
        self.error_function = None
        self.errors = [None] * 3
        self.weighted_error = None

        # Initialize optimization methods.
        self.methods = [TrustConstr(), Cobyla(), Slsqp()]
        self.method = None
    
        # Initialize list of available models.
        self.w.TComboboxModel['values'] = [m.name for m in self.models]
        self.w.TComboboxModel.bind('<<ComboboxSelected>>', self.ComboboxModelSelected)
        self.w.TComboboxModel.current(0)

        # Initialize list of available error functions.
        self.w.TComboboxError['values'] = [e.name for e in self.error_functions]
        self.w.TComboboxError.bind('<<ComboboxSelected>>', self.ComboboxErrorSelected)
        self.w.TComboboxError.current(0)

        # Initialize list of available optimization methods.
        self.w.TComboboxMethod['values'] = [m.name for m in self.methods]
        self.w.TComboboxMethod.bind('<<ComboboxSelected>>', self.ComboboxMethodSelected)
        self.w.TComboboxMethod.current(0)

        # Initialize list of available deformation quantities.
        self.w.TComboboxDeformation['values'] = [d.name for d in self.deformation_quantities]
        self.w.TComboboxDeformation.current(0)

        # Initialize list of available stress quantities.
        self.w.TComboboxStress['values'] = [s.name for s in self.stress_quantities]
        self.w.TComboboxStress.current(0)

        # Initialize functions and derivatives.
        self.ComboboxModelSelected(None)

        # Initialize error functions.
        self.ComboboxErrorSelected(None)

        # Initialize optimization method.
        self.ComboboxMethodSelected(None)

        # Initialize plotting utilities.
        self.titles = ['UT', 'ET', 'PS']
        self.data_colors = ['b', 'r', 'g']
        self.fit_colors = ['#8080ff', '#ff8080', '#80ff80']

        # Initialize widget references. 
        self.fit_checkbuttons = [self.w.CheckbuttonFitUT, self.w.CheckbuttonFitET, self.w.CheckbuttonFitPS]
        self.weight_entries = [self.w.EntryUT, self.w.EntryET, self.w.EntryPS]
        self.plot_checkbuttons = [self.w.CheckbuttonPlotUT, self.w.CheckbuttonPlotET, self.w.CheckbuttonPlotPS]

        # Initialize default texts.
        self.set_entry(self.w.EntryFilename, 'data/TRELOARUT.csv')
        self.set_entry(self.w.EntryDelimiter, ',')
        self.set_entry(self.w.EntryDeformation, '0')
        self.set_entry(self.w.EntryStress, '1')
        [self.set_entry(entry, '1') for entry in self.weight_entries]

    def SetTkVar(self):      
        """Sets the TkInter variables referenced by the widgets."""
        self.radiovar = tk.IntVar()
        self.isFit = [tk.BooleanVar() for i in range(3)]
        self.isPlot = [tk.BooleanVar() for i in range(3)]
        self.calcJac = tk.BooleanVar()
        self.calcHess = tk.BooleanVar()

    def ButtonProcess_Click(self):
        # Start a background thread to process file.
        thread_dataproc = threading.Thread(target=self.try_process_data)
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
        self.plot()

    def ComboboxErrorSelected(self, eventObj):
        current_error_function = self.w.TComboboxError.current()
        if self.error_function == self.error_functions[current_error_function]:
            return
        self.params = None
        self.error_function = self.error_functions[current_error_function]
        self.plot()

    def ComboboxMethodSelected(self, eventObj):
        current_method = self.w.TComboboxMethod.current()
        if self.method == self.methods[current_method]:
            return
        self.params = None
        self.method = self.methods[current_method]
        if all([xdata is None for xdata in self.xdatas]):
            return
        self.config_method_widgets(True)
        self.plot()

    def config_method_widgets(self, state):
        self.config_checkbutton(self.w.CheckbuttonJac, self.calcJac,
                                state and self.method.support_jac,
                                self.method.default_jac)
        self.config_checkbutton(self.w.CheckbuttonHess, self.calcHess,
                                state and self.method.support_hess,
                                self.method.default_hess)
        self.config_entry(self.w.EntryXtol,
                          state and self.method.support_xtol,
                          self.method.default_xtol)
        self.config_entry(self.w.EntryGtol,
                          state and self.method.support_gtol,
                          self.method.default_gtol)
        self.config_entry(self.w.EntryIterations,
                          state and self.method.support_iterations,
                          self.method.default_iterations)
        self.config_entry(self.w.EntryRounds,
                          state and self.method.support_rounds,
                          self.method.default_rounds)

    def config_checkbutton(self, checkbutton, boolvar, isEnabled, default):
        if isEnabled:
            state = tk.NORMAL
        else:
            state = tk.DISABLED
        checkbutton.config(state=state)
        boolvar.set(default)

    def config_entry(self, entry, isEnabled, default):
        if isEnabled:
            state = tk.NORMAL
        else:
            state = tk.DISABLED
        entry.config(state=state)
        self.set_entry(entry, default)

    def ButtonFitModel_Click(self):
        # Start a background thread to fit model parameters.
        thread_fitmodel = threading.Thread(target=self.try_fit_model)
        thread_fitmodel.setDaemon(True)
        thread_fitmodel.start()

    def set_state(self, isNormal):
        """Sets the state of all the input widgets normal
        or disabled/readonly according to the given parameter.
        ----------
        Keyword arguments:
        isNormal -- True, if widgets should be normal (active);
                    otherwise, false.
        """
        if isNormal:
            state = tk.NORMAL
            combobox_state = 'readonly'
        else:
            state = tk.DISABLED
            combobox_state = tk.DISABLED
            
        self.w.ButtonFitModel.config(state=state)
        self.w.ButtonProcess.config(state=state)
        self.w.EntryFilename.config(state=state)
        self.w.EntrySamples.config(state=state)
        self.w.EntryDelimiter.config(state=state)
        self.w.RadiobuttonUT.config(state=state)
        self.w.RadiobuttonET.config(state=state)
        self.w.RadiobuttonPS.config(state=state)
        self.w.TComboboxModel.config(state=combobox_state)
        self.w.TComboboxError.config(state=combobox_state)
        self.w.TComboboxMethod.config(state=combobox_state)
        [self.weight_entries[i].config(state=state) for i in range(3) if self.xdatas[i] is not None]
        [self.fit_checkbuttons[i].config(state=state) for i in range(3) if self.xdatas[i] is not None]
        [self.plot_checkbuttons[i].config(state=state) for i in range(3) if self.xdatas[i] is not None]
        self.config_method_widgets(isNormal)

    def try_process_data(self):
        self.set_state(False)
        self.process_data()
        self.set_state(True)

    def try_fit_model(self):
        self.set_state(False)
        self.fit_model()
        self.set_state(True)

    def process_data(self):
        # Read file parsing parameters from GUI.
        filename = self.w.EntryFilename.get()        
        self.print('Processing file "' + filename + '"...')
        samples_string = self.w.EntrySamples.get()
        if self.is_empty_or_whitespace(samples_string):
            samples = -1
        else:
            try:
                samples = int(samples_string)
            except ValueError:
                self.print('ERROR: Invalid number of samples.')
                return
        delimiter = self.w.EntryDelimiter.get().strip()
        if len(delimiter) != 1:
            self.print('ERROR: Delimiter must be a 1-character string.')
            return
        deformation_quantity = self.deformation_quantities[self.w.TComboboxDeformation.current()]
        stress_quantity = self.stress_quantities[self.w.TComboboxStress.current()]
        try:
            deformation_column = int(self.w.EntryDeformation.get())
        except ValueError:
            self.print('ERROR: Invalid deformation column index.')
            return
        if deformation_column < 0:
            self.print('ERROR: Deformation column index must be a nonnegative integer.')
        try:
            stress_column = int(self.w.EntryStress.get())
        except ValueError:
            self.print('ERROR: Invalid stress column index.')
            return
        if stress_column < 0:
            self.print('ERROR: Stress column index must be a nonnegative integer.')
        defmode = self.radiovar.get()

        # Execute file processing.
        try:
            self.xdatas[defmode], self.ydatas[defmode] = dataprocessor.process_file(filename,
                delimiter=delimiter, samples=samples,
                deformation_quantity=deformation_quantity, deformation_column=deformation_column,
                stress_quantity=stress_quantity, stress_column=stress_column)
        except FileNotFoundError:
            self.print('ERROR: File not found.')
            return
        except ValueError:
            self.print('ERROR: Invalid file format.')
            return
        
        # Add (0, 0) to datas if it does not exist.
        if (0 not in self.xdatas[defmode]):
            self.xdatas[defmode].append(0)
            self.ydatas[defmode].append(0)
            self.print('Origin has been added to measurement points.')

        # Enable this deformation mode in GUI.
        self.isFit[defmode].set(True)
        self.isPlot[defmode].set(True)
        self.plot()

        self.print('File has been processed.')

    def fit_model(self):
        self.print("""Fitting {0} model...
Error function: {1}
Optimization method: {2}""".format(self.model.name,
                                   self.error_function.name,
                                   self.method.name))
        start_time = process_time()
        try:
            weights = [float(self.weight_entries[i].get().strip()) \
                if self.xdatas[i] is not None and self.isFit[i].get() \
                else 0.0 for i in range(3)]
        except ValueError:
            self.print('ERROR: Weights must be real numbers.')
            return
        try:
            if self.method.support_xtol:
                xtol = float(self.w.EntryXtol.get().strip())
            else:
                xtol = None
        except ValueError:
            self.print('ERROR: Xtol must be a real number.')
            return
        try:
            if self.method.support_gtol:
                gtol = float(self.w.EntryGtol.get().strip())
            else:
                gtol = None
        except ValueError:
            self.print('ERROR: Gtol must be a real number.')
            return
        try:
            if self.method.support_iterations:
                iterations = int(self.w.EntryIterations.get().strip())
                if iterations < 1:
                    raise ValueError
            else:
                iterations = None
        except ValueError:
            self.print('ERROR: Number of iterations must be a positive integer.')
        try:
            if self.method.support_rounds:
                rounds = int(self.w.EntryRounds.get().strip())
                if rounds < 1:
                    raise ValueError
            else:
                rounds = None
        except ValueError:
            self.print('ERROR: Number of rounds must be a positive integer.')
        
        # Define (0, 0) point indices to remove origin when calculating errors.
        originIndices = [self.xdatas[i].index(0) if self.xdatas[i] is not None else None for i in range(3)]

        # Initialize error instances with origins removed.
        self.errors = [self.error_function(self.functions[i], self.jacobians[i], self.hessians[i],
                           self.xdatas[i][:originIndices[i]] + self.xdatas[i][originIndices[i] + 1:],
                           self.ydatas[i][:originIndices[i]] + self.ydatas[i][originIndices[i] + 1:]) \
            if self.xdatas[i] is not None else None for i in range(3)]
        self.weighted_error = WeightedError(self.errors, weights)

        # Define Jacobian and Hessian calculation method based on GUI input.
        if self.calcJac.get():
            jac = self.weighted_error.jac
        else:
            jac = '2-point'
        if self.calcHess.get():
            hess = self.weighted_error.hess
        else:
            hess = BFGS()
        self.error_values = []
        x0 = [1.0] * self.model.paramcount if self.params is None else self.params
        result = self.method.minimize(objfunc=self.weighted_error.objfunc,
                             x0=x0, constraint=self.model.constraint,
                             jac=jac, hess=hess,
                             callback=self.update_model,
                             xtol=xtol, gtol=gtol,
                             maxiter=iterations, rounds=rounds)
        self.update_model(result.x, result)
        end_time = process_time()
        elapsed_time = end_time - start_time
        self.print("""Model has been fitted:
Error: {0:.4f}
Elapsed time: {1:.4f} ms
Number of iterations: {2}
Number of function evaluations: {3}
{4}""".format(result.fun, elapsed_time, result.niter, result.nfev, result.message))

    def update_model(self, xk, state):
        self.params = xk
        self.error_values.append(state.fun)
        if state.niter % 100 == 0:
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
                label += ' - error={:.4f}'.format(self.errors[defmode].objfunc(self.params))
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
            self.set_text(self.w.TextState, '')
            pass
        else:
            # Write parameters and weighted error to Text widget.
            self.set_text(self.w.TextState, '\n'.join(['{} = {:.4f}' \
                .format(self.model.paramnames[i], self.params[i]) \
                for i in range(self.model.paramcount)]) + \
                '\n\nerror = {:.4f}'.format(self.weighted_error.objfunc(self.params)))
            pass

        if not is_empty:        
            self.plt.set_xlabel('engineering stretch')
            self.plt.set_ylabel('engineering stress')
            self.plt.legend()
            self.canvas.draw()
            self.toolbar.update()
           
        # Plot error function.
        self.pltErr.clear()
        if self.error_values is not None:
            self.pltErr.plot(range(len(self.error_values)), self.error_values, '-k')
            self.pltErr.set_xlabel('iterations')
            self.pltErr.set_ylabel('error function')
            self.canvasErr.draw()
            self.toolbarErr.update()

    def set_entry(self, entry, value):
        """Sets the content of the given Entry widget.
        ----------
        Keyword arguments:
        entry -- The Entry whose text should be set.
        value -- The new text string.
        """
        old_state = entry['state']
        entry.config(state=tk.NORMAL)
        entry.delete(0, tk.END)
        entry.insert(tk.END, value)
        entry.config(state=old_state)

    def set_text(self, text, value):
        """Sets the content of the given Text widget.
        ----------
        Keyword arguments:
        text -- The Text whose text should be set.
        value -- The new text string.
        """
        text.config(state=tk.NORMAL)
        text.delete(1.0, tk.END)
        text.insert(tk.END, value)
        text.config(state=tk.DISABLED)

    def print(self, string):
        """Prints the given text to the logger
        with a time stamp and a new line at end."""
        text = self.w.TextLog
        text.config(state=tk.NORMAL)
        time = datetime.now().strftime("[%H:%M:%S] ")
        line_sep = '\n' + ' ' * 18
        string = line_sep.join(string.split('\n'))
        text.insert(tk.END, time + string + '\n')
        text.see(tk.END)
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
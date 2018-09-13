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
from ogden1 import Ogden1
from ogden2 import Ogden2
from ogden3 import Ogden3
from errorfuncs import MSE, WeightedError

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

    # Initialize models.
    global models, model
    models = [Ogden1(), Ogden2(), Ogden3()]
    model = None
    
    # Initialize list of available models.
    w.TComboboxModel['values'] = [m.name for m in models]
    w.TComboboxModel.bind('<<ComboboxSelected>>', ComboboxSelected)
    w.TComboboxModel.current(0)

    # Initialize functions and derivatives.
    ComboboxSelected(None)

    # Initialize plotting utilities.
    global titles, data_markers, fit_markers
    titles = ['UT', 'ET', 'PS']
    data_markers = ['bo', 'ro', 'go']
    fit_markers = ['b-', 'r-', 'g-']

    # Initialize widget references.
    global fit_checkbuttons, weight_entries, plot_checkbuttons    
    fit_checkbuttons = [w.CheckbuttonFitUT, w.CheckbuttonFitET, w.CheckbuttonFitPS]
    weight_entries = [w.EntryUT, w.EntryET, w.EntryPS]
    plot_checkbuttons = [w.CheckbuttonPlotUT, w.CheckbuttonPlotET, w.CheckbuttonPlotPS]

    # Initialize default texts.
    set_entry(w.EntryFilename, 'data/TRELOARUT.csv')
    set_entry(w.EntryDelimiter, ',')

    # Initialize a Figure for plotting.
    fig = Figure(figsize=(5, 2), dpi=100)
    global plt
    plt = fig.add_subplot(111)
    fig.set_tight_layout(True)
    
    # Initialize a canvas to draw the Figure.
    global canvas
    canvas = FigureCanvasTkAgg(fig, w.FramePlot)
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Initialize navigation toolbar.
    global toolbar
    toolbar = NavigationToolbar2Tk(canvas, w.FrameNavigation)
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Add animation to plot interactively.
    global ani
    ani = animation.FuncAnimation(fig, plot, interval=500)

    # Initialize data variables.
    global xdatas, ydatas
    xdatas = [None] * 3
    ydatas = [None] * 3

    # Initialize error variables
    global errors, weighted_error
    errors = [None] * 3
    weighted_error = None

    # Initialize parameters variable
    global params
    params = None

def set_Tk_var():
    global radiovar
    radiovar = tk.IntVar()

    global isFit
    isFit = [tk.BooleanVar() for i in range(3)]

    global isPlot
    isPlot = [tk.BooleanVar() for i in range(3)]

def ButtonProcess_Click():
    # Read file parsing parameters from GUI.
    filename = w.EntryFilename.get()
    samplesString = w.EntrySamples.get()
    if is_empty_or_whitespace(samplesString):
        samples = -1
    else:
        try:
            samples = int(samplesString)
        except ValueError:
            messagebox.showerror('Error', 'Invalid number of samples.')
            return
    delimiter = w.EntryDelimiter.get().strip()
    if len(delimiter) != 1:
        messagebox.showerror('Error', 'Delimiter must be a 1-character string.')

    # Start a background thread to process file.
    thread_dataproc = threading.Thread(target=process_data, args=(filename, delimiter, samples))
    thread_dataproc.setDaemon(True)
    thread_dataproc.start()

def ComboboxSelected(eventObj):
    global model, functions, jacobians, hessians, params
    if model != models[w.TComboboxModel.current()]:
        params = None
        model = models[w.TComboboxModel.current()]
        functions = [model.ut, model.et, model.ps]
        jacobians = [model.ut_jac, model.et_jac, model.ps_jac]
        hessians = [model.ut_hess, model.et_hess, model.ps_hess]

def ButtonFitModel_Click():
    # Start a background thread to fit model parameters.
    thread_fitmodel = threading.Thread(target=fit_model)
    thread_fitmodel.setDaemon(True)
    thread_fitmodel.start()

def process_data(filename, delimiter, samples):
    defmode = radiovar.get()
    try:
        xdatas[defmode], ydatas[defmode] = dataprocessor.process_file(filename, delimiter=delimiter, samples=samples)
    except FileNotFoundError:
        messagebox.showerror('Error', 'File not found.')
        return
    except ValueError:
        messagebox.showerror('Error', 'Invalid file format.')
        return
    
    # Enable this deformation mode in GUI.
    fit_checkbuttons[defmode].config(state=tk.NORMAL)
    isFit[defmode].set(True)
    weight_entries[defmode].config(state=tk.NORMAL)
    set_entry(weight_entries[defmode], '1')
    w.ButtonFitModel.config(state=tk.NORMAL)
    plot_checkbuttons[defmode].config(state=tk.NORMAL)
    isPlot[defmode].set(True)

def fit_model():
    try:
        weights = [float(weight_entries[i].get().strip()) if xdatas[i] is not None and isFit[i].get() else 0.0 for i in range(3)]
    except ValueError:
        messagebox.showerror('Error', 'Weights must be real numbers.')
        return
    global errors, weighted_error, functions, jacobians, hessians
    errors = [MSE(functions[i], jacobians[i], hessians[i], xdatas[i], ydatas[i]) \
        if xdatas[i] is not None else None for i in range(3)]
    weighted_error = WeightedError(errors, weights)
    minimize(weighted_error.objfunc, [1.0] * model.paramcount, method='trust-constr', \
                   constraints=model.constraint(), \
                   jac="2-point", hess=BFGS(),
                   callback=update_model,
                   options={'maxiter':10000, 'disp': True})

def update_model(xk, state):
    global params
    params = xk

def plot(i):
    plt.clear()
    is_empty = True

    for defmode in range(3):
        # Plot read data as points.
        if xdatas[defmode] is None or not isPlot[defmode].get():
            continue
        xdata = xdatas[defmode]
        ydata = ydatas[defmode]
        label = titles[defmode]
        if errors[defmode] is not None and params is not None:
            label += ' - error={:.4f}'.format(errors[defmode].objfunc(params))
        plt.plot(xdata, ydata, data_markers[defmode], label=label)
        is_empty = False
        
        # Plot fitted models as curves.
        if params is None:
            continue
        xlin = np.linspace(min(xdata), max(xdata))
        plt.plot(xlin, functions[defmode](xlin, *params), fit_markers[defmode])    

    if params is None:
        # Clear Text widget.
        set_text(w.TextParameters, '')
    else:
        # Write parameters and weighted error to Text widget.
        set_text(w.TextParameters, '\n'.join(['{} = {}'.format(model.paramnames[i], params[i]) \
           for i in range(model.paramcount)]) + '\n\nerror = {}'.format(weighted_error.objfunc(params)))

    if not is_empty:        
        plt.set_xlabel('stretch')
        plt.set_ylabel('pressure')
        plt.legend()
        canvas.draw()
        toolbar.update()

def set_entry(entry, value):
    entry.delete(0, tk.END)
    entry.insert(tk.END, value)

def set_text(text, value):
    text.config(state=tk.NORMAL)
    text.delete(1.0, tk.END)
    text.insert(tk.END, value)
    text.config(state=tk.DISABLED)

def is_empty_or_whitespace(string):
    if not string:
        return True
    
    return string.isspace()

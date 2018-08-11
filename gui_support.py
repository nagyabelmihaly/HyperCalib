import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib import style
style.use('ggplot')
import tkinter as tk
import threading

import dataprocessor
import funcs
import minimizer

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

    global functions, titles
    functions = [funcs.ogden_ut, funcs.ogden_et, funcs.ogden_ps]
    titles = ['Uniaxial tension', 'Equibiaxial tension', 'Pure shear']

    # Initialize text entry default texts.
    w.EntryFilename.insert(0, 'data/TRELOARUT.csv')
    w.EntryDelimiter.insert(0, ',')

    # Initialize a Figure for plotting.
    fig = Figure(figsize=(5, 5), dpi=100)
    global plt
    plt = fig.add_subplot(111)
    
    # Initialize a canvas to draw the Figure.
    global canvas
    canvas = FigureCanvasTkAgg(fig, w.FramePlot)
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)    

    # Initialize navigation toolbar.
    global toolbar
    toolbar = NavigationToolbar2TkAgg(canvas, w.FrameNavigation)
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Add animation to plot interactively.
    global ani
    ani = animation.FuncAnimation(fig, plot, interval=500)

def set_Tk_var():
    global radiovar
    radiovar = tk.IntVar()

def ButtonProcess_Click():
    # Read file parsing parameters from GUI.
    filename = w.EntryFilename.get()
    try:
        samples = int(w.EntrySamples.get())
    except ValueError:
        samples = -1
    delimiter = w.EntryDelimiter.get()

    # Start a background thread to process file.
    thread_dataproc = threading.Thread(target=process_data, args=(filename, delimiter, samples))
    thread_dataproc.start()

def ButtonFitModel_Click():
    # Read model fitting parameters from GUI.
    global func
    func = functions[radiovar.get()]

    # Start a background thread to fit model parameters.
    thread_fitmodel = threading.Thread(target=fit_model)
    thread_fitmodel.start()

def process_data(filename, delimiter, samples):
    global xdata, ydata
    xdata, ydata = dataprocessor.process_file(filename, delimiter=delimiter, samples=samples)

def fit_model():
    minimizer.fit_model(func, xdata, ydata, \
                        constraints=funcs.ogden_constraints(), \
                        callback=update_model)

def update_model(xk, state):
    global params
    params = xk

def plot(i):
    plt.clear()
    is_empty = True

    # Plot read data as points.
    try:
        plt.plot(xdata, ydata, 'bo', label='data')
        is_empty = False
    except NameError:
        pass

    # Plot fitted model as a curve.
    try:
        xlin = np.linspace(min(xdata), max(xdata))
        plt.plot(xlin, func(xlin, *params), 'r-', label='fit')
        is_empty = False
    except NameError:
        pass

    if not is_empty:        
        plt.set_xlabel('stretch')
        plt.set_ylabel('pressure')
        plt.set_title(titles[radiovar.get()])
        plt.legend()
        canvas.draw()
        toolbar.update()

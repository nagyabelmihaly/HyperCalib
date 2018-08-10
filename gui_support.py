import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk

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

    w.EntryFilename.insert(0, 'data/TRELOARUT.csv')
    w.EntryDelimiter.insert(0, ',')

def set_Tk_var():
    global radiovar
    radiovar = tk.IntVar()

def ButtonProcess_Click():
    filename = w.EntryFilename.get()
    try:
        samples = int(w.EntrySamples.get())
    except ValueError:
        samples = -1
    delimiter = w.EntryDelimiter.get()
    xdata, ydata = dataprocessor.process_file(filename, delimiter=delimiter, samples=samples)
    func = functions[radiovar.get()]
    params = minimizer.fit_model(func, xdata, ydata, constraints=funcs.ogden_constraints())

    fig = Figure(figsize=(5, 5), dpi=100)
    plt = fig.add_subplot(111)
    plt.plot(xdata, ydata, 'bo', label='data')
    xlin = np.linspace(min(xdata), max(xdata))
    plt.plot(xlin, func(xlin, *params), 'r-', label='fit')
    #plt.xlabel('stretch')
    #plt.ylabel('pressure')
    #plt.title(titles[0])
    plt.legend()
    
    for child in w.FramePlot.winfo_children():
        child.destroy()
    canvas = FigureCanvasTkAgg(fig, w.FramePlot)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    for child in w.FrameNavigation.winfo_children():
            child.destroy()
    toolbar = NavigationToolbar2TkAgg(canvas, w.FrameNavigation)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from utilities import *
from dataprocessor import DataProcessor
from deformation import EngineeringStrain, Stretch, TrueStrain
from stress import EngineeringStress, TrueStress

class FileDialog(tk.Toplevel):
    def __init__(self, callback):
        tk.Toplevel.__init__(self)
        self.grab_set()
        self.title('Process file')
        self.geometry("%dx%d%+d%+d" % (400, 300, 300, 200))
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.callback = callback

        self.processor = DataProcessor()

        self.defaultvar = 'none'
        self.radiovar = tk.StringVar(None, self.defaultvar)

        self.frames = {}
        for F in (PageTwo, PageThree, PageFour, PageFive):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("PageTwo")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

    def close(self):
        defmode = self.radiovar.get()
        stretch = self.processor.stretch_limited
        true_stress = self.processor.true_stress_limited
        self.callback(defmode, stretch, true_stress, self.filename)
        self.destroy()

class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Initialize widgets.
        label = tk.Label(self, text="Filename:")
        self.entryFilename = tk.Entry(self)
        buttonBrowse = tk.Button(self, text="Browse", command=self.browse)
        labelDefmode = tk.Label(self, text='Deformation mode:')
        rbUT = tk.Radiobutton(self, text='Uniaxial tension', value='UT', variable=controller.radiovar)
        rbET = tk.Radiobutton(self, text='Equibiaxial tension', value='ET', variable=controller.radiovar)
        rbPS = tk.Radiobutton(self, text='Pure shear', value='PS', variable=controller.radiovar)
        buttonPrev = tk.Button(self, text="<<", state=tk.DISABLED)
        buttonNext = tk.Button(self, text=">>", command=self.next)
        
        # Arrange widgets in a grid.
        label.grid(row=0, column=0, sticky=tk.W, padx=5)
        self.entryFilename.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5)
        buttonBrowse.grid(row=0, column=2, sticky=tk.E, padx=5, pady=5)
        labelDefmode.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        rbUT.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=3)
        rbET.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=3)
        rbPS.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=3)
        tk.Frame(self).grid(row=5, column=0, columnspan=2)
        buttonPrev.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        buttonNext.grid(row=6, column=2, sticky=tk.E, padx=5, pady=5)

        self.grid_rowconfigure(5, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def browse(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd()+'/data',
                                              title='Open')
        set_entry(self.entryFilename, filename)

    def next(self):
        self.filename = self.entryFilename.get()
        self.controller.filename = self.filename
        try:
            self.controller.processor.load_file(self.filename)
        except FileNotFoundError:
            messagebox.showerror('Error', 'File ' + self.filename + ' was not found.')
            return
        if self.controller.radiovar.get() == self.controller.defaultvar:
            messagebox.showerror('Error', 'No deformation mode is selected.')
            return
        set_text(self.controller.frames['PageThree'].text, '\n'.join(self.controller.processor.lines))
        self.controller.show_frame("PageThree")

class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Initialize widgets.
        label1 = tk.Label(self, text='Delimiter:')
        label2 = tk.Label(self, text='Decimal separator:')
        self.entryDelimiter = tk.Entry(self)
        self.entryDecimalsep = tk.Entry(self)
        self.text = tk.Text(self, state=tk.DISABLED)
        buttonPrev = tk.Button(self, text="<<", command=lambda: controller.show_frame("PageTwo"))
        buttonNext = tk.Button(self, text=">>", command=self.next)
        
        # Arrange widgets in a grid.
        label1.grid(row=0, column=0, sticky=tk.W, padx=5)
        label2.grid(row=1, column=0, sticky=tk.W, padx=5)
        self.entryDelimiter.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        self.entryDecimalsep.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        self.text.grid(row=2, column=0, columnspan=2, padx=5, pady=3)
        buttonPrev.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        buttonNext.grid(row=3, column=1, sticky=tk.E, padx=5, pady=5)

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(1, weight=1)
    
    def next(self):
        delimiter = self.entryDelimiter.get().strip()
        if len(delimiter) != 1:
            messagebox.showerror('ERROR', 'Delimiter must be a 1-character string.')
            return
        decimalsep = self.entryDecimalsep.get().strip()
        if len(decimalsep) != 1:
            messagebox.showerror('ERROR', 'Decimal separator must be a 1-character string.')
            return
        processor = self.controller.processor
        processor.parse_csv(delimiter, decimalsep)
        columns = processor.max_column_count
        table = self.controller.frames['PageFour'].table
        table['columns'] = [str(c) for c in range(1, columns + 1)]
        table.heading('#0', text='No.')
        table.column('#0', anchor='center', width=50, stretch=False)
        for c in range(1, columns + 1):
            table.heading(str(c), text='Column ' + str(c))
            table.column(str(c), anchor='center', stretch=True)
        table.delete(*table.get_children())
        i = 0
        for row in processor.raw:
            i += 1
            table.insert('', 'end', text='#' + str(i), values=row)
        self.controller.show_frame("PageFour")

class PageFour(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Initialize widgets.
        self.comboboxDeformation = ttk.Combobox(self, state='readonly')
        self.comboboxStress = ttk.Combobox(self, state='readonly')
        label3 = tk.Label(self, text='Column:')
        label4 = tk.Label(self, text='Column:')
        self.entryDeformation = tk.Entry(self)
        self.entryStress = tk.Entry(self)
        self.table = ttk.Treeview(self)
        buttonPrev = tk.Button(self, text="<<", command=lambda: controller.show_frame("PageThree"))
        buttonNext = tk.Button(self, text=">>", command=self.next)
        
        # Arrange widgets in a grid.
        self.comboboxDeformation.grid(row=0, column=0, sticky=tk.W+tk.E, padx=5, pady=3)
        self.comboboxStress.grid(row=1, column=0, sticky=tk.W+tk.E, padx=5, pady=3)
        label3.grid(row=0, column=1, sticky=tk.W, padx=5)
        label4.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.entryDeformation.grid(row=0, column=2, sticky=tk.W+tk.E, padx=5)
        self.entryStress.grid(row=1, column=2, sticky=tk.W+tk.E, padx=5)
        self.table.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)
        buttonPrev.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        buttonNext.grid(row=3, column=2, sticky=tk.E, padx=5, pady=5)

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Initialize deformation and stress quantities.
        self.deformation_quantities = [Stretch, EngineeringStrain, TrueStrain]
        self.stress_quantities = [TrueStress, EngineeringStress]

        # Initialize list of available deformation quantities.
        self.comboboxDeformation['values'] = [d.name for d in self.deformation_quantities]
        self.comboboxDeformation.current(0)

        # Initialize list of available stress quantities.
        self.comboboxStress['values'] = [s.name for s in self.stress_quantities]
        self.comboboxStress.current(0)

        # Set default columns.
        set_entry(self.entryDeformation, '1')
        set_entry(self.entryStress, '2')

    def next(self):
        deformation_quantity = self.deformation_quantities[self.comboboxDeformation.current()]
        stress_quantity = self.stress_quantities[self.comboboxStress.current()]
        try:
            deformation_column = int(self.entryDeformation.get())
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid deformation column index.')
            return
        try:
            stress_column = int(self.entryStress.get())
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid stress column index.')
            return
        try:
            self.controller.processor.define_data(deformation_quantity, deformation_column,
                                              stress_quantity, stress_column)
        except ValueError as err:
            messagebox.showerror('ERROR', str(err))
            return
        xdata = self.controller.processor.stretch
        ydata = self.controller.processor.true_stress
        plt = self.controller.frames['PageFive'].plt
        plt.clear()
        plt.plot(xdata, ydata, 'k.')
        plt.set_xlabel('stretch')
        plt.set_ylabel('true stress')
        self.controller.frames['PageFive'].canvas.draw()
        self.controller.show_frame("PageFive")

class PageFive(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Initialize widgets.
        labelSamples = tk.Label(self, text='Samples:')
        self.entrySamples = tk.Entry(self)
        self.framePlot = tk.Frame(self)
        buttonPrev = tk.Button(self, text="<<", command=lambda: controller.show_frame("PageFour"))
        buttonNext = tk.Button(self, text="FINISH", command=self.next)
        
        # Arrange widgets in a grid.
        labelSamples.grid(row=0, column=0, sticky=tk.W, padx=5)
        self.entrySamples.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.framePlot.grid(row=1, column=0, columnspan=2, sticky=tk.N+tk.S+tk.W+tk.E, padx=5)
        buttonPrev.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        buttonNext.grid(row=2, column=1, sticky=tk.E, padx=5, pady=5)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Initialize a Figure for plotting the model.
        fig = Figure(figsize=(5, 2), dpi=100)
        self.plt = fig.add_subplot(111)
        fig.set_tight_layout(True)
    
        # Initialize a canvas to draw the Figure.
        self.canvas = FigureCanvasTkAgg(fig, self.framePlot)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def next(self):
        samples_string = self.entrySamples.get()
        if is_empty_or_whitespace(samples_string):
            samples = -1
        else:
            try:
                samples = int(samples_string)
            except ValueError:
                messagebox.showerror('ERROR', 'Invalid number of samples.')
                return
        self.controller.processor.limit_data(samples)
        xdatas = self.controller.processor.stretch_limited
        ydatas = self.controller.processor.true_stress_limited

        # Add (1, 0) to datas if it does not exist.
        if (1.0 not in xdatas):
            xdatas.append(1.0)
            ydatas.append(0.0)

        self.controller.close()
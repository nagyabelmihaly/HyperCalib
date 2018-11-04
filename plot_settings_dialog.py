import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

from utilities import *

from rmsae import RMSAE
from rmsre import RMSRE
from cod import COD

from deformation import EngineeringStrain, TrueStrain, Stretch
from stress import EngineeringStress, TrueStress

class PlotSettingsDialog(tk.Toplevel):
    def __init__(self, callback, loaded_defmode, plot_defmode,
                 current_error, current_deformation, current_stress):
        tk.Toplevel.__init__(self)
        self.grab_set()
        self.title('Plot settings')
        self.geometry("%dx%d%+d%+d" % (400, 300, 300, 200))        

        self.callback = callback
        
        # Initialize tk bound variables.
        self.plotUT = tk.BooleanVar()
        self.plotET = tk.BooleanVar()
        self.plotPS = tk.BooleanVar()
        self.plotUT.set(plot_defmode[0])
        self.plotET.set(plot_defmode[1])
        self.plotPS.set(plot_defmode[2])

        # Initialize widgets.
        labelPlot = tk.Label(self, text='Plot?')
        labelError = tk.Label(self, text='Fit measure:')
        labelDeformation = tk.Label(self, text='Deformation quantity:')
        labelStress = tk.Label(self, text='Stress quantity:')
        self.checkbuttonUT = tk.Checkbutton(self, variable=self.plotUT,
            state=tk.NORMAL if loaded_defmode[0] else tk.DISABLED,
            text='Uniaxial tension')
        self.checkbuttonET = tk.Checkbutton(self, variable=self.plotET,
            state=tk.NORMAL if loaded_defmode[1] else tk.DISABLED,
            text='Equibiaxial tension')
        self.checkbuttonPS = tk.Checkbutton(self, variable=self.plotPS,
            state=tk.NORMAL if loaded_defmode[2] else tk.DISABLED,
            text='Pure shear')
        self.comboboxError = ttk.Combobox(self, state='readonly')
        self.comboboxDeformation = ttk.Combobox(self, state='readonly')
        self.comboboxStress = ttk.Combobox(self, state='readonly')
        buttonOK = tk.Button(self, text="OK", command=self.ok)
        buttonCancel = tk.Button(self, text="Cancel", command=self.cancel)

        # Arrange widgets in grid.
        labelPlot.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5)
        labelError.grid(row=4, column=0, sticky=tk.W, padx=5)
        labelDeformation.grid(row=5, column=0, sticky=tk.W, padx=5)
        labelStress.grid(row=6, column=0, sticky=tk.W, padx=5)
        self.checkbuttonUT.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5)
        self.checkbuttonET.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
        self.checkbuttonPS.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5)
        self.comboboxError.grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        self.comboboxDeformation.grid(row=5, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        self.comboboxStress.grid(row=6, column=1, sticky=tk.W+tk.E, padx=5, pady=3)
        tk.Frame(self).grid(row=7, column=0, columnspan=2, padx=5)
        buttonOK.grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        buttonCancel.grid(row=8, column=1, sticky=tk.E, padx=5, pady=5)

        self.grid_rowconfigure(7, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Initialize errors, deformation and stress quantities.
        self.errors = [RMSAE, RMSRE, COD]
        self.deformation_quantities = [Stretch, EngineeringStrain, TrueStrain]
        self.stress_quantities = [TrueStress, EngineeringStress]

        # Initialize list of available fit measures.
        self.comboboxError['values'] = [e.name for e in self.errors]
        self.comboboxError.current(self.errors.index(current_error))

        # Initialize list of available fit measures.
        self.comboboxDeformation['values'] = [d.name for d in self.deformation_quantities]
        self.comboboxDeformation.current(self.deformation_quantities.index(current_deformation))

        # Initialize list of available fit measures.
        self.comboboxStress['values'] = [s.name for s in self.stress_quantities]
        self.comboboxStress.current(self.stress_quantities.index(current_stress))

    def ok(self):
        plot_defmode = [self.plotUT.get(), self.plotET.get(), self.plotPS.get()]
        error = self.errors[self.comboboxError.current()]
        deformation_quantity = self.deformation_quantities[self.comboboxDeformation.current()]
        stress_quantity = self.stress_quantities[self.comboboxStress.current()]

        self.destroy()
        self.callback(plot_defmode, error, deformation_quantity, stress_quantity)

    def cancel(self):
        self.destroy()
from subprocess import CalledProcessError

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

from utilities import *

from rmsae import RMSAE
from rmsre import RMSRE
from cod import COD

from deformation import EngineeringStrain, TrueStrain, Stretch
from stress import EngineeringStress, TrueStress

from pdf_generator import PdfGenerator

class CreateReportDialog(tk.Toplevel):
    def __init__(self, xdatas, ydatas, model, params, plot_defmode,
                 current_error, current_deformation, current_stress,
                 filenames, weights, method, fit_error):
        tk.Toplevel.__init__(self)
        self.grab_set()
        self.title('Create report')
        self.geometry("%dx%d%+d%+d" % (600, 400, 300, 200))

        self.xdatas = xdatas
        self.ydatas = ydatas
        self.model = model
        self.params = params
        loaded_defmode = [xdatas[i] is not None for i in range(3)]
        self.filenames = filenames
        self.weights = weights
        self.method = method
        self.fit_error = fit_error
        
        # Initialize tk bound variables.
        self.plotUT = tk.BooleanVar()
        self.plotET = tk.BooleanVar()
        self.plotPS = tk.BooleanVar()
        self.plotUT.set(plot_defmode[0])
        self.plotET.set(plot_defmode[1])
        self.plotPS.set(plot_defmode[2])

        # Initialize widgets.
        labelDefmodes = tk.Label(self, text='Deformation modes:')
        labelError = tk.Label(self, text='Error:')
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
        buttonCreate = tk.Button(self, text="Create", command=self.create)
        buttonCancel = tk.Button(self, text="Cancel", command=self.cancel)

        # Arrange widgets in grid.
        labelDefmodes.grid(row=0, column=0, columnspan=1)
        labelError.grid(row=4, column=0)
        labelDeformation.grid(row=5, column=0)
        labelStress.grid(row=6, column=0)
        self.checkbuttonUT.grid(row=1, column=0, columnspan=1)
        self.checkbuttonET.grid(row=2, column=0, columnspan=1)
        self.checkbuttonPS.grid(row=3, column=0, columnspan=1)
        self.comboboxError.grid(row=4, column=1)
        self.comboboxDeformation.grid(row=5, column=1)
        self.comboboxStress.grid(row=6, column=1)
        buttonCreate.grid(row=7, column=0)
        buttonCancel.grid(row=7, column=1)

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

    def create(self):
        plot_defmode = [self.plotUT.get(), self.plotET.get(), self.plotPS.get()]
        error = self.errors[self.comboboxError.current()]
        deformation_quantity = self.deformation_quantities[self.comboboxDeformation.current()]
        stress_quantity = self.stress_quantities[self.comboboxStress.current()]

        pdfgen = PdfGenerator()
        pdfgen.xdatas = self.xdatas
        pdfgen.ydatas = self.ydatas
        pdfgen.model = self.model
        pdfgen.params = self.params
        pdfgen.plot_defmode = plot_defmode
        pdfgen.plot_error = error
        pdfgen.plot_deformation = deformation_quantity
        pdfgen.plot_stress = stress_quantity
        pdfgen.filenames = self.filenames
        pdfgen.weights = self.weights
        pdfgen.method = self.method
        pdfgen.fit_error = self.fit_error

        try:
            pdfgen.generate()
        except CalledProcessError:
            messagebox.showerror('Error', 'Cannot open file to write.')
            return

        self.destroy()

    def cancel(self):
        self.destroy()

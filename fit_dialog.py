import numpy as np
from scipy.optimize import BFGS

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

from utilities import *

from ogden import Ogden
from neo_hooke import NeoHooke
from mooney_rivlin import MooneyRivlin
from yeoh import Yeoh
from arruda_boyce import ArrudaBoyce

from mse import MSE
from msre import MSRE
from weighted_error import WeightedError

from trust_constr import TrustConstr
from cobyla import Cobyla
from slsqp import Slsqp

class FitDialog(tk.Toplevel):
    def __init__(self, xdatas, ydatas, callback):
        tk.Toplevel.__init__(self)
        self.grab_set()
        self.title('Fit model')
        self.geometry("%dx%d%+d%+d" % (600, 400, 300, 200))
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.xdatas = xdatas
        self.ydatas = ydatas
        self.callback = callback

        self.frames = {}
        for F in (PageOne, PageTwo, PageTrustConstr, PageCobyla, PageSlsqp):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("PageOne")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

    def close(self):
        self.destroy()
        self.callback(self.model, self.error_function,
                      self.errors, self.weights,
                      self.weighted_error, self.method)

class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Initialize tk bound variables.
        self.fitUT = tk.BooleanVar()
        self.fitET = tk.BooleanVar()
        self.fitPS = tk.BooleanVar()
        self.fitUT.set(False if self.controller.xdatas[0] is None else True)
        self.fitET.set(False if self.controller.xdatas[1] is None else True)
        self.fitPS.set(False if self.controller.xdatas[2] is None else True)

        # Initialize widgets.
        labelWeight = tk.Label(self, text='Weights used in optimization:')
        self.checkbuttonUT = tk.Checkbutton(self, variable=self.fitUT,
            state=tk.DISABLED if self.controller.xdatas[0] is None else tk.NORMAL,
            command=self.update_state)
        self.checkbuttonET = tk.Checkbutton(self, variable=self.fitET,
            state=tk.DISABLED if self.controller.xdatas[1] is None else tk.NORMAL,
            command=self.update_state)
        self.checkbuttonPS = tk.Checkbutton(self, variable=self.fitPS,
            state=tk.DISABLED if self.controller.xdatas[2] is None else tk.NORMAL,
            command=self.update_state)
        labelUT = tk.Label(self, text='Uniaxial tension')
        labelET = tk.Label(self, text='Equibiaxial tension')
        labelPS = tk.Label(self, text='Pure shear')
        self.entryUT = tk.Entry(self)
        self.entryET = tk.Entry(self)
        self.entryPS = tk.Entry(self)
        buttonPrev = tk.Button(self, text="<<", state=tk.DISABLED)
        buttonNext = tk.Button(self, text=">>", command=self.next)

        self.update_state()

        # Arrange widgets in grid.
        labelWeight.grid(row=0, column=0, columnspan=3)
        self.checkbuttonUT.grid(row=1, column=0)
        labelUT.grid(row=1, column=1)
        self.entryUT.grid(row=1, column=2)
        self.checkbuttonET.grid(row=2, column=0)
        labelET.grid(row=2, column=1)
        self.entryET.grid(row=2, column=2)
        self.checkbuttonPS.grid(row=3, column=0)
        labelPS.grid(row=3, column=1)
        self.entryPS.grid(row=3, column=2)
        buttonPrev.grid(row=4, column=0)
        buttonNext.grid(row=4, column=2)

        # Initialize default entry values.
        set_entry(self.entryUT, '' if self.controller.xdatas[0] is None else '1')
        set_entry(self.entryET, '1' if self.controller.xdatas[1] is None else '1')
        set_entry(self.entryPS, '1' if self.controller.xdatas[2] is None else '1')

    def update_state(self):
        self.entryUT['state'] = tk.NORMAL if self.fitUT.get() else tk.DISABLED
        self.entryET['state'] = tk.NORMAL if self.fitET.get() else tk.DISABLED
        self.entryPS['state'] = tk.NORMAL if self.fitPS.get() else tk.DISABLED

    def next(self):
        isFit = False
        if self.fitUT.get():
            isFit = True
            s = self.entryUT.get()
            try:
                weight = float(s)
            except ValueError:
                messagebox.showerror('Error', 'Invalid uniaxial tension weight "' + s + '".')
                return
            if weight <= 0.0:
                messagebox.showerror('Error', 'Uniaxial tension weight should be positive.')
                return
            self.controller.weightUT = weight
        else:
            self.controller.weightUT = 0
        if self.fitET.get():
            isFit = True
            s = self.entryET.get()
            try:
                weight = float(s)
            except ValueError:
                messagebox.showerror('Error', 'Invalid equibiaxial tension weight "' + s + '".')
                return
            if weight <= 0.0:
                messagebox.showerror('Error', 'Equibiaxial tension weight should be positive.')
                return
            self.controller.weightET = weight
        else:
            self.controller.weightET = 0
        if self.fitPS.get():
            isFit = True
            s = self.entryPS.get()
            try:
                weight = float(s)
            except ValueError:
                messagebox.showerror('Error', 'Invalid pure shear weight "' + s + '".')
                return
            if weight <= 0.0:
                messagebox.showerror('Error', 'Pure shear weight should be positive.')
                return
            self.controller.weightPS = weight
        else:
            self.controller.weightPS = 0
        if not isFit:
            messagebox.showerror('Error', 'At least one deformation mode has to be fitted.')
            return
        self.controller.show_frame("PageTwo")

class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Initialize widgets.
        labelModel = tk.Label(self, text='Hyperelastic model:')
        labelError = tk.Label(self, text='Error function:')
        labelMethod = tk.Label(self, text='Optimization method:')
        self.comboboxModel = ttk.Combobox(self, state='readonly')
        self.comboboxError = ttk.Combobox(self, state='readonly')
        self.comboboxMethod = ttk.Combobox(self, state='readonly')
        buttonPrev = tk.Button(self, text="<<",
                               command=lambda: controller.show_frame("PageOne"))
        buttonNext = tk.Button(self, text=">>",
                               command=self.next)
        
        # Arrange widgets in grid.
        labelModel.grid(row=0, column=0)
        labelError.grid(row=1, column=0)
        labelMethod.grid(row=2, column=0)
        self.comboboxModel.grid(row=0, column=1)
        self.comboboxError.grid(row=1, column=1)
        self.comboboxMethod.grid(row=2, column=1)
        buttonPrev.grid(row=3, column=0)
        buttonNext.grid(row=3, column=1)

        # Initialize models, errors and methods.
        self.models = [Ogden(1), Ogden(2), Ogden(3), NeoHooke(), MooneyRivlin(), 
                       Yeoh(), ArrudaBoyce()]
        self.error_functions = [MSE, MSRE]
        self.methods = [TrustConstr(), Cobyla(), Slsqp()]

        # Initialize list of available models.
        self.comboboxModel['values'] = [m.name for m in self.models]
        self.comboboxModel.current(0)

        # Initialize list of available error functions.
        self.comboboxError['values'] = [e.name for e in self.error_functions]
        self.comboboxError.current(0)

        # Initialize list of available optimization methods.
        self.comboboxMethod['values'] = [m.name for m in self.methods]
        self.comboboxMethod.current(0)

    def next(self):
        model = self.models[self.comboboxModel.current()]
        error_function = self.error_functions[self.comboboxError.current()]
        method = self.methods[self.comboboxMethod.current()]
        self.controller.model = model
        self.controller.error_function = error_function
        self.controller.method = method
        
        # Define errors.
        if self.controller.xdatas[0] is None:
            errorUT = None
        else:
            points = zip(self.controller.xdatas[0], self.controller.ydatas[0])
            points_without_origin = [point for point in points if point[0] != 1.0]
            xdata = [point[0] for point in points_without_origin]
            ydata = [point[1] for point in points_without_origin]
            errorUT = error_function(model.ut, model.ut_jac, model.ut_hess, xdata, ydata)

        if self.controller.xdatas[1] is None:
            errorET = None
        else:
            points = zip(self.controller.xdatas[1], self.controller.ydatas[1])
            points_without_origin = [point for point in points if point[0] != 1.0]
            xdata = [point[0] for point in points_without_origin]
            ydata = [point[1] for point in points_without_origin]
            errorET = error_function(model.et, model.et_jac, model.et_hess, xdata, ydata)

        if self.controller.xdatas[2] is None:
            errorPS = None
        else:
            points = zip(self.controller.xdatas[2], self.controller.ydatas[2])
            points_without_origin = [point for point in points if point[0] != 1.0]
            xdata = [point[0] for point in points_without_origin]
            ydata = [point[1] for point in points_without_origin]
            errorPS = error_function(model.ps, model.ps_jac, model.ps_hess, xdata, ydata)

        errors = [errorUT, errorET, errorPS]
        weights = [self.controller.weightUT,
                   self.controller.weightET,
                   self.controller.weightPS]

        weighted_error = WeightedError(errors, weights)

        self.controller.errors = errors
        self.controller.weights = weights
        self.controller.weighted_error = weighted_error

        # Set method parameters.
        method.objfunc = weighted_error.objfunc
        method.x0 = [1.0] * model.paramcount
        method.constraint = model.constraint

        # Navigate to page appropriate to optimization method.
        if isinstance(method, TrustConstr):
            page_name = "PageTrustConstr"
        if isinstance(method, Cobyla):
            page_name = "PageCobyla"
        if isinstance(method, Slsqp):
            page_name = "PageSlsqp"
        self.controller.frames[page_name].update()
        self.controller.show_frame(page_name)

class PageTrustConstr(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Initialize tk bound variables.
        self.calcJac = tk.BooleanVar()
        self.calcHess = tk.BooleanVar()
        self.calcJac.set(True)
        self.calcHess.set(False)

        # Initialize widgets.
        labelJac = tk.Label(self, text='Calculate Jacobian:')
        labelHess = tk.Label(self, text='Calculate Hessian:')
        labelXtol = tk.Label(self, text='Xtol:')
        labelGtol = tk.Label(self, text='Gtol:')
        labelBtol = tk.Label(self, text='Barrier tolerance:')
        labelMaxiter = tk.Label(self, text='Maximum iterations:')
        labelInitConstrPen = tk.Label(self, text='Initial constrains penalty:')
        labelInitTrustRad = tk.Label(self, text='Initial trust radius:')
        labelInitBarrPar = tk.Label(self, text='Initial barrier parameter:')
        labelInitBarrTol = tk.Label(self, text='Initial barrier tolerance:')
        self.checkbuttonJac = tk.Checkbutton(self, variable=self.calcJac)
        self.checkbuttonHess = tk.Checkbutton(self, variable=self.calcHess)
        self.entryXtol = tk.Entry(self)
        self.entryGtol = tk.Entry(self)
        self.entryBtol = tk.Entry(self)
        self.entryMaxiter = tk.Entry(self)
        self.entryInitConstrPen = tk.Entry(self)
        self.entryInitTrustRad = tk.Entry(self)
        self.entryInitBarrPar = tk.Entry(self)
        self.entryInitBarrTol = tk.Entry(self)
        buttonPrev = tk.Button(self, text="<<",
                               command=lambda: controller.show_frame("PageTwo"))
        buttonNext = tk.Button(self, text="FIT", command=self.next)
        
        # Arrange widgets in grid.
        labelJac.grid(row=0, column=0)
        labelHess.grid(row=1, column=0)
        labelXtol.grid(row=2, column=0)
        labelGtol.grid(row=3, column=0)
        labelBtol.grid(row=4, column=0)
        labelMaxiter.grid(row=5, column=0)
        labelInitConstrPen.grid(row=6, column=0)
        labelInitTrustRad.grid(row=7, column=0)
        labelInitBarrPar.grid(row=8, column=0)
        labelInitBarrTol.grid(row=9, column=0)
        self.checkbuttonJac.grid(row=0, column=1)
        self.checkbuttonHess.grid(row=1, column=1)
        self.entryXtol.grid(row=2, column=1)
        self.entryGtol.grid(row=3, column=1)
        self.entryBtol.grid(row=4, column=1)
        self.entryMaxiter.grid(row=5, column=1)
        self.entryInitConstrPen.grid(row=6, column=1)
        self.entryInitTrustRad.grid(row=7, column=1)
        self.entryInitBarrPar.grid(row=8, column=1)
        self.entryInitBarrTol.grid(row=9, column=1)
        buttonPrev.grid(row=10, column=0)
        buttonNext.grid(row=10, column=1)       

    def update(self):
         # Initialize default entry values.
        method = self.controller.method
        set_entry(self.entryXtol, str(method.xtol))
        set_entry(self.entryGtol, str(method.gtol))
        set_entry(self.entryBtol, str(method.barrier_tol))
        set_entry(self.entryMaxiter, str(method.maxiter))
        set_entry(self.entryInitConstrPen, str(method.initial_constr_penalty))
        set_entry(self.entryInitTrustRad, str(method.initial_tr_radius))
        set_entry(self.entryInitBarrPar, str(method.initial_barrier_parameter))
        set_entry(self.entryInitBarrTol, str(method.initial_barrier_tolerance))

    def next(self):
        # Parse parameters from GUI.
        try:
            s = self.entryXtol.get().strip()
            xtol = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid xtol value "' + s + '".')
            return
        try:
            s = self.entryGtol.get().strip()
            gtol = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid gtol value "' + s + '".')
            return
        try:
            s = self.entryBtol.get().strip()
            btol = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid barrier tolerance value "' + s + '".')
            return
        try:
            s = self.entryMaxiter.get().strip()
            maxiter = int(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid maximum iterations value "' + s + '".')
            return
        if maxiter < 1:
            messagebox.showerror('ERROR', 'Maximum iterations must be positive.')
            return
        try:
            s = self.entryInitConstrPen.get().strip()
            init_constr_pen = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid initial constrains penalty value "' + s + '".')
            return
        try:
            s = self.entryInitTrustRad.get().strip()
            init_trust_rad = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid initial trust radius value "' + s + '".')
            return
        try:
            s = self.entryInitBarrPar.get().strip()
            init_barr_par = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid initial barrier parameter value "' + s + '".')
            return
        try:
            s = self.entryInitBarrTol.get().strip()
            init_barr_tol = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid initial barrier tolerance value "' + s + '".')
            return

        # Set trust-constr specific parameters.
        method = self.controller.method
        weighted_error = self.controller.weighted_error
        if self.calcJac.get():
            method.calcJac = True
            method.jac = weighted_error.jac
        else:
            method.calcJac = False
            method.jac = '2-point'
        if self.calcHess.get():
            method.calcHess = True
            method.hess = weighted_error.hess
        else:
            method.calcHess = False
            method.hess = BFGS()
        method.xtol = xtol
        method.gtol = gtol
        method.barrier_tol = btol
        method.maxiter = maxiter
        method.initial_constr_penalty = init_constr_pen
        method.initial_tr_radius = init_trust_rad
        method.initial_barrier_parameter = init_barr_par
        method.initial_barrier_tolerance = init_barr_tol

        self.controller.close()

class PageCobyla(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Initialize widgets.
        labelTol = tk.Label(self, text='Tol:')
        labelRhobeg = tk.Label(self, text='Rhobeg:')
        labelMaxiter = tk.Label(self, text='Maxiter:')
        labelCatol = tk.Label(self, text='Catol:')
        self.entryTol = tk.Entry(self)
        self.entryRhobeg = tk.Entry(self)
        self.entryMaxiter = tk.Entry(self)
        self.entryCatol = tk.Entry(self)
        buttonPrev = tk.Button(self, text="<<",
                               command=lambda: controller.show_frame("PageTwo"))
        buttonNext = tk.Button(self, text="FIT", command=self.next)
        
        # Arrange widgets in grid.
        labelTol.grid(row=0, column=0)
        labelRhobeg.grid(row=1, column=0)
        labelMaxiter.grid(row=2, column=0)
        labelCatol.grid(row=3, column=0)
        self.entryTol.grid(row=0, column=1)
        self.entryRhobeg.grid(row=1, column=1)
        self.entryMaxiter.grid(row=2, column=1)
        self.entryCatol.grid(row=3, column=1)
        buttonPrev.grid(row=4, column=0)
        buttonNext.grid(row=4, column=1)

    def update(self):
        # Initialize default entry values.
        method = self.controller.method
        set_entry(self.entryTol, '')
        set_entry(self.entryRhobeg, str(method.rhobeg))
        set_entry(self.entryMaxiter, str(method.maxiter))
        set_entry(self.entryCatol, str(method.catol))

    def next(self):
        # Parse parameters from GUI.
        try:
            s = self.entryTol.get().strip()
            if s == '':
                tol = None
            else:
                tol = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid tol value "' + s + '".')
            return
        try:
            s = self.entryRhobeg.get().strip()
            rhobeg = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid rhobeg value "' + s + '".')
            return
        try:
            s = self.entryMaxiter.get().strip()
            maxiter = int(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid maximum iterations value "' + s + '".')
            return
        if maxiter < 1:
            messagebox.showerror('ERROR', 'Maximum iterations must be positive.')
            return
        try:
            s = self.entryCatol.get().strip()
            catol = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid catol value "' + s + '".')

        # Set COBYLA specific parameters.
        method = self.controller.method
        method.tol = tol
        method.rhobeg = rhobeg
        method.maxiter = maxiter
        method.catol = catol

        self.controller.close()

class PageSlsqp(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Initialize widgets.
        labelTol = tk.Label(self, text='Tol:')
        labelMaxiter = tk.Label(self, text='Maxiter:')
        labelFtol = tk.Label(self, text='Ftol:')
        labelEps = tk.Label(self, text='Eps:')
        self.entryTol = tk.Entry(self)
        self.entryMaxiter = tk.Entry(self)
        self.entryFtol = tk.Entry(self)
        self.entryEps = tk.Entry(self)
        buttonPrev = tk.Button(self, text="<<",
                               command=lambda: controller.show_frame("PageTwo"))
        buttonNext = tk.Button(self, text="FIT", command=self.next)
        
        # Arrange widgets in grid.
        labelTol.grid(row=0, column=0)
        labelMaxiter.grid(row=1, column=0)
        labelFtol.grid(row=2, column=0)
        labelEps.grid(row=3, column=0)
        self.entryTol.grid(row=0, column=1)
        self.entryMaxiter.grid(row=1, column=1)
        self.entryFtol.grid(row=2, column=1)
        self.entryEps.grid(row=3, column=1)
        buttonPrev.grid(row=4, column=0)
        buttonNext.grid(row=4, column=1)

    def update(self):
        # Initialize default entry values.
        method = self.controller.method
        set_entry(self.entryTol, '')
        set_entry(self.entryMaxiter, str(method.maxiter))
        set_entry(self.entryFtol, str(method.ftol))
        set_entry(self.entryEps, str(method.eps))

    def next(self):
        # Parse parameters from GUI.
        try:
            s = self.entryTol.get().strip()
            if s == '':
                tol = None
            else:
                tol = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid tol value "' + s + '".')
            return
        try:
            s = self.entryMaxiter.get().strip()
            maxiter = int(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid maximum iterations value "' + s + '".')
            return
        if maxiter < 1:
            messagebox.showerror('ERROR', 'Maximum iterations must be positive.')
            return
        try:
            s = self.entryFtol.get().strip()
            ftol = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid ftol value "' + s + '".')
            return
        try:
            s = self.entryEps.get().strip()
            eps = float(s)
        except ValueError:
            messagebox.showerror('ERROR', 'Invalid eps value "' + s + '".')

        # Set COBYLA specific parameters.
        method = self.controller.method
        method.tol = tol
        method.maxiter = maxiter
        method.ftol = ftol
        method.eps = eps

        self.controller.close()
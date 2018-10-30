import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, MiniPage, LargeText, Command, NoEscape, \
    Center, Itemize
from pylatex.utils import italic, bold
from pylatex.package import Package
import os

class PdfGenerator:

    def __init__(self):
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'

        self.data_colors = ['b', 'r', 'g']
        self.fit_colors = ['#8080ff', '#ff8080', '#80ff80']
        self.titles = ['UT', 'ET', 'PS']

    def set_datas(self, xdatas, ydatas, model, params):
        self.datas = []
        self.model_datas = []
        for defmode in range(3):
            xdata = xdatas[defmode]
            ydata = ydatas[defmode]
            self.datas.append(zip(xdata, ydata))
            xlin = np.linspace(min(xdata), max(xdata))
            self.model_datas.append(zip(xlin, model.func(defmode, xlin, *tuple(params))))

    def generate(self):
        geometry_options = {"lmargin": "2.5cm", "rmargin": "2.5cm", "tmargin": "1cm"}
        doc = Document(geometry_options=geometry_options)
        doc.packages.append(Package('float'))
        with doc.create(Center()):
            doc.append(LargeText('Optimization Report'))
        with doc.create(Figure(position='H')) as plot:
            self.plot()
            plot.add_plot(width=NoEscape('17cm'))
        with doc.create(Section('Input:', False)):
            with doc.create(Itemize()) as input:
                input.add_item("Filenames:")
                with doc.create(Itemize()) as filenames:
                    for defmode in range(3):
                        if self.plot_defmode[defmode]:
                            filenames.add_item(self.titles[defmode] \
                                + ': ' + self.filenames[defmode])
                input.add_item("Weights:")
                with doc.create(Itemize()) as weights:
                    for defmode in range(3):
                        if self.plot_defmode[defmode]:
                            weights.add_item('{}: {:.4g}'.format( \
                                self.titles[defmode], self.weights[defmode]))
                input.add_item("Model: " + self.model.name)
                input.add_item("Error function: " + self.fit_error.name)
                input.add_item("Method: " + self.method.name)
                with doc.create(Itemize()) as methodparams:
                    for methodparam in self.method.print_params().split('\n'):
                        methodparams.add_item(methodparam)
        with doc.create(Section('Output:', False)):
            with doc.create(Itemize()) as output:
                s = '\\text{Errors: (}' + self.fit_error.name_latex + '\\text{)}'
                output.add_item(Math(inline=True, data=s, escape=False))
                with doc.create(Itemize()) as errorlist:
                    weighted_error = 0.0
                    for defmode in range(3):
                        error = self.fit_error(self.model.getfunc(defmode),
                                               self.model.getjac(defmode),
                                               self.model.gethess(defmode),
                                               self.xdatas[defmode], self.ydatas[defmode])
                        errval = error.objfunc(self.params)
                        weighted_error += errval * self.weights[defmode]
                        errorlist.add_item('{}: {:.4g}'.format(self.titles[defmode], errval))
                    errorlist.add_item('Weighted: {:.4g}'.format(weighted_error))
                output.add_item('Parameters:')
                with doc.create(Itemize()) as parameters:
                    for pname, p in zip(self.model.paramnames_latex, self.params):
                        s = '{}: {:.4g}'.format(pname, p)
                        parameters.add_item(Math(inline=True, data=s, escape=False))

        doc.generate_pdf('report', clean_tex=False, compiler='pdflatex')

    def plot(self):
        plt.clf()

        for defmode in range(3):
            # Plot read data as points.
            if self.xdatas[defmode] is None or not self.plot_defmode[defmode]:
                continue
            
            xdata = self.plot_deformation.from_stretch(self.xdatas[defmode])
            ydata = self.plot_stress.from_true_stress(self.ydatas[defmode], self.xdatas[defmode])
            error = self.plot_error(self.model.getfunc(defmode), self.model.getjac(defmode),
                                    self.model.gethess(defmode),
                                    self.xdatas[defmode], self.ydatas[defmode])

            label = self.titles[defmode]
            label += ' - {}={:.4g}'.format(self.plot_error.shortname, error.objfunc(self.params))
            plt.plot(xdata, ydata, marker='o', linestyle='',
                            color=self.data_colors[defmode], label=label)
        
            # Plot fitted models as curves.
            xlin = np.linspace(min(xdata), max(xdata))
            y = np.zeros(len(xlin))
            for i in range(len(xlin)):
                stretch = self.plot_deformation.to_stretch(xlin[i])
                y[i] = self.plot_stress.from_true_stress(self.model.getfunc(defmode)(stretch, *self.params), stretch)
            plt.plot(xlin, y, marker='', linestyle='-', color=self.fit_colors[defmode])
                        
        plt.xlabel(self.plot_deformation.name)
        plt.ylabel(self.plot_stress.name)
        plt.legend()
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

class TexGenerator:

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
        geometry_options = {"paperwidth": "10cm", "paperheight": "8cm", "margin": "0cm"}
        doc = Document(geometry_options=geometry_options)
        doc.packages.append(Package('float'))
        with doc.create(Figure(position='H')) as plot:
            self.plot()
            plot.add_plot(width=NoEscape('10cm'), height=NoEscape('8cm'))
        
        filename = os.path.splitext(self.filename)[0]
        doc.generate_pdf(filename, clean_tex=False, compiler='pdflatex')

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
            label += ' - {} = {:.4g}'.format(self.plot_error.shortname, error.objfunc(self.params))
            plt.plot(xdata, ydata, marker='o', linestyle='',
                            color=self.data_colors[defmode], label=label)
        
            # Plot fitted models as curves.
            xlin = np.linspace(min(xdata), max(xdata))
            y = np.zeros(len(xlin))
            for i in range(len(xlin)):
                stretch = self.plot_deformation.to_stretch(xlin[i])
                y[i] = self.plot_stress.from_true_stress(self.model.getfunc(defmode)(stretch, *self.params), stretch)
            plt.plot(xlin, y, marker='', linestyle='-', color=self.fit_colors[defmode])
        
        s = '\n'.join(['${}$ = {: .4g}'.format(pname, p) for pname, p in zip(self.model.paramnames_latex, self.params)])
        plt.text(0.75, 0.05, s, horizontalalignment='left', verticalalignment='bottom',
                 transform=plt.gca().transAxes)
        plt.xlabel(self.plot_deformation.name)
        plt.ylabel(self.plot_stress.name)
        plt.legend()

#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.14
# In conjunction with Tcl version 8.6
#    Oct 31, 2018 01:50:10 AM

import sys

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import gui_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = Tk()
    gui_support.set_Tk_var()
    top = HyperCalib (root)
    gui_support.init(root, top)
    root.mainloop()

w = None
def create_HyperCalib(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = Toplevel (root)
    gui_support.set_Tk_var()
    top = HyperCalib (w)
    gui_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_HyperCalib():
    global w
    w.destroy()
    w = None


class HyperCalib:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85' 
        _ana2color = '#d9d9d9' # X11 color: 'gray85' 
        font9 = "-family {Segoe UI} -size 9 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"

        top.geometry("985x694+50+50")
        top.title("HyperCalib")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")



        self.ButtonProcess = Button(top)
        self.ButtonProcess.grid(row=0, column=0, columnspan=2, sticky=W+E, padx=(10,5), pady=(10,5))
        self.ButtonProcess.configure(activebackground="#d9d9d9")
        self.ButtonProcess.configure(activeforeground="#000000")
        self.ButtonProcess.configure(background="#d9d9d9")
        self.ButtonProcess.configure(command=gui_support.ButtonProcess_Click)
        self.ButtonProcess.configure(disabledforeground="#a3a3a3")
        self.ButtonProcess.configure(foreground="#000000")
        self.ButtonProcess.configure(highlightbackground="#d9d9d9")
        self.ButtonProcess.configure(highlightcolor="black")
        self.ButtonProcess.configure(pady="0")
        self.ButtonProcess.configure(text='''Process File''')

        self.FramePlot = Frame(top)
        self.FramePlot.grid(row=4, column=0, columnspan=2, sticky=W+E+N+S, padx=(10,5), pady=(5,5))
        self.FramePlot.configure(relief=GROOVE)
        self.FramePlot.configure(borderwidth="2")
        self.FramePlot.configure(relief=GROOVE)
        self.FramePlot.configure(background="#d9d9d9")
        self.FramePlot.configure(highlightbackground="#d9d9d9")
        self.FramePlot.configure(highlightcolor="black")

        self.FrameNavigationPlot = Frame(top)
        self.FrameNavigationPlot.grid(row=5, column=0, columnspan=2, sticky=W+E, padx=(10,5), pady=(5,10))
        self.FrameNavigationPlot.configure(relief=GROOVE)
        self.FrameNavigationPlot.configure(borderwidth="2")
        self.FrameNavigationPlot.configure(relief=GROOVE)
        self.FrameNavigationPlot.configure(background="#d9d9d9")
        self.FrameNavigationPlot.configure(highlightbackground="#d9d9d9")
        self.FrameNavigationPlot.configure(highlightcolor="black")

        self.ButtonFitModel = Button(top)
        self.ButtonFitModel.grid(row=1, column=0, columnspan=2, sticky=W+E, padx=(10,5), pady=(5,5))
        self.ButtonFitModel.configure(activebackground="#d9d9d9")
        self.ButtonFitModel.configure(activeforeground="#000000")
        self.ButtonFitModel.configure(background="#d9d9d9")
        self.ButtonFitModel.configure(command=gui_support.ButtonFitModel_Click)
        self.ButtonFitModel.configure(disabledforeground="#a3a3a3")
        self.ButtonFitModel.configure(foreground="#000000")
        self.ButtonFitModel.configure(highlightbackground="#d9d9d9")
        self.ButtonFitModel.configure(highlightcolor="black")
        self.ButtonFitModel.configure(pady="0")
        self.ButtonFitModel.configure(state=DISABLED)
        self.ButtonFitModel.configure(text='''Fit model''')

        self.TextLog = Text(top)
        self.TextLog.grid(row=0, rowspan=4, column=2, sticky=N+S+W+E, padx=(5,10), pady=(10,5))
        self.TextLog.configure(background="white")
        self.TextLog.configure(font="TkTextFont")
        self.TextLog.configure(foreground="#000000")
        self.TextLog.configure(highlightbackground="#d9d9d9")
        self.TextLog.configure(highlightcolor="black")
        self.TextLog.configure(insertbackground="black")
        self.TextLog.configure(selectbackground="#c4c4c4")
        self.TextLog.configure(selectforeground="black")
        self.TextLog.configure(state=DISABLED)
        self.TextLog.configure(wrap=WORD)
        self.TextLog.configure(height=10)

        self.TextState = Text(top)
        self.TextState.grid(row=2, column=0, columnspan=2, sticky=N+S+W+E, padx=(10,5), pady=(5,5))
        self.TextState.configure(background="white")
        self.TextState.configure(font=font9)
        self.TextState.configure(foreground="black")
        self.TextState.configure(highlightbackground="#d9d9d9")
        self.TextState.configure(highlightcolor="black")
        self.TextState.configure(insertbackground="black")
        self.TextState.configure(selectbackground="#c4c4c4")
        self.TextState.configure(selectforeground="black")
        self.TextState.configure(state=DISABLED)
        self.TextState.configure(wrap=WORD)
        self.TextState.configure(height=10)

        self.FrameError = Frame(top)
        self.FrameError.grid(row=4, column=2, sticky=N+S+W+E, padx=(5,10), pady=(5,5))
        self.FrameError.configure(relief=GROOVE)
        self.FrameError.configure(borderwidth="2")
        self.FrameError.configure(relief=GROOVE)
        self.FrameError.configure(background="#d9d9d9")
        self.FrameError.configure(highlightbackground="#d9d9d9")
        self.FrameError.configure(highlightcolor="black")

        self.FrameNavigationError = Frame(top)
        self.FrameNavigationError.grid(row=5, column=2, sticky=W+E, padx=(5,10), pady=(5,10))
        self.FrameNavigationError.configure(relief=GROOVE)
        self.FrameNavigationError.configure(borderwidth="2")
        self.FrameNavigationError.configure(relief=GROOVE)
        self.FrameNavigationError.configure(background="#d9d9d9")
        self.FrameNavigationError.configure(highlightbackground="#d9d9d9")
        self.FrameNavigationError.configure(highlightcolor="black")

        self.ButtonPlotSettings = Button(top)
        self.ButtonPlotSettings.grid(row=3, column=0, sticky=W+E, padx=(10,5), pady=(5,5))
        self.ButtonPlotSettings.configure(activebackground="#d9d9d9")
        self.ButtonPlotSettings.configure(activeforeground="#000000")
        self.ButtonPlotSettings.configure(background="#d9d9d9")
        self.ButtonPlotSettings.configure(command=gui_support.ButtonPlotSettings_Click)
        self.ButtonPlotSettings.configure(disabledforeground="#a3a3a3")
        self.ButtonPlotSettings.configure(foreground="#000000")
        self.ButtonPlotSettings.configure(highlightbackground="#d9d9d9")
        self.ButtonPlotSettings.configure(highlightcolor="black")
        self.ButtonPlotSettings.configure(pady="0")
        self.ButtonPlotSettings.configure(text='''Plot settings''')

        self.ButtonCreateReport = Button(top)
        self.ButtonCreateReport.grid(row=3, column=1, sticky=W+E, padx=(5,5), pady=(5,5))
        self.ButtonCreateReport.configure(activebackground="#d9d9d9")
        self.ButtonCreateReport.configure(activeforeground="#000000")
        self.ButtonCreateReport.configure(background="#d9d9d9")
        self.ButtonCreateReport.configure(command=gui_support.ButtonCreateReport_Click)
        self.ButtonCreateReport.configure(disabledforeground="#a3a3a3")
        self.ButtonCreateReport.configure(foreground="#000000")
        self.ButtonCreateReport.configure(highlightbackground="#d9d9d9")
        self.ButtonCreateReport.configure(highlightcolor="black")
        self.ButtonCreateReport.configure(pady="0")
        self.ButtonCreateReport.configure(state=DISABLED)
        self.ButtonCreateReport.configure(text='''Create report''')

        top.grid_rowconfigure(2, weight=1)
        top.grid_rowconfigure(4, weight=1)
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(2, weight=2)



if __name__ == '__main__':
    vp_start_gui()




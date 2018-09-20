#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.14
# In conjunction with Tcl version 8.6
#    Sep 21, 2018 12:06:56 AM

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
    top = Hyperelastic_curve_fit (root)
    gui_support.init(root, top)
    root.mainloop()

w = None
def create_Hyperelastic_curve_fit(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = Toplevel (root)
    gui_support.set_Tk_var()
    top = Hyperelastic_curve_fit (w)
    gui_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Hyperelastic_curve_fit():
    global w
    w.destroy()
    w = None


class Hyperelastic_curve_fit:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85' 
        _ana2color = '#d9d9d9' # X11 color: 'gray85' 
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("997x579+473+263")
        top.title("Hyperelastic curve fit")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")



        self.EntryFilename = Entry(top)
        self.EntryFilename.place(x=100, y=20,height=24, width=204)
        self.EntryFilename.configure(background="white")
        self.EntryFilename.configure(disabledforeground="#a3a3a3")
        self.EntryFilename.configure(font="TkFixedFont")
        self.EntryFilename.configure(foreground="#000000")
        self.EntryFilename.configure(highlightbackground="#d9d9d9")
        self.EntryFilename.configure(highlightcolor="black")
        self.EntryFilename.configure(insertbackground="black")
        self.EntryFilename.configure(selectbackground="#c4c4c4")
        self.EntryFilename.configure(selectforeground="black")

        self.Label1 = Label(top)
        self.Label1.place(x=20, y=20,  height=30, width=70)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(activeforeground="black")
        self.Label1.configure(anchor=NW)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(highlightbackground="#d9d9d9")
        self.Label1.configure(highlightcolor="black")
        self.Label1.configure(text='''Filename:''')
        self.Label1.configure(width=70)

        self.ButtonProcess = Button(top)
        self.ButtonProcess.place(x=20, y=120,  height=26, width=470)
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
        self.FramePlot.place(x=525, y=277, height=232, width=449)
        self.FramePlot.configure(relief=GROOVE)
        self.FramePlot.configure(borderwidth="2")
        self.FramePlot.configure(relief=GROOVE)
        self.FramePlot.configure(background="#d9d9d9")
        self.FramePlot.configure(highlightbackground="#d9d9d9")
        self.FramePlot.configure(highlightcolor="black")
        self.FramePlot.configure(width=470)

        self.EntrySamples = Entry(top)
        self.EntrySamples.place(x=100, y=50,height=24, width=204)
        self.EntrySamples.configure(background="white")
        self.EntrySamples.configure(disabledforeground="#a3a3a3")
        self.EntrySamples.configure(font="TkFixedFont")
        self.EntrySamples.configure(foreground="#000000")
        self.EntrySamples.configure(highlightbackground="#d9d9d9")
        self.EntrySamples.configure(highlightcolor="black")
        self.EntrySamples.configure(insertbackground="black")
        self.EntrySamples.configure(selectbackground="#c4c4c4")
        self.EntrySamples.configure(selectforeground="black")

        self.Label2 = Label(top)
        self.Label2.place(x=20, y=50,  height=26, width=65)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(anchor=NW)
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Samples:''')

        self.FrameNavigation = Frame(top)
        self.FrameNavigation.place(x=525, y=518, height=41, width=449)
        self.FrameNavigation.configure(relief=GROOVE)
        self.FrameNavigation.configure(borderwidth="2")
        self.FrameNavigation.configure(relief=GROOVE)
        self.FrameNavigation.configure(background="#d9d9d9")
        self.FrameNavigation.configure(highlightbackground="#d9d9d9")
        self.FrameNavigation.configure(highlightcolor="black")
        self.FrameNavigation.configure(width=465)

        self.Label3 = Label(top)
        self.Label3.place(x=20, y=80,  height=26, width=71)
        self.Label3.configure(activebackground="#f9f9f9")
        self.Label3.configure(activeforeground="black")
        self.Label3.configure(anchor=NW)
        self.Label3.configure(background="#d9d9d9")
        self.Label3.configure(disabledforeground="#a3a3a3")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(highlightbackground="#d9d9d9")
        self.Label3.configure(highlightcolor="black")
        self.Label3.configure(text='''Delimiter:''')

        self.EntryDelimiter = Entry(top)
        self.EntryDelimiter.place(x=100, y=80,height=24, width=204)
        self.EntryDelimiter.configure(background="white")
        self.EntryDelimiter.configure(disabledforeground="#a3a3a3")
        self.EntryDelimiter.configure(font="TkFixedFont")
        self.EntryDelimiter.configure(foreground="#000000")
        self.EntryDelimiter.configure(highlightbackground="#d9d9d9")
        self.EntryDelimiter.configure(highlightcolor="black")
        self.EntryDelimiter.configure(insertbackground="black")
        self.EntryDelimiter.configure(selectbackground="#c4c4c4")
        self.EntryDelimiter.configure(selectforeground="black")

        self.RadiobuttonUT = Radiobutton(top)
        self.RadiobuttonUT.place(x=320, y=20, height=30, width=150)
        self.RadiobuttonUT.configure(activebackground="#d9d9d9")
        self.RadiobuttonUT.configure(activeforeground="#000000")
        self.RadiobuttonUT.configure(anchor=NW)
        self.RadiobuttonUT.configure(background="#d9d9d9")
        self.RadiobuttonUT.configure(disabledforeground="#a3a3a3")
        self.RadiobuttonUT.configure(foreground="#000000")
        self.RadiobuttonUT.configure(highlightbackground="#d9d9d9")
        self.RadiobuttonUT.configure(highlightcolor="black")
        self.RadiobuttonUT.configure(justify=LEFT)
        self.RadiobuttonUT.configure(text='''Uniaxial tension''')
        self.RadiobuttonUT.configure(value="0")
        self.RadiobuttonUT.configure(variable=gui_support.gui_support.radiovar)

        self.RadiobuttonET = Radiobutton(top)
        self.RadiobuttonET.place(x=320, y=50, height=30, width=150)
        self.RadiobuttonET.configure(activebackground="#d9d9d9")
        self.RadiobuttonET.configure(activeforeground="#000000")
        self.RadiobuttonET.configure(anchor=NW)
        self.RadiobuttonET.configure(background="#d9d9d9")
        self.RadiobuttonET.configure(disabledforeground="#a3a3a3")
        self.RadiobuttonET.configure(foreground="#000000")
        self.RadiobuttonET.configure(highlightbackground="#d9d9d9")
        self.RadiobuttonET.configure(highlightcolor="black")
        self.RadiobuttonET.configure(justify=LEFT)
        self.RadiobuttonET.configure(text='''Equibiaxial tension''')
        self.RadiobuttonET.configure(value="1")
        self.RadiobuttonET.configure(variable=gui_support.gui_support.radiovar)

        self.RadiobuttonPS = Radiobutton(top)
        self.RadiobuttonPS.place(x=320, y=80, height=30, width=150)
        self.RadiobuttonPS.configure(activebackground="#d9d9d9")
        self.RadiobuttonPS.configure(activeforeground="#000000")
        self.RadiobuttonPS.configure(anchor=NW)
        self.RadiobuttonPS.configure(background="#d9d9d9")
        self.RadiobuttonPS.configure(disabledforeground="#a3a3a3")
        self.RadiobuttonPS.configure(foreground="#000000")
        self.RadiobuttonPS.configure(highlightbackground="#d9d9d9")
        self.RadiobuttonPS.configure(highlightcolor="black")
        self.RadiobuttonPS.configure(justify=LEFT)
        self.RadiobuttonPS.configure(text='''Pure shear''')
        self.RadiobuttonPS.configure(value="2")
        self.RadiobuttonPS.configure(variable=gui_support.gui_support.radiovar)

        self.ButtonFitModel = Button(top)
        self.ButtonFitModel.place(x=20, y=370,  height=26, width=470)
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

        self.TextParameters = Text(top)
        self.TextParameters.place(x=525, y=30, height=204, width=449)
        self.TextParameters.configure(background="white")
        self.TextParameters.configure(font="TkTextFont")
        self.TextParameters.configure(foreground="#000000")
        self.TextParameters.configure(highlightbackground="#d9d9d9")
        self.TextParameters.configure(highlightcolor="black")
        self.TextParameters.configure(insertbackground="black")
        self.TextParameters.configure(selectbackground="#c4c4c4")
        self.TextParameters.configure(selectforeground="black")
        self.TextParameters.configure(state=DISABLED)
        self.TextParameters.configure(width=460)
        self.TextParameters.configure(wrap=WORD)

        self.CheckbuttonFitUT = Checkbutton(top)
        self.CheckbuttonFitUT.place(x=20, y=270, height=31, width=134)
        self.CheckbuttonFitUT.configure(activebackground="#d9d9d9")
        self.CheckbuttonFitUT.configure(activeforeground="#000000")
        self.CheckbuttonFitUT.configure(anchor=NW)
        self.CheckbuttonFitUT.configure(background="#d9d9d9")
        self.CheckbuttonFitUT.configure(disabledforeground="#a3a3a3")
        self.CheckbuttonFitUT.configure(foreground="#000000")
        self.CheckbuttonFitUT.configure(highlightbackground="#d9d9d9")
        self.CheckbuttonFitUT.configure(highlightcolor="black")
        self.CheckbuttonFitUT.configure(justify=LEFT)
        self.CheckbuttonFitUT.configure(state=DISABLED)
        self.CheckbuttonFitUT.configure(text='''Uniaxial tension''')
        self.CheckbuttonFitUT.configure(variable=gui_support.gui_support.isFit[0])

        self.CheckbuttonFitET = Checkbutton(top)
        self.CheckbuttonFitET.place(x=20, y=300, height=31, width=154)
        self.CheckbuttonFitET.configure(activebackground="#d9d9d9")
        self.CheckbuttonFitET.configure(activeforeground="#000000")
        self.CheckbuttonFitET.configure(anchor=NW)
        self.CheckbuttonFitET.configure(background="#d9d9d9")
        self.CheckbuttonFitET.configure(disabledforeground="#a3a3a3")
        self.CheckbuttonFitET.configure(foreground="#000000")
        self.CheckbuttonFitET.configure(highlightbackground="#d9d9d9")
        self.CheckbuttonFitET.configure(highlightcolor="black")
        self.CheckbuttonFitET.configure(justify=LEFT)
        self.CheckbuttonFitET.configure(state=DISABLED)
        self.CheckbuttonFitET.configure(text='''Equibiaxial tension''')
        self.CheckbuttonFitET.configure(variable=gui_support.gui_support.isFit[1])

        self.CheckbuttonFitPS = Checkbutton(top)
        self.CheckbuttonFitPS.place(x=20, y=330, height=31, width=97)
        self.CheckbuttonFitPS.configure(activebackground="#d9d9d9")
        self.CheckbuttonFitPS.configure(activeforeground="#000000")
        self.CheckbuttonFitPS.configure(anchor=NW)
        self.CheckbuttonFitPS.configure(background="#d9d9d9")
        self.CheckbuttonFitPS.configure(disabledforeground="#a3a3a3")
        self.CheckbuttonFitPS.configure(foreground="#000000")
        self.CheckbuttonFitPS.configure(highlightbackground="#d9d9d9")
        self.CheckbuttonFitPS.configure(highlightcolor="black")
        self.CheckbuttonFitPS.configure(justify=LEFT)
        self.CheckbuttonFitPS.configure(state=DISABLED)
        self.CheckbuttonFitPS.configure(text='''Pure shear''')
        self.CheckbuttonFitPS.configure(variable=gui_support.gui_support.isFit[2])

        self.EntryUT = Entry(top)
        self.EntryUT.place(x=280, y=270,height=24, width=204)
        self.EntryUT.configure(background="white")
        self.EntryUT.configure(disabledforeground="#a3a3a3")
        self.EntryUT.configure(font="TkFixedFont")
        self.EntryUT.configure(foreground="#000000")
        self.EntryUT.configure(highlightbackground="#d9d9d9")
        self.EntryUT.configure(highlightcolor="black")
        self.EntryUT.configure(insertbackground="black")
        self.EntryUT.configure(selectbackground="#c4c4c4")
        self.EntryUT.configure(selectforeground="black")
        self.EntryUT.configure(state=DISABLED)

        self.EntryET = Entry(top)
        self.EntryET.place(x=280, y=300,height=24, width=204)
        self.EntryET.configure(background="white")
        self.EntryET.configure(disabledforeground="#a3a3a3")
        self.EntryET.configure(font="TkFixedFont")
        self.EntryET.configure(foreground="#000000")
        self.EntryET.configure(highlightbackground="#d9d9d9")
        self.EntryET.configure(highlightcolor="black")
        self.EntryET.configure(insertbackground="black")
        self.EntryET.configure(selectbackground="#c4c4c4")
        self.EntryET.configure(selectforeground="black")
        self.EntryET.configure(state=DISABLED)

        self.EntryPS = Entry(top)
        self.EntryPS.place(x=280, y=330,height=24, width=204)
        self.EntryPS.configure(background="white")
        self.EntryPS.configure(disabledforeground="#a3a3a3")
        self.EntryPS.configure(font="TkFixedFont")
        self.EntryPS.configure(foreground="#000000")
        self.EntryPS.configure(highlightbackground="#d9d9d9")
        self.EntryPS.configure(highlightcolor="black")
        self.EntryPS.configure(insertbackground="black")
        self.EntryPS.configure(selectbackground="#c4c4c4")
        self.EntryPS.configure(selectforeground="black")
        self.EntryPS.configure(state=DISABLED)

        self.CheckbuttonPlotUT = Checkbutton(top)
        self.CheckbuttonPlotUT.place(x=110, y=460, height=31, width=134)
        self.CheckbuttonPlotUT.configure(activebackground="#d9d9d9")
        self.CheckbuttonPlotUT.configure(activeforeground="#000000")
        self.CheckbuttonPlotUT.configure(anchor=NW)
        self.CheckbuttonPlotUT.configure(background="#d9d9d9")
        self.CheckbuttonPlotUT.configure(disabledforeground="#a3a3a3")
        self.CheckbuttonPlotUT.configure(foreground="#000000")
        self.CheckbuttonPlotUT.configure(highlightbackground="#d9d9d9")
        self.CheckbuttonPlotUT.configure(highlightcolor="black")
        self.CheckbuttonPlotUT.configure(justify=LEFT)
        self.CheckbuttonPlotUT.configure(state=DISABLED)
        self.CheckbuttonPlotUT.configure(text='''Uniaxial tension''')
        self.CheckbuttonPlotUT.configure(variable=gui_support.gui_support.isPlot[0])

        self.CheckbuttonPlotET = Checkbutton(top)
        self.CheckbuttonPlotET.place(x=110, y=490, height=31, width=154)
        self.CheckbuttonPlotET.configure(activebackground="#d9d9d9")
        self.CheckbuttonPlotET.configure(activeforeground="#000000")
        self.CheckbuttonPlotET.configure(anchor=NW)
        self.CheckbuttonPlotET.configure(background="#d9d9d9")
        self.CheckbuttonPlotET.configure(disabledforeground="#a3a3a3")
        self.CheckbuttonPlotET.configure(foreground="#000000")
        self.CheckbuttonPlotET.configure(highlightbackground="#d9d9d9")
        self.CheckbuttonPlotET.configure(highlightcolor="black")
        self.CheckbuttonPlotET.configure(justify=LEFT)
        self.CheckbuttonPlotET.configure(state=DISABLED)
        self.CheckbuttonPlotET.configure(text='''Equibiaxial tension''')
        self.CheckbuttonPlotET.configure(variable=gui_support.gui_support.isPlot[1])

        self.CheckbuttonPlotPS = Checkbutton(top)
        self.CheckbuttonPlotPS.place(x=110, y=520, height=31, width=97)
        self.CheckbuttonPlotPS.configure(activebackground="#d9d9d9")
        self.CheckbuttonPlotPS.configure(activeforeground="#000000")
        self.CheckbuttonPlotPS.configure(anchor=NW)
        self.CheckbuttonPlotPS.configure(background="#d9d9d9")
        self.CheckbuttonPlotPS.configure(disabledforeground="#a3a3a3")
        self.CheckbuttonPlotPS.configure(foreground="#000000")
        self.CheckbuttonPlotPS.configure(highlightbackground="#d9d9d9")
        self.CheckbuttonPlotPS.configure(highlightcolor="black")
        self.CheckbuttonPlotPS.configure(justify=LEFT)
        self.CheckbuttonPlotPS.configure(state=DISABLED)
        self.CheckbuttonPlotPS.configure(text='''Pure shear''')
        self.CheckbuttonPlotPS.configure(variable=gui_support.gui_support.isPlot[2])

        self.Label4 = Label(top)
        self.Label4.place(x=50, y=490,  height=26, width=35)
        self.Label4.configure(activebackground="#f9f9f9")
        self.Label4.configure(activeforeground="black")
        self.Label4.configure(anchor=NW)
        self.Label4.configure(background="#d9d9d9")
        self.Label4.configure(disabledforeground="#a3a3a3")
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(highlightbackground="#d9d9d9")
        self.Label4.configure(highlightcolor="black")
        self.Label4.configure(text='''Plot:''')

        self.TComboboxModel = ttk.Combobox(top)
        self.TComboboxModel.place(x=130, y=170, height=26, width=357)
        READONLY = 'readonly'
        self.TComboboxModel.configure(state=READONLY)
        self.TComboboxModel.configure(takefocus="")

        self.Label5 = Label(top)
        self.Label5.place(x=20, y=170,  height=26, width=52)
        self.Label5.configure(activebackground="#f9f9f9")
        self.Label5.configure(activeforeground="black")
        self.Label5.configure(anchor=NW)
        self.Label5.configure(background="#d9d9d9")
        self.Label5.configure(disabledforeground="#a3a3a3")
        self.Label5.configure(foreground="#000000")
        self.Label5.configure(highlightbackground="#d9d9d9")
        self.Label5.configure(highlightcolor="black")
        self.Label5.configure(text='''Model:''')

        self.Label6 = Label(top)
        self.Label6.place(x=30, y=490,  height=26, width=6)
        self.Label6.configure(activebackground="#f9f9f9")
        self.Label6.configure(activeforeground="black")
        self.Label6.configure(background="#d9d9d9")
        self.Label6.configure(disabledforeground="#a3a3a3")
        self.Label6.configure(foreground="#000000")
        self.Label6.configure(highlightbackground="#d9d9d9")
        self.Label6.configure(highlightcolor="black")
        self.Label6.configure(textvariable=gui_support.dummy)

        self.Label7 = Label(top)
        self.Label7.place(x=20, y=200,  height=26, width=99)
        self.Label7.configure(activebackground="#f9f9f9")
        self.Label7.configure(activeforeground="black")
        self.Label7.configure(anchor=NW)
        self.Label7.configure(background="#d9d9d9")
        self.Label7.configure(disabledforeground="#a3a3a3")
        self.Label7.configure(foreground="#000000")
        self.Label7.configure(highlightbackground="#d9d9d9")
        self.Label7.configure(highlightcolor="black")
        self.Label7.configure(text='''Error function:''')

        self.TComboboxError = ttk.Combobox(top)
        self.TComboboxError.place(x=130, y=200, height=26, width=357)
        READONLY = 'readonly'
        self.TComboboxError.configure(state=READONLY)
        self.TComboboxError.configure(takefocus="")






if __name__ == '__main__':
    vp_start_gui()




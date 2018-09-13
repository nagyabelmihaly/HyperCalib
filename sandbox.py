import tkinter as tk
from tkinter import ttk

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        self.combobox = ttk.Combobox(self, state="readonly")
        self.combobox['values'] = ['Aladár', 'Béla', 'Cecil']
        self.combobox.grid()

app = Application()
app.master.title('Sample application')
app.mainloop()
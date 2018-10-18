import tkinter as tk

def set_text(text, value):
    """Sets the content of the given Text widget.
    ----------
    Keyword arguments:
    text -- The Text whose text should be set.
    value -- The new text string.
    """
    old_state = text['state']
    text.config(state=tk.NORMAL)
    text.delete(1.0, tk.END)
    text.insert(tk.END, value)
    text.config(state=old_state)

def add_text(text, value):
    old_state = text['state']
    text.config(state=tk.NORMAL)
    text.insert(tk.END, value)
    text.config(state=old_state)

def set_entry(entry, value):
    """Sets the content of the given Entry widget.
    ----------
    Keyword arguments:
    entry -- The Entry whose text should be set.
    value -- The new text string.
    """
    old_state = entry['state']
    entry.config(state=tk.NORMAL)
    entry.delete(0, tk.END)
    entry.insert(tk.END, value)
    entry.config(state=old_state)

def is_empty_or_whitespace(string):
    if not string:
        return True    
    return string.isspace()

def try_parse_float(value):
    try:
        return float(value)
    except ValueError:
        return value
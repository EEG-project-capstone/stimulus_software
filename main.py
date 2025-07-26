# main.py
import tkinter as tk
from lib.app import TkApp

if __name__ == "__main__":
    root = tk.Tk()
    app = TkApp(root)
    root.mainloop()
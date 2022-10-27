import tkinter as tk
from tkinter import ttk
import webbrowser
import os


def cmd1():
    os.system('cmd /c "docker pull heartexlabs/label-studio:latest"')


def cmd2():
    os.system('cmd /c "docker run -d -p 8080:8080 -v ./mydata:/label-studio/data heartexlabs/label-studio:latest"')


def cmd3():
    webbrowser.open("http://localhost:8080/")


class GUIDraft(tk.Tk):
    def __init__(self):
        super().__init__()
        self.is_world = tk.IntVar()
        self.title("GUI Draft")
        self.geometry('600x200')
        button_a = tk.Button(self, text="Get Image", command=cmd1)  # keeps overloading the gui for some reason
        tk.Label(self, text="Start Server will start a new server, but it also crashes this window."
                            "\nJust open the GUI again to open the server."
                            "\n Close the server from the Docker window.").pack()
        button_b = tk.Button(self, text="Start Server", command=cmd2)
        button_c = tk.Button(self, text="Open Server", command=cmd3)
        button_a.pack()
        button_b.pack()
        button_c.pack()


example = GUIDraft()
example.mainloop()


from gui_window import App
import sys

def on_closing():
    # Something is wrong with image rendering I guess?
    app.destroy()
    sys.exit()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()
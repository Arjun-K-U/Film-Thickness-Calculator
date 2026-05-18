# Thin Film GUI with User-Input Refractive Indices, Thickness Entry, and Fixed Y-Axis

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
from tqdm import tqdm

# Initialization
root = tk.Tk()
root.withdraw()

# Load file (Happens only once)
file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
if not file_path:
    print("No file selected. Exiting...")
    os._exit(0)

# Read data
try:
    df = pd.read_csv(file_path, skiprows=19, header=None, usecols=[0, 1])
    df.columns = ["Wavelength", "Transmittance"]
    df.dropna(inplace=True)
    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
    df['Transmittance'] = pd.to_numeric(df['Transmittance'], errors='coerce')
    df.dropna(inplace=True)
except Exception as e:
    print(f"Error loading CSV: {e}")
    os._exit(0)

# --- CONTINUOUS LOOP STARTS HERE ---
while True:
    # Ask wavelength range & refractive indices
    min_wl = simpledialog.askfloat("Min Wavelength", "Enter min wavelength (nm):")
    max_wl = simpledialog.askfloat("Max Wavelength", "Enter max wavelength (nm):")
    n1 = simpledialog.askfloat("Refractive Index", "Enter n1 (Ambient):", initialvalue=1.0)
    n2 = simpledialog.askfloat("Refractive Index", "Enter n2 (Film):", initialvalue=1.6)
    n3 = simpledialog.askfloat("Refractive Index", "Enter n3 (Substrate):", initialvalue=1.5)
    
    # Exit loop if the user hits "Cancel" on any dialog
    if any(x is None for x in [min_wl, max_wl, n1, n2, n3]):
        print("Input canceled. Exiting...")
        break

    n_vals = {'n1': n1, 'n2': n2, 'n3': n3}

    step = 0.5
    wl_exp = df[(df['Wavelength'] >= min_wl) & (df['Wavelength'] <= max_wl)]['Wavelength'].values
    T_exp = df[(df['Wavelength'] >= min_wl) & (df['Wavelength'] <= max_wl)]['Transmittance'].values
    T_exp = T_exp / np.max(T_exp)
    wl_theo = np.arange(min_wl, max_wl + step, step)
    wl_theo_m = wl_theo * 1e-9

    # Model
    def transmittance(thickness, n1, n2, n3):
        k0 = 2 * np.pi / wl_theo_m
        delta = n2 * k0 * thickness
        r01 = (n1 - n2) / (n1 + n2)
        r12 = (n2 - n3) / (n2 + n3)
        t01 = 2 * n1 / (n1 + n2)
        t12 = 2 * n2 / (n2 + n3)

        T = np.zeros_like(wl_theo)
        for i in range(len(wl_theo)):
            P = np.array([[np.exp(-1j * delta[i]), 0], [0, np.exp(1j * delta[i])]])
            M01 = np.array([[1, r01], [r01, 1]]) / t01
            M12 = np.array([[1, r12], [r12, 1]]) / t12
            M = M01 @ P @ M12
            t_total = 1 / M[0, 0]
            T[i] = (n3 / n1) * abs(t_total)**2

        return T / np.max(T) if np.max(T) != 0 else T

    def calculate_error(thickness, n1, n2, n3):
        T_model = transmittance(thickness, n1, n2, n3)
        interp = interp1d(wl_theo, T_model, kind='cubic', bounds_error=False, fill_value=np.nan)
        T_interp = interp(wl_exp)
        T_interp /= np.max(T_interp) if np.max(T_interp) != 0 else 1
        T_exp_norm = T_exp / np.max(T_exp)
        corr = np.corrcoef(T_interp, T_exp_norm)[0, 1]
        shape_error = 1 - abs(corr)
        amp_error = np.sqrt(np.mean((T_interp - T_exp_norm) ** 2))
        return 0.7 * shape_error + 0.3 * amp_error

    # Optimize thickness
    thicknesses = np.linspace(100e-9, 2000e-9, 3000)
    err_best, t_best = float('inf'), None
    for t in tqdm(thicknesses, desc="Optimizing", disable=True):
        err = calculate_error(t, **n_vals)
        if err < err_best:
            err_best = err
            t_best = t

    # GUI starts here
    def run_gui():
        gui = tk.Toplevel()
        gui.title("Thin Film TMM Viewer")
        
        def on_closing():
            gui.quit()       # Stop mainloop
            gui.destroy()    # Destroy GUI window
            plt.close('all') # Clear matplotlib memory
        
        gui.protocol("WM_DELETE_WINDOW", on_closing)

        fig, ax = plt.subplots(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=gui)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(gui)
        controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Thickness Entry
        ttk.Label(controls, text="Thickness (nm):").pack(pady=(10,0))
        t_var = tk.StringVar(value=f"{t_best * 1e9:.2f}")
        ttk.Entry(controls, textvariable=t_var, width=15).pack(pady=(0,10))

        # Update Function
        def update_plot(*args):
            val = t_var.get()
            if not val:
                return
            
            try:
                thickness = float(val) * 1e-9
                T_model = transmittance(thickness, **n_vals)
                err = calculate_error(thickness, **n_vals)
                ax.clear()
                ax.plot(wl_theo, T_model, label="Model")
                ax.scatter(wl_exp, T_exp, color='black', label="Experimental")
                ax.set_title(f"Thickness: {thickness*1e9:.1f} nm | Error: {err:.5f}\nn1={n_vals['n1']}, n2={n_vals['n2']}, n3={n_vals['n3']}")
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Normalized Transmittance")
                ax.set_ylim(0.8, 1.2)  # Fixed Y-axis from 0 to 1
                ax.grid(True)
                ax.legend()
                canvas.draw()
            except ValueError:
                pass # Ignore invalid characters while typing
            except Exception as e:
                messagebox.showerror("Plot Error", str(e))

        t_var.trace_add("write", lambda *_, v=t_var: update_plot())

        # Export Buttons
        def save_plot():
            f = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if f:
                fig.savefig(f)

        def save_csv():
            try:
                f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file", "*.csv")])
                if f:
                    T = transmittance(float(t_var.get()) * 1e-9, **n_vals)
                    pd.DataFrame({'Wavelength (nm)': wl_theo, 'T_model': T}).to_csv(f, index=False)
            except ValueError:
                messagebox.showerror("Export Error", "Invalid thickness value.")

        ttk.Button(controls, text="Save Plot", command=save_plot).pack(fill=tk.X, pady=5)
        ttk.Button(controls, text="Export CSV", command=save_csv).pack(fill=tk.X, pady=5)
        
        update_plot()
        gui.mainloop() # Code execution halts here until you close the plot window

    run_gui()
    
    # Terminal Prompt to Loop or Exit
    choice = input("\nDo you want to generate another plot with new values? (y/n): ").strip().lower()
    if choice != 'y':
        print("Terminating process...")
        break

# Clean Shutdown and Hard Kill
try:
    root.quit()
    root.destroy()
except tk.TclError:
    pass # Root might already be destroyed if closed via GUI
os._exit(0)
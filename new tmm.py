import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
import sv_ttk

df = None
current_wl_theo = None
current_n_ambient = None
current_n_layers = None
current_n_substrate = None
current_wl_exp = None
current_T_exp = None

layer_thickness_vars = []
layer_lock_vars = []

root = tk.Tk()
root.title("Multi-Layer Thin Film TMM Viewer")
root.geometry("1100x750")

root.option_add("*Font", "{Segoe UI} 10")
sv_ttk.set_theme("dark")

def on_closing():
    root.quit()
    root.destroy()
    plt.close('all')
    sys.exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)

plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

controls = ttk.Frame(root)
controls.pack(side=tk.RIGHT, fill=tk.Y, padx=15, pady=15)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#FAF9F6')
ax.set_facecolor('#FAF9F6')
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

file_label_var = tk.StringVar(value="No file loaded")
min_wl_var = tk.StringVar(value="400")
max_wl_var = tk.StringVar(value="800")
n_ambient_var = tk.StringVar(value="1.0")
n_layers_var = tk.StringVar(value="1.6, 1.4")
n_substrate_var = tk.StringVar(value="1.5")

def load_data():
    global df
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    
    if not file_path:
        return

    try:
        skip_lines = 0
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        break
                    except ValueError:
                        pass
                skip_lines += 1

        df_temp = pd.read_csv(file_path, skiprows=skip_lines, header=None, usecols=[0, 1])
        df_temp.columns = ["Wavelength", "Transmittance"]
        df_temp['Wavelength'] = pd.to_numeric(df_temp['Wavelength'], errors='coerce')
        df_temp['Transmittance'] = pd.to_numeric(df_temp['Transmittance'], errors='coerce')
        df_temp.dropna(inplace=True)
        
        df = df_temp
        file_name = os.path.basename(file_path)
        file_label_var.set(f"Loaded: {file_name}")
        
        ax.clear()
        ax.set_facecolor('#FAF9F6')
        canvas.draw()
        
    except Exception as e:
        messagebox.showerror("File Error", f"Error loading CSV: {e}")

def run_optimization():
    global df, current_wl_theo, current_n_ambient, current_n_layers, current_n_substrate, current_wl_exp, current_T_exp
    
    if df is None:
        messagebox.showerror("Data Error", "Please load a CSV file first.")
        return

    try:
        min_wl = float(min_wl_var.get())
        max_wl = float(max_wl_var.get())
        current_n_ambient = float(n_ambient_var.get())
        current_n_layers = [float(x.strip()) for x in n_layers_var.get().split(',')]
        current_n_substrate = float(n_substrate_var.get())
        
        current_thicknesses = []
        locks = []
        for tv, lv in zip(layer_thickness_vars, layer_lock_vars):
            current_thicknesses.append(float(tv.get()) * 1e-9)
            locks.append(lv.get())
            
    except ValueError:
        messagebox.showerror("Input Error", "Ensure valid numeric values in all fields.")
        return

    step = 0.5
    mask = (df['Wavelength'] >= min_wl) & (df['Wavelength'] <= max_wl)
    current_wl_exp = df[mask]['Wavelength'].values
    raw_T_exp = df[mask]['Transmittance'].values
    
    if len(current_wl_exp) == 0:
        messagebox.showerror("Data Error", "No data found in this wavelength range.")
        return

    current_T_exp = raw_T_exp / np.max(raw_T_exp)
    current_wl_theo = np.arange(min_wl, max_wl + step, step)
    
    free_indices = [i for i, locked in enumerate(locks) if not locked]
    locked_indices = [i for i, locked in enumerate(locks) if locked]

    if not free_indices:
        messagebox.showinfo("Optimization Locked", "All layers are locked. No variables to optimize.")
        return

    def optimize_thread():
        def transmittance(full_thicknesses_array):
            wl_m = current_wl_theo * 1e-9
            k0 = 2 * np.pi / wl_m
            n_list = [current_n_ambient] + current_n_layers + [current_n_substrate]
            num_layers = len(current_n_layers)

            T = np.zeros_like(current_wl_theo)
            for i in range(len(current_wl_theo)):
                M_total = np.eye(2, dtype=complex)
                for j in range(num_layers + 1):
                    na = n_list[j]
                    nb = n_list[j+1]
                    r = (na - nb) / (na + nb)
                    t_coeff = 2 * na / (na + nb)
                    
                    M_int = np.array([[1, r], [r, 1]]) / t_coeff
                    M_total = M_total @ M_int
                    
                    if j < num_layers:
                        delta = nb * k0[i] * full_thicknesses_array[j]
                        P = np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]])
                        M_total = M_total @ P

                t_total = 1 / M_total[0, 0]
                T[i] = (current_n_substrate / current_n_ambient) * abs(t_total)**2

            return T / np.max(T) if np.max(T) != 0 else T

        def calculate_error(free_thicknesses_array):
            full_t = np.zeros(len(current_n_layers))
            for i, idx in enumerate(free_indices):
                full_t[idx] = free_thicknesses_array[i]
            for idx in locked_indices:
                full_t[idx] = current_thicknesses[idx]

            T_model = transmittance(full_t)
            interp = interp1d(current_wl_theo, T_model, kind='cubic', bounds_error=False, fill_value=np.nan)
            T_interp = interp(current_wl_exp)
            T_interp /= np.max(T_interp) if np.max(T_interp) != 0 else 1
            corr = np.corrcoef(T_interp, current_T_exp)[0, 1]
            shape_error = 1 - abs(corr)
            amp_error = np.sqrt(np.mean((T_interp - current_T_exp) ** 2))
            return 0.7 * shape_error + 0.3 * amp_error

        bounds = [(10e-9, 2000e-9)] * len(free_indices)
        result = differential_evolution(calculate_error, bounds, seed=42)
        root.after(0, process_results, result.x, current_thicknesses, free_indices, locked_indices, result.fun)

    def process_results(free_best, current_t, f_idx, l_idx, err_best):
        progress_bar.stop()
        
        full_best = np.zeros(len(current_t))
        for i, idx in enumerate(f_idx):
            full_best[idx] = free_best[i]
        for idx in l_idx:
            full_best[idx] = current_t[idx]

        for i, tv in enumerate(layer_thickness_vars):
            tv.set(f"{full_best[i] * 1e9:.2f}")

        print(f"Optimization complete. Error: {err_best:.5f}")
        run_btn.config(state="normal", text="Run Optimization")

    print("Running optimization with parameter fixing...")
    run_btn.config(state="disabled", text="Optimizing...")
    progress_bar.start(15)
    thread = threading.Thread(target=optimize_thread, daemon=True)
    thread.start()

def update_plot(*args):
    if current_wl_theo is None:
        return
    
    try:
        local_n_layers = [float(x.strip()) for x in n_layers_var.get().split(',')]
        local_n_ambient = float(n_ambient_var.get())
        local_n_substrate = float(n_substrate_var.get())
        
        thicknesses_m = []
        for tv in layer_thickness_vars:
            val = tv.get().strip()
            if not val:
                return
            thicknesses_m.append(float(val) * 1e-9)
            
        if len(thicknesses_m) != len(local_n_layers):
            return
            
        wl_m = current_wl_theo * 1e-9
        k0 = 2 * np.pi / wl_m
        n_list = [local_n_ambient] + local_n_layers + [local_n_substrate]
        num_layers = len(local_n_layers)

        T_model = np.zeros_like(current_wl_theo)
        for i in range(len(current_wl_theo)):
            M_total = np.eye(2, dtype=complex)
            for j in range(num_layers + 1):
                na = n_list[j]
                nb = n_list[j+1]
                r = (na - nb) / (na + nb)
                t_coeff = 2 * na / (na + nb)
                
                M_int = np.array([[1, r], [r, 1]]) / t_coeff
                M_total = M_total @ M_int
                
                if j < num_layers:
                    delta = nb * k0[i] * thicknesses_m[j]
                    P = np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]])
                    M_total = M_total @ P

            t_total = 1 / M_total[0, 0]
            T_model[i] = (local_n_substrate / local_n_ambient) * abs(t_total)**2

        T_model = T_model / np.max(T_model) if np.max(T_model) != 0 else T_model

        interp = interp1d(current_wl_theo, T_model, kind='cubic', bounds_error=False, fill_value=np.nan)
        T_interp = interp(current_wl_exp)
        T_interp /= np.max(T_interp) if np.max(T_interp) != 0 else 1
        corr = np.corrcoef(T_interp, current_T_exp)[0, 1]
        shape_error = 1 - abs(corr)
        amp_error = np.sqrt(np.mean((T_interp - current_T_exp) ** 2))
        live_err = 0.7 * shape_error + 0.3 * amp_error

        ax.clear()
        ax.set_facecolor('#FAF9F6')
        
        ax.plot(current_wl_theo, T_model, label="Model", linewidth=1.5, color='#1f77b4')
        ax.scatter(current_wl_exp, current_T_exp, color='#ff7f0e', label="Experimental", s=10)
        
        t_str = ", ".join([f"{t*1e9:.1f}" for t in thicknesses_m])
        ax.set_title(f"Thicknesses: {t_str} nm | Error: {live_err:.5f}\nLayers n: {local_n_layers}", pad=15, color='black')
        ax.set_xlabel("Wavelength (nm)", color='black')
        ax.set_ylabel("Normalized Transmittance", color='black')
        ax.set_ylim(0.9, 1.05)
        
        ax.grid(True, color='black', alpha=0.3)
        ax.tick_params(axis='both', colors='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            
        ax.legend(facecolor='#FAF9F6', edgecolor='black', labelcolor='black')
        
        fig.tight_layout()
        canvas.draw()
    except ValueError:
        pass

def save_plot():
    f = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
    if f:
        fig.savefig(f)

def build_layer_ui(*args):
    for widget in dynamic_layers_frame.winfo_children():
        widget.destroy()
    layer_thickness_vars.clear()
    layer_lock_vars.clear()

    try:
        n_layers = [float(x.strip()) for x in n_layers_var.get().split(',')]
    except ValueError:
        return

    for i, n in enumerate(n_layers):
        row = ttk.Frame(dynamic_layers_frame)
        row.pack(fill="x", pady=2)
        
        ttk.Label(row, text=f"Layer {i+1} (n={n}):", width=15).pack(side="left")
        
        t_val = tk.StringVar(value="100.0")
        t_val.trace_add("write", update_plot)
        ttk.Entry(row, textvariable=t_val, width=10).pack(side="left", padx=5)
        layer_thickness_vars.append(t_val)

        l_val = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Lock", variable=l_val).pack(side="left")
        layer_lock_vars.append(l_val)
        
    update_plot()

file_frame = ttk.LabelFrame(controls, text="File Operations")
file_frame.pack(fill="x", padx=10, pady=5)
ttk.Button(file_frame, text="Load CSV Data", command=load_data).pack(fill="x", padx=10, pady=(10, 5))
ttk.Label(file_frame, textvariable=file_label_var, wraplength=180).pack(anchor="w", padx=10, pady=(0, 10))

wl_frame = ttk.LabelFrame(controls, text="Wavelength Range (nm)")
wl_frame.pack(fill="x", padx=10, pady=5)
ttk.Label(wl_frame, text="Min Wavelength:").pack(anchor="w", padx=10, pady=(10, 2))
ttk.Entry(wl_frame, textvariable=min_wl_var).pack(fill="x", padx=10, pady=(0, 5))
ttk.Label(wl_frame, text="Max Wavelength:").pack(anchor="w", padx=10, pady=(5, 2))
ttk.Entry(wl_frame, textvariable=max_wl_var).pack(fill="x", padx=10, pady=(0, 10))

index_frame = ttk.LabelFrame(controls, text="Refractive Indices")
index_frame.pack(fill="x", padx=10, pady=5)
ttk.Label(index_frame, text="Ambient (n):").pack(anchor="w", padx=10, pady=(10, 2))
ttk.Entry(index_frame, textvariable=n_ambient_var).pack(fill="x", padx=10, pady=(0, 5))
ttk.Label(index_frame, text="Layers (n) comma-separated:").pack(anchor="w", padx=10, pady=(5, 2))
ttk.Entry(index_frame, textvariable=n_layers_var).pack(fill="x", padx=10, pady=(0, 5))
ttk.Label(index_frame, text="Substrate (n):").pack(anchor="w", padx=10, pady=(5, 2))
ttk.Entry(index_frame, textvariable=n_substrate_var).pack(fill="x", padx=10, pady=(0, 10))

opt_frame = ttk.LabelFrame(controls, text="Optimization & Layer Setup")
opt_frame.pack(fill="x", padx=10, pady=5)

dynamic_layers_frame = ttk.Frame(opt_frame)
dynamic_layers_frame.pack(fill="x", padx=10, pady=(10, 5))

run_btn = ttk.Button(opt_frame, text="Run Optimization", command=run_optimization)
run_btn.pack(fill="x", padx=10, pady=(10, 5))

progress_bar = ttk.Progressbar(opt_frame, mode='indeterminate')
progress_bar.pack(fill="x", padx=10, pady=(0, 10))

ttk.Button(controls, text="Save Plot", command=save_plot).pack(fill="x", padx=10, pady=10)

n_layers_var.trace_add("write", build_layer_ui)
build_layer_ui() 

root.mainloop()

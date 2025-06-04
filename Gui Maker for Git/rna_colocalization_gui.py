import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import trimesh
from my_functions import (
    extract_rna_spots,
    rotate_rna_spots,
    compute_original_colocalization,
    compute_rotated_colocalization,
    compute_random_colocalization,
    compute_kde_colocalization,
)

class RNAColocalizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RNA Colocalization Analysis")

        # File upload section
        tk.Label(root, text="Upload Mesh File (.obj):").grid(row=0, column=0, sticky="w")
        self.mesh_file_entry = tk.Entry(root, width=50)
        self.mesh_file_entry.grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_mesh_file).grid(row=0, column=2)

        tk.Label(root, text="Upload RNA 1 File (.txt):").grid(row=1, column=0, sticky="w")
        self.rna1_file_entry = tk.Entry(root, width=50)
        self.rna1_file_entry.grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_rna1_file).grid(row=1, column=2)

        tk.Label(root, text="Upload RNA 2 File (.txt):").grid(row=2, column=0, sticky="w")
        self.rna2_file_entry = tk.Entry(root, width=50)
        self.rna2_file_entry.grid(row=2, column=1)
        tk.Button(root, text="Browse", command=self.browse_rna2_file).grid(row=2, column=2)

        tk.Label(root, text="Upload RNA 3 File (.txt):").grid(row=3, column=0, sticky="w")
        self.rna3_file_entry = tk.Entry(root, width=50)
        self.rna3_file_entry.grid(row=3, column=1)
        tk.Button(root, text="Browse", command=self.browse_rna3_file).grid(row=3, column=2)

        # Parameters section
        tk.Label(root, text="Number of Trials:").grid(row=4, column=0, sticky="w")
        self.num_trials_entry = tk.Entry(root, width=10)
        self.num_trials_entry.grid(row=4, column=1, sticky="w")
        self.num_trials_entry.insert(0, "25")

        tk.Label(root, text="Colocalization Threshold:").grid(row=5, column=0, sticky="w")
        self.threshold_entry = tk.Entry(root, width=10)
        self.threshold_entry.grid(row=5, column=1, sticky="w")
        self.threshold_entry.insert(0, "1.0")

        # Run button
        tk.Button(root, text="Run Analysis", command=self.run_analysis).grid(row=6, column=0, columnspan=3)

        # Results section
        self.results_text = tk.Text(root, height=15, width=80)
        self.results_text.grid(row=7, column=0, columnspan=3)

    def browse_mesh_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("OBJ Files", "*.obj")])
        self.mesh_file_entry.delete(0, tk.END)
        self.mesh_file_entry.insert(0, file_path)

    def browse_rna1_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TXT Files", "*.txt")])
        self.rna1_file_entry.delete(0, tk.END)
        self.rna1_file_entry.insert(0, file_path)

    def browse_rna2_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TXT Files", "*.txt")])
        self.rna2_file_entry.delete(0, tk.END)
        self.rna2_file_entry.insert(0, file_path)

    def browse_rna3_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TXT Files", "*.txt")])
        self.rna3_file_entry.delete(0, tk.END)
        self.rna3_file_entry.insert(0, file_path)

    def run_analysis(self):
        try:
            # Get inputs
            mesh_file = self.mesh_file_entry.get()
            rna1_file = self.rna1_file_entry.get()
            rna2_file = self.rna2_file_entry.get()
            rna3_file = self.rna3_file_entry.get()
            num_trials = int(self.num_trials_entry.get())
            threshold = float(self.threshold_entry.get())

            # Validate inputs
            if not mesh_file or not rna1_file or not rna2_file:
                messagebox.showerror("Error", "Please upload at least a mesh file and two RNA files.")
                return

            # Load mesh
            mesh = trimesh.load_mesh(mesh_file, file_type="obj")

            # Load RNA spots
            rna_spots_list = [extract_rna_spots(rna1_file), extract_rna_spots(rna2_file)]
            if rna3_file:
                rna_spots_list.append(extract_rna_spots(rna3_file))

            # Run analyses
            results = []

            # Original colocalization
            avg_original = compute_original_colocalization(rna_spots_list, num_trials, threshold)
            results.append(f"Original Colocalization: {avg_original:.2f}%")

            # Rotated colocalization
            avg_rotated = compute_rotated_colocalization(rna_spots_list, mesh, num_trials, threshold)
            results.append(f"Rotated Colocalization: {avg_rotated:.2f}%")

            # Randomized colocalization
            avg_random = compute_random_colocalization(rna_spots_list, mesh, num_trials, threshold)
            results.append(f"Randomized Colocalization: {avg_random:.2f}%")

            # KDE-based colocalization
            avg_kde = compute_kde_colocalization(rna_spots_list, mesh, num_trials, threshold)
            results.append(f"KDE-Based Colocalization: {avg_kde:.2f}%")

            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "\n".join(results))

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = RNAColocalizationApp(root)
    root.mainloop()
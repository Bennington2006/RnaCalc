import argparse
import pandas as pd
import numpy as np
import trimesh
from my_functions6 import (
    extract_rna_spots,
    compute_original_colocalization,
    compute_rotated_colocalization,
    compute_random_colocalization,
    compute_kde_colocalization
)

def main():
    parser = argparse.ArgumentParser(
        description="RNA Colocalization Analysis CLI"
    )
    parser.add_argument('--mesh', required=True, help='Path to mesh file (.obj)')
    parser.add_argument('--rna1', required=True, help='Path to RNA 1 spots file (.txt)')
    parser.add_argument('--rna2', required=True, help='Path to RNA 2 spots file (.txt)')
    parser.add_argument('--rna3', help='Path to RNA 3 spots file (.txt, optional)')
    parser.add_argument('--trials', type=int, default=25, help='Number of trials for randomization/KDE (default: 25)')
    parser.add_argument('--threshold', type=float, default=1.0, help='Colocalization distance threshold (default: 1.0)')
    parser.add_argument('--output', help='Optional output file to save results (CSV)')

    args = parser.parse_args()

    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load_mesh(args.mesh, file_type="obj")

    # Load RNA spots
    print("Loading RNA spot files...")
    rna_spots_list = [
        extract_rna_spots(args.rna1),
        extract_rna_spots(args.rna2)
    ]
    if args.rna3:
        rna_spots_list.append(extract_rna_spots(args.rna3))

    # Run analyses
    print("Running original colocalization analysis...")
    avg_original = compute_original_colocalization(rna_spots_list, args.trials, args.threshold)
    print(f"Original Colocalization: {avg_original:.2f}%")

    print("Running rotated colocalization analysis...")
    avg_rotated = compute_rotated_colocalization(rna_spots_list, mesh, args.trials, args.threshold)
    print(f"Rotated Colocalization: {avg_rotated:.2f}%")

    print("Running randomized colocalization analysis...")
    avg_random = compute_random_colocalization(rna_spots_list, mesh, args.trials, args.threshold)
    print(f"Randomized Colocalization: {avg_random:.2f}%")

    print("Running KDE-based colocalization analysis...")
    avg_kde = compute_kde_colocalization(rna_spots_list, mesh, args.trials, args.threshold)
    print(f"KDE-Based Colocalization: {avg_kde:.2f}%")

    # Optionally save results
    if args.output:
        results = pd.DataFrame({
            "Type": ["Original", "Rotated", "Randomized", "KDE-Based"],
            "Average Colocalization (%)": [
                avg_original, avg_rotated, avg_random, avg_kde
            ]
        })
        results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
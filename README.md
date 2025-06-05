# RNA Colocalization Analysis App

A free and open-source tool for object-based colocalization analysis of RNA spots in 3D meshes. This app provides a simple graphical user interface (GUI) and a command-line interface (CLI), making it accessible for researchers at all levels. It supports robust segmentation, denoising, and in-depth analysis of colocalization between objects in up to three channels.

---

## Introduction

RNA Colocalization Analysis App enables object-based colocalization analysis with quantification, implemented in Python. The tool is available as both a user-friendly GUI (via Streamlit) and a CLI for flexible batch processing. It is suitable for a wide range of biological datasets and provides automated, reproducible, and robust quantification of co-localized particles. The app outputs quantitative results for inspection.

---

## Features

- Upload mesh and RNA spot files (up to three channels)
- Compute colocalization metrics: original, rotated, randomized, and KDE-based
- Visualize results in a user-friendly interface
- Export results for further analysis

---

## Local Installation

### Prerequisites

- Python >= 3.8
- At least 8GB RAM recommended
- Supported OS: Windows 10/11
- Internet connection required for installation

#### Required Python Packages

- numpy (>=1.20.0)
- pandas
- scipy
- matplotlib
- trimesh
- pyqtgraph
- PyQt5
- numba
- natsort
- readlif
- read_lif
- cellpose
- scikit-image
- trackpy
- seaborn
- streamlit
- rtree

Install all dependencies with:

```
pip install -r requirements.txt
```

---

## Running the App

### Web Version

1. Clone or download this repository:
    ```
    git clone https://github.com/Bennington2006/RnaCalc.git
    ```

2. Install dependencies (see above).

3. Start the Streamlit app:
    ```
    streamlit run app6.py
    ```

4. Follow the on-screen instructions to upload your mesh and RNA spot files and run the analysis.

### CLI (Work in Progress)

You can also run the analysis from the command line (example command):

```
python app6.py --mesh path_to_mesh.obj --rna1 gene1.txt --rna2 gene2.txt --rna3 gene3.txt --trials 1 --threshold 1.0
```
(FILES MUST BE IN CLI FOLDER OR USE CORRECT PATHING TO FUNCTION PROPERLY)
---

## Parameter Settings

| Option         | Description                                 |
|----------------|---------------------------------------------|
| --mesh         | Input mesh file (.obj)                      |
| --rna1         | RNA spots file 1 (.txt)                     |
| --rna2         | RNA spots file 2 (.txt)                     |
| --rna3         | RNA spots file 3 (.txt, optional)           |
| --trials       | Number of trials for randomization/KDE      |
| --threshold    | Colocalization distance threshold           |

Default parameters are used if not specified.

---

## Output

- Quantitative colocalization results (original, rotated, randomized, KDE-based)
- Visualizations of mesh and RNA spots
- Exportable tables for further analysis

---

## GUI (currently unavailable)

---


## Common Issues

- Mesh file must be in OBJ format
- RNA spot files must be tab-delimited text with at least 3 columns (X, Y, Z)
- Large images or meshes may require more memory
- For GUI, use Windows for best compatibility


---

## License

This project is licensed under the GNU GPL v3 - see the [LICENSE.txt](LICENSE.txt) file for details.

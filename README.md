<p align="left">
  <img src="https://raw.githubusercontent.com/amesch441/iSCALE/main/assets/iSCALE_logo2.png" width="200"/>
</p>


# Scaling up spatial transcriptomics for large-sized tissues with **iSCALE**

**iSCALE** (*Inferring Spatially resolved Cellular Architectures for Large-sized tissue Environments*)  
is a novel framework designed to integrate multiple daughter captures and utilize H&E information from large tissue samples, enabling prediction of gene expression with near single-cell resolution across whole-slide tissues.

<p align="center">
  <img src="https://raw.githubusercontent.com/amesch441/iSCALE/main/assets/iSCALE_workflow2.png" width="1200"/>
</p>
<p align="center">
  <strong>Figure:</strong> <em>iSCALE workflow</em>
</p>


---

## 🔧 Installation & Setup

Clone the repository:
```bash
git clone https://github.com/amesch441/iSCALE.git
cd iSCALE-main
```

### Option A (recommended): Conda
```bash
conda env create -f environment.yml
conda activate iSCALE_env
cd iSCALE
```

### Option B: Pip
```bash
python -m venv iSCALE_env
source iSCALE_env/bin/activate   # Linux/Mac
# or: .\iSCALE_env\Scripts\activate   # Windows
pip install -r requirements.txt
cd iSCALE
```

> ⚡ **GPU usage is strongly recommended** for speed and scalability. CPU mode is supported but slower.

---

## 📦 Download Demo Data & Checkpoints

Download from [Box link](https://upenn.box.com/s/cburekr425ibu276wyxki09q35z2o3x0).

- Place the model checkpoints:
  - `vit4k_xs_dino.pth`
  - `vit256_small_dino.pth`  
  into:
  ```
  iSCALE-main/iSCALE/checkpoints/
  ```

- Place the demo dataset into:
  ```
  iSCALE-main/iSCALE/data/demo/
  ```

---

## ▶️ Running iSCALE

To run the demo:
```bash
sbatch run_iSCALE.sh
```
with `prefix="Data/demo/gastricTumor/"`.  
Ground truth for this demo gastric tumor tissue can be found in the `cnts-truth-agg` folder.

- Use `_run_iSCALE_sbatch.sh` if your system uses **SLURM**.  
- Use `_run_iSCALE_bsub.sh` if your system uses **LSF**.  
  (These scripts are identical except for scheduler setup.)

---

## 📂 Repository Structure

```
iSCALE-main/
│
├── environment.yml         # conda environment specification
├── requirements.txt        # pip requirements
│
├── iSCALE/
│   ├── checkpoints/        # pretrained models (place downloaded .pth files here)
│   ├── data/               # input data (demo goes here)
│   ├── Alignment_scripts/  # tools for semi-automatic alignment
│   ├── *.py                # main Python scripts
│   ├── *.sh                # run scripts (SLURM/LSF)
│   └── ...
```

---

## 📂 Input Data & Formats

Each project has the following structure:

```
iSCALE-main/iSCALE/Data/<project_name>/
│
├── DaughterCaptures/
│   ├── UnallignedToMother/        # raw ST data (Visium, Visium HD, Xenium, CosMx)
│   │   ├── D1/
│   │   │   ├── cnts.tsv           # count matrix (genes × spots)
│   │   │   ├── locs.tsv           # coordinates (spot_id, x, y)
│   │   │   └── he.*               # H&E image (see formats below)
│   │   ├── D2/
│   │   └── ...
│   │
│   └── AllignedToMother/          # aligned data (produced after registration)
│       ├── D1/
│       │   ├── cnts.tsv
│       │   └── locs.tsv
│       ├── D2/
│       └── ...
│
└── MotherImage/
    ├── he-raw.*                   # raw H&E (before scaling)
    ├── he-scaled.*                # scaled H&E (after resizing)
    ├── he.tiff                    # final processed H&E with padding
    ├── radius-raw.txt             # raw spot radius in µm
    ├── radius.txt                 # scaled radius (pixels, auto-generated if missing)
    └── markers.csv (optional)     # marker genes for auto-annotation
```

### Notes
- **Supported H&E formats**:  
  `.tiff`, `.tif`, `.svs`, `.ome.tif`, `.ome.tiff`, `.jpg`, `.png`, `.ndpi`, `.scn`, `.mrxs`  
- **locs.tsv**: must contain  
  ```
  spot_id   x   y
  ```
- **cnts.tsv**: genes × spots matrix (tab-delimited).  
- **markers.csv** (optional):  
  ```
  gene,label
  MKI67,Tumor
  KRT20,Mucosa
  ...
  ```

---

## ⚙️ Input Parameters

Parameters are set in the run scripts (`_run_iSCALE_sbatch.sh` or `_run_iSCALE_bsub.sh`).

| Parameter                | Description                                                                 | Default Example   |
|--------------------------|-----------------------------------------------------------------------------|-------------------|
| `prefix_general`         | Project directory path (must contain `DaughterCaptures` and `MotherImage`)  | `Data/demo/`      |
| `daughterCapture_folders`| List of daughter capture folders                                            | `("D1" "D2" "D3")`|
| `device`                 | Compute device: `"cuda"` (GPU) or `"cpu"`                                   | `"cuda"`          |
| `pixel_size_raw`         | Pixel size (µm/pixel) of raw H&E                                            | `0.252`           |
| `pixel_size`             | Desired pixel size after rescaling                                          | `0.5`             |
| `n_genes`                | Number of most variable genes to impute                                     | `100`             |
| `n_clusters`             | Number of clusters for downstream analysis                                  | `20`              |
| `dist_ST`                | Smoothing parameter across ST captures (integration sharpness)              | `100`             |

**Notes**  
- `prefix_general` is the main project folder.  
- `dist_ST=100` works well in most cases, but check QC plots in `iSCALE_output/spot_level_st_plots/` to tune if needed.  

---

## 📤 Output

All results are saved to `iSCALE_output/`:

- **spot_level_st_plots/**  
  QC plots to confirm correct alignment of daughter captures onto mother image.  
- **super_res_gene_expression/**  
  Imputed super-resolution expression (pickle files).  
  - `refined/` subfolder removes predictions outside nuclei regions.  
- **super_res_ST_plots/**  
  Visualizations of super-resolution gene expression.  
  - includes `refined/`.  
- **clusters-gene_#/**  
  Clustering results using imputed gene expression.  
- **annotation/**  
  Cell-type/region annotations if markers.csv was provided.

---

## 📖 Citation

If you use iSCALE, please cite:

> Schroeder AR, et al. *Scaling up spatial transcriptomics for large-sized tissues: uncovering cellular-level tissue architecture beyond conventional platforms.*  
> **Nature Methods** (2025).  
> [https://www.nature.com/articles/s41592-025-02770-8](https://www.nature.com/articles/s41592-025-02770-8)

---

## 📜 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file included in this repository.

# Linear Algebra for Machine Learning

A six-week, first-principles course that rebuilds the linear algebra underpinning
modern machine learning **from scratch in pure NumPy** — then connects every
construct back to the model that depends on it. No black-box `np.linalg` calls
until you have implemented the algorithm yourself and understand exactly what it
returns and why.

Each lecture is a self-contained Jupyter notebook: theoretical derivation,
a from-scratch implementation, visualizations of the geometry, a verification
against the reference library, and exercises with worked solutions.

---

## Philosophy

Most ML practitioners treat linear algebra as an API. This course treats it as a
set of geometric ideas you can *see* and *build*. The through-line is always the
same loop:

> **derive the math → implement it in NumPy → visualize the geometry → verify against `scipy`/`numpy` → apply it to a real ML problem.**

By the end you will have hand-written: Gaussian elimination, Gram–Schmidt, the
power method, QR iteration, a full SVD, PCA, ridge regression via the normal
equations, and a linear-algebra-only neural network layer — and you will know
exactly which decomposition each one secretly relies on.

---

## Prerequisites

- Comfortable Python and basic NumPy (arrays, slicing, broadcasting)
- High-school algebra; no prior linear algebra assumed
- Curiosity about *why* the methods work, not just how to call them

---

## Curriculum

### Week 1 — Vectors, Spaces & Geometry
The objects everything else is built on. Vectors as arrows and as data points,
norms as notions of distance, the dot product as similarity, projections, and
linear (in)dependence. Why a dataset is a cloud of vectors and a feature is a
coordinate.
- `01_vectors_and_vector_spaces.ipynb`
- `02_norms_dot_products_projections.ipynb`

### Week 2 — Matrices as Linear Maps
Stop seeing matrices as grids of numbers. A matrix is a function that moves
space. Matrix–vector products, composition, the four fundamental subspaces,
rank, and the geometric meaning of determinant and trace.
- `03_matrices_as_transformations.ipynb`
- `04_rank_fundamental_subspaces.ipynb`

### Week 3 — Solving Linear Systems
Gaussian elimination and LU decomposition implemented by hand. Conditioning and
why naive solutions blow up. The normal equations and least squares — the engine
behind linear regression.
- `05_gaussian_elimination_lu.ipynb`
- `06_least_squares_normal_equations.ipynb`

### Week 4 — Orthogonality & QR
Orthonormal bases, the Gram–Schmidt process built from scratch, QR decomposition,
and orthogonal projections. Why orthogonality makes everything numerically
stable, and how QR powers stable least squares.
- `07_orthogonality_gram_schmidt.ipynb`
- `08_qr_decomposition.ipynb`

### Week 5 — Eigenvalues & Eigenvectors
The spectral heart of linear algebra. Eigendecomposition, the power method and
QR iteration coded by hand, diagonalization, and spectral theory for symmetric
matrices. PageRank and Markov chains as eigenproblems.
- `09_eigenvalues_eigenvectors.ipynb`
- `10_power_method_qr_iteration.ipynb`

### Week 6 — SVD & Applications in ML
The single most important decomposition in machine learning. Build the SVD from
the eigendecomposition, then deploy it: PCA from first principles, low-rank
approximation and image compression, ridge regression through the SVD lens, and
a closing capstone that ties dimensionality reduction to a real model.
- `11_singular_value_decomposition.ipynb`
- `12_pca_lowrank_capstone.ipynb`

---

## Repository structure

```
linear_algebra_for_ml/
├── README.md
├── requirements.txt
├── notebooks/          # the 12 lecture notebooks
├── utils/              # shared plotting & helper functions
│   └── linalg_viz.py
├── data/               # small datasets used in applied sections
└── assets/             # exported figures
```

## Getting started

```bash
git clone https://github.com/HAYDARKILIC/linear_algebra_for_ml.git
cd linear_algebra_for_ml
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/01_vectors_and_vector_spaces.ipynb` and work top to bottom.
Every notebook runs standalone.

---

## How to use this course

1. **Read the derivation** — the math is motivated, not dumped.
2. **Type the implementation yourself** before reading the provided one.
3. **Run the visualizations** — rotate them, change the inputs, break them.
4. **Do the exercises** — solutions are collapsed at the bottom of each notebook.
5. **Connect it forward** — each notebook ends with "Where this shows up in ML."

---

## License

MIT — free to use for teaching and self-study.

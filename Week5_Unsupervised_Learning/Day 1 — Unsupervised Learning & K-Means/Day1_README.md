## Notebook Overview

This notebook  an introduction to **Unsupervised Machine Learning**, with a primary focus on **K-Means Clustering**.

It introduces the difference between supervised and unsupervised learning, explains the main concepts behind clustering, covers the mathematical logic of K-Means, and demonstrates how to choose the appropriate number of clusters ($k$) using the **Elbow Method** and **Silhouette Analysis**.

The notebook concludes with coding exercises **Iris Dataset**, where K-Means is used to discover natural groupings in the data without using the target labels.

---

## Notebook Contents

| Section | Description |
| --- | --- |
| **Introduction** | Introduces unsupervised learning and explains how models discover patterns without labeled data. |
| **Section 1 — Unsupervised Learning** | Explains the definition, general workflow, and differences between supervised and unsupervised learning. |
| **Section 2 — Clustering Algorithms** | Introduces clustering and common algorithms, including K-Means, Hierarchical Clustering, DBSCAN, Mean-Shift, and Spectral Clustering. |
| **Section 3 — K-Means Clustering** | Explains how K-Means works, including Euclidean Distance, centroid updates, `scikit-learn` syntax, feature scaling, and methods for selecting $k$. |
| **Section 4 — Hands-on Lab** | Applies `StandardScaler` and `KMeans` to the **Iris Dataset** to discover natural clusters. |
| **References** | Contains external documentation and learning resources. |

---

## Dataset Used

**Iris Dataset** — `sklearn.datasets.load_iris`

- **Instances:** 150 flower samples
- **Features:** 4 numerical measurements:
  - `sepal length (cm)`
  - `sepal width (cm)`
  - `petal length (cm)`
  - `petal width (cm)`
- **Target Classes:** *Iris-Setosa*, *Iris-Versicolour*, and *Iris-Virginica*
- **Note:** The target labels are **not used during clustering**, since K-Means is an unsupervised learning algorithm.

---

## How to Use and Run This Notebook

#### Prerequisites

Ensure you have Python 3 installed along with the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter

```

#### Running the Notebook

1. **Clone/Download** the repository containing the `.ipynb` file and relevant image assets (`unsupervised_learning.PNG`, `clustering.PNG`, etc.).


2. **Launch Jupyter Notebook** or JupyterLab in your terminal:
```bash
jupyter notebook

```

3. **Open the Notebook** from the Jupyter dashboard.
4. **Execute the Cells**: Run the cells sequentially from top to bottom (`Shift + Enter`).


* The code cells execute data loading, feature scaling with `StandardScaler`, K-Means clustering, and plotting of the Elbow curve.
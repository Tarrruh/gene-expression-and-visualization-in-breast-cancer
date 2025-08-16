import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def fdr_bh(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    cumsum_min = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        val = pvals[order[i]] * n / rank
        cumsum_min = min(cumsum_min, val)
        ranked[order[i]] = cumsum_min
    return np.clip(ranked, 0, 1)


def make_synthetic_data(n_genes=1500, n_cancer=40, n_normal=20, seed=42):
    rng = np.random.default_rng(seed)
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]
    samples = [f"CANCER_{i:02d}" for i in range(n_cancer)] + [f"NORMAL_{i:02d}" for i in range(n_normal)]

    base = rng.normal(loc=6.0, scale=1.0, size=(n_genes, n_cancer + n_normal))

    de_idx_up = rng.choice(n_genes, size=60, replace=False)
    de_idx_down = rng.choice(list(set(range(n_genes)) - set(de_idx_up)), size=60, replace=False)

    base[de_idx_up, :n_cancer] += rng.normal(1.2, 0.2, size=(len(de_idx_up), n_cancer))
    base[de_idx_down, :n_cancer] -= rng.normal(1.2, 0.2, size=(len(de_idx_down), n_cancer))
    expr = pd.DataFrame(base, index=genes, columns=samples)

    meta = pd.DataFrame({
        "SampleID": samples,
        "Type": ["Cancer"] * n_cancer + ["Normal"] * n_normal
    })
    return expr, meta
def classify_type(x):
    text = str(x).lower()
    if "normal" in text:
        return "Normal"
    elif "tumor" in text or "cancer" in text or "carcinoma" in text:
        return "Cancer"
    else:
        return "Cancer"

def load_data(expr_path=None, meta_path=None, log2_if_needed=True):
    if expr_path and os.path.exists(expr_path) and (meta_path is None):
        expression = pd.read_csv(expr_path, sep="\t", comment="!", index_col=0)
        metadata = []
        with open(expr_path) as f:
            for line in f:
                if line.startswith("!Sample_characteristics_ch1"):
                    parts = line.strip().split("\t")
                    metadata.append(parts)

        meta_dict = {}
        for row in metadata:
            key = row[0].replace("!Sample_characteristics_ch1 = ", "")
            values = row[1:]
            meta_dict[key] = values

        metadata = pd.DataFrame(meta_dict)
        metadata.insert(0, "SampleID", expression.columns)


        metadata["Type"] = metadata.drop(columns=["SampleID"]).apply(
            lambda row: classify_type(" ".join(row.values.astype(str))),
            axis=1
        )

    elif expr_path and meta_path and os.path.exists(expr_path) and os.path.exists(meta_path):
        expression = pd.read_csv(expr_path, index_col=0)
        metadata = pd.read_csv(meta_path)

    else:
        print("[INFO] No valid files provided; generating a synthetic dataset so the pipeline can run.")
        expression, metadata = make_synthetic_data()

    return expression, metadata

def plot_pca(expression, metadata, out_path):
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(expression.T)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)

    label_map = metadata.set_index("SampleID")["Type"].to_dict()
    labels = np.array([label_map[s] for s in expression.columns])

    color_map = {"Cancer": "red", "Normal": "blue"}
    colors = [color_map[l] for l in labels]

    plt.figure(figsize=(7, 6), dpi=130)
    for l in np.unique(labels):
        idx = labels == l
        plt.scatter(pcs[idx, 0], pcs[idx, 1], c=np.array(colors)[idx], label=l, alpha=0.7)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("PCA: Breast Cancer vs Normal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def differential_expression(expression, metadata):

    cancer_ids = metadata.loc[metadata["Type"] == "Cancer", "SampleID"]
    normal_ids = metadata.loc[metadata["Type"] == "Normal", "SampleID"]

    mean_cancer = expression[cancer_ids].mean(axis=1)
    mean_normal = expression[normal_ids].mean(axis=1)
    log_fc = (mean_cancer - mean_normal)

    pvals = []
    tstats = []
    for g in expression.index:
        x = expression.loc[g, cancer_ids].values
        y = expression.loc[g, normal_ids].values
        t, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        tstats.append(t if np.isfinite(t) else 0.0)
        pvals.append(p if np.isfinite(p) else 1.0)

    qvals = fdr_bh(pvals)
    res = pd.DataFrame({
        "Gene": expression.index,
        "logFC": log_fc.values,
        "t_stat": tstats,
        "p_value": pvals,
        "q_value": qvals
    }).set_index("Gene").sort_values(["q_value", "p_value"])
    return res

def main():
    parser = argparse.ArgumentParser(description="Breast Cancer Gene Expression Analysis Pipeline")
    parser.add_argument("--expression", type=str, default=None, help="Path to expression CSV")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata CSV")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory for outputs")
    parser.add_argument("--topn_heatmap", type=int, default=50, help="Number of top genes for heatmap")
    parser.add_argument("--logfc_thresh", type=float, default=1.0, help="Volcano vertical threshold")
    parser.add_argument("--q_thresh", type=float, default=0.05, help="FDR threshold line on volcano")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    expression, metadata = load_data(args.expression, args.metadata)

    pca_path = os.path.join(args.outdir, "pca_scatter.png")
    plot_pca(expression, metadata, pca_path)
    print(f"[DONE] PCA plot -> {pca_path}")

    de = differential_expression(expression, metadata)
    de_csv = os.path.join(args.outdir, "de_results.csv")
    de.to_csv(de_csv)
    print(f"[DONE] DE results -> {de_csv}")


    sig = de[de["q_value"] <= 0.05].copy()
    up = sig.sort_values("logFC", ascending=False).head(10)
    down = sig.sort_values("logFC", ascending=True).head(10)
    print("\nTop upregulated (q<=0.05):")
    print(up[["logFC", "q_value"]])
    print("\nTop downregulated (q<=0.05):")
    print(down[["logFC", "q_value"]])

    print("\n[COMPLETE] All outputs saved in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
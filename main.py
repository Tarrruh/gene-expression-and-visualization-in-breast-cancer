import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def classify_type(x):
    """Classification function with keywords"""
    text = str(x).lower()
    
    # Normal/control keywords
    normal_keywords = ["normal", "control", "healthy", "adjacent", "non-tumor", "non-cancer", 
                      "non-malignant", "benign", "wildtype", "wt", "ctrl"]
    
    # Cancer/tumor keywords  
    cancer_keywords = ["tumor", "cancer", "carcinoma", "malignant", "adenocarcinoma", 
                      "ductal", "invasive", "metastatic", "primary", "tumor tissue",
                      "cancerous", "neoplasm", "oncology", "tumour"]

    if any(keyword in text for keyword in normal_keywords):
        return "Normal"
    
    elif any(keyword in text for keyword in cancer_keywords):
        return "Cancer"
    
    else:
        return "Unclassified"


def load_data(expr_path=None, meta_path=None, log2_if_needed=True):
    """
    Load expression and metadata, with improved sample classification
    """
    
    if expr_path and os.path.exists(expr_path) and (meta_path is None):
        # Case 1: Single GEO file with embedded metadata
        expression = pd.read_csv(expr_path, sep="\t", comment="!", index_col=0)
        
        metadata_info = {}
        sample_names = list(expression.columns)
        
        with open(expr_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith("!Sample_") and "=" in line:
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip().replace("!", "")
                    values = parts[1].strip().split("\t")
                    metadata_info[key] = values

        metadata = pd.DataFrame({"SampleID": sample_names})
        sample_info = []
        
        for sample in sample_names:
            sample_idx = sample_names.index(sample)
            combined_info = []
            for key, values in metadata_info.items():
                if sample_idx < len(values) and values[sample_idx].strip():
                    combined_info.append(values[sample_idx].strip())
            sample_info.append(" ".join(combined_info))

        metadata["Type"] = [classify_type(info) for info in sample_info]
        metadata["Description"] = sample_info
        
        print(f"[DEBUG] Sample classification summary:")
        print(metadata["Type"].value_counts())
        
    elif expr_path and meta_path and os.path.exists(expr_path) and os.path.exists(meta_path):
        expression = pd.read_csv(expr_path, index_col=0)
        metadata = pd.read_csv(meta_path)
        
        if "SampleID" not in metadata.columns:
            id_candidates = ["sample_id", "Sample_ID", "ID", "sample"]
            for candidate in id_candidates:
                if candidate in metadata.columns:
                    metadata["SampleID"] = metadata[candidate]
                    break
            else:
                metadata["SampleID"] = metadata.index
        
        if "Type" not in metadata.columns:
            desc_columns = [col for col in metadata.columns if any(keyword in col.lower() 
                           for keyword in ["description", "title", "source", "characteristics", "type", "group"])]
            
            if desc_columns:
                combined_desc = metadata[desc_columns].apply(
                    lambda x: " ".join(x.astype(str)), axis=1
                )
                metadata["Type"] = combined_desc.apply(classify_type)
            else:
                metadata["Type"] = metadata["SampleID"].apply(classify_type)
        
        print(f"[DEBUG] Sample classification summary:")
        print(metadata["Type"].value_counts())
    
    else:
        print("[INFO] No valid files provided; generating a synthetic dataset so the pipeline can run.")
        expression, metadata = make_synthetic_data()

    return expression, metadata


def make_synthetic_data(n_genes=1500, n_cancer=40, n_normal=20, seed=42):
    """Generate synthetic data for testing"""
    import numpy as np
    
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


def plot_pca(expression, metadata, out_path="pca_plot.png"):
    """PCA INFO"""
    print("[INFO] Generating PCA plot...")
    
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(expression.T)
    
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)

    label_map = metadata.set_index("SampleID")["Type"].to_dict()
    labels = np.array([label_map[s] for s in expression.columns])
    
    color_map = {"Cancer": "#FF1900", "Normal": "#0099FF", "Unclassified": "#95A5A6"}
    colors = [color_map[l] for l in labels]

    plt.figure(figsize=(10, 8), dpi=100)
    
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(pcs[idx, 0], pcs[idx, 1], 
                   c=color_map[label], label=label, 
                   alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    plt.title("PCA: Gene Expression Analysis\nCancer vs Normal Samples", 
              fontsize=14, fontweight='bold', pad=20)

    plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[DONE] PCA plot saved to: {out_path}")

    plt.show()

    print(f"[INFO] PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance")
    print(f"[INFO] PC2 explains {pca.explained_variance_ratio_[1]*100:.2f}% of variance")
    print(f"[INFO] Combined: {sum(pca.explained_variance_ratio_)*100:.2f}% of total variance")
    
    return pca.explained_variance_ratio_


def run_pca_analysis(expression, metadata, output_dir="outputs"):
    """
    Run PCA analysis and save results

    """
    os.makedirs(output_dir, exist_ok=True)

    pca_path = os.path.join(output_dir, "pca_analysis.png")
    variance_ratios = plot_pca(expression, metadata, pca_path)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(expression.T)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    
    pca_results = pd.DataFrame({
        'SampleID': expression.columns,
        'PC1': pcs[:, 0],
        'PC2': pcs[:, 1],
        'Type': [metadata.set_index("SampleID")["Type"].to_dict()[s] for s in expression.columns]
    })
    
    pca_csv_path = os.path.join(output_dir, "pca_coordinates.csv")
    pca_results.to_csv(pca_csv_path, index=False)
    print(f"[DONE] PCA coordinates saved to: {pca_csv_path}")
    
    return variance_ratios, pca_results

if __name__ == "__main__":
    print("Loading data...")
    expr, meta = load_data()
    
    print("\nExpression data shape:", expr.shape)
    print("Metadata shape:", meta.shape)
    print("\nMetadata preview:")
    print(meta.head())
    print("\nType distribution:")
    print(meta["Type"].value_counts())
    
    print("\n" + "="*50)
    print("RUNNING PCA ANALYSIS")
    print("="*50)

    variance_ratios, pca_results = run_pca_analysis(expr, meta)
    
    print("\n" + "="*50)
    print("PCA ANALYSIS COMPLETE")
    print("="*50)
    print(f"✅ PCA plot generated and saved")
    print(f"✅ PCA coordinates saved to CSV")
    print(f"✅ Total variance explained: {sum(variance_ratios)*100:.1f}%")
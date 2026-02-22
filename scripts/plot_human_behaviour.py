import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

df = pd.read_csv("outputs/human_stats_by_category.csv")

# -----------------------------------
# 1️⃣ Behavioural Space (Center vs Entropy)
# -----------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(data=df,
                x="mean_center_distance",
                y="mean_entropy",
                s=100)

for i in range(len(df)):
    plt.text(df.iloc[i]["mean_center_distance"],
             df.iloc[i]["mean_entropy"],
             df.iloc[i]["category"],
             fontsize=8)

plt.title("Behavioural Space of CAT2000 Categories")
plt.xlabel("Mean Center Distance")
plt.ylabel("Mean Entropy")
plt.tight_layout()
plt.savefig("outputs/behaviour_space.png")
plt.close()

# -----------------------------------
# 2️⃣ Dispersion vs Entropy
# -----------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(data=df,
                x="mean_dispersion",
                y="mean_entropy",
                s=100)

for i in range(len(df)):
    plt.text(df.iloc[i]["mean_dispersion"],
             df.iloc[i]["mean_entropy"],
             df.iloc[i]["category"],
             fontsize=8)

plt.title("Dispersion vs Entropy")
plt.xlabel("Mean Dispersion")
plt.ylabel("Mean Entropy")
plt.tight_layout()
plt.savefig("outputs/dispersion_vs_entropy.png")
plt.close()

# -----------------------------------
# 3️⃣ Heatmap of Standardised Metrics
# -----------------------------------
metrics = df[["mean_center_distance", "mean_dispersion", "mean_entropy"]]

metrics_norm = (metrics - metrics.mean()) / metrics.std()
metrics_norm.index = df["category"]

plt.figure(figsize=(8,10))
sns.heatmap(metrics_norm,
            cmap="coolwarm",
            annot=False)

plt.title("Standardised Behavioural Metrics by Category")
plt.tight_layout()
plt.savefig("outputs/category_heatmap.png")
plt.close()

# -----------------------------------
# 4️⃣ Correlation Matrix
# -----------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(metrics.corr(),
            annot=True,
            cmap="coolwarm",
            vmin=-1, vmax=1)

plt.title("Correlation Between Behavioural Metrics")
plt.tight_layout()
plt.savefig("outputs/metric_correlations.png")
plt.close()

print("All plots saved to outputs/")
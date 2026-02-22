import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("outputs/human_consensus_peaks_by_category.csv")

# 1) Bar: mean_peak_prob (sorted)
df1 = df.sort_values("mean_peak_prob", ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=df1, x="category", y="mean_peak_prob")
plt.xticks(rotation=90)
plt.title("Human Consensus Proxy by Category (Peak Probability)")
plt.xlabel("")
plt.ylabel("Mean peak probability (blurred fixation density)")
plt.tight_layout()
plt.savefig("outputs/consensus_peakprob_bar.png", dpi=200)
plt.close()

# 2) Bar: mean_num_peaks (sorted)
df2 = df.sort_values("mean_num_peaks", ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=df2, x="category", y="mean_num_peaks")
plt.xticks(rotation=90)
plt.title("Multi-modality of Attention by Category (Number of Peaks)")
plt.xlabel("")
plt.ylabel("Mean number of peaks")
plt.tight_layout()
plt.savefig("outputs/num_peaks_bar.png", dpi=200)
plt.close()

# 3) Scatter: peak prob vs num peaks (the sexy plot)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="mean_num_peaks", y="mean_peak_prob", s=120)

for _, r in df.iterrows():
    plt.text(r["mean_num_peaks"], r["mean_peak_prob"], r["category"], fontsize=8)

plt.title("Consensus vs Multi-modality (Category Behaviour Map)")
plt.xlabel("Mean number of peaks")
plt.ylabel("Mean peak probability")
plt.tight_layout()
plt.savefig("outputs/consensus_vs_multimodality.png", dpi=200)
plt.close()

print("Saved plots to outputs/:")
print(" - consensus_peakprob_bar.png")
print(" - num_peaks_bar.png")
print(" - consensus_vs_multimodality.png")

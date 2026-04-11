import pandas as pd

# Human behavioural stats
human_stats = pd.read_csv("outputs/human_stats_by_category.csv")
human_peaks = pd.read_csv("outputs/human_consensus_peaks_by_category.csv")

# Merge human files first
human = human_stats.merge(
    human_peaks[["category", "mean_num_peaks", "mean_peak_prob"]],
    on="category"
)

# DeepGaze results
deepgaze = pd.read_csv("outputs/samresnet_per_category_behaviour.csv")

# Merge human vs model
merged = human.merge(deepgaze, on="category")

# Deltas
merged["delta_entropy"] = merged["model_entropy"] - merged["mean_entropy"]
merged["delta_center_distance"] = merged["model_center_distance"] - merged["mean_center_distance"]
merged["delta_num_peaks"] = merged["model_num_peaks"] - merged["mean_num_peaks"]

merged.to_csv("outputs/human_vs_samresnet_comparison.csv", index=False)

print(merged[[
    "category",
    "mean_entropy", "model_entropy", "delta_entropy",
    "mean_center_distance", "model_center_distance", "delta_center_distance",
    "mean_num_peaks", "model_num_peaks", "delta_num_peaks"
]])
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_and_plot(base_dir="patch_results"):
    # 1. Find all CSV files recursively
    all_csv_files = glob.glob(f"{base_dir}/**/*.csv", recursive=True)
    
    if not all_csv_files:
        print("No CSV files found!")
        return

    print(f"Found {len(all_csv_files)} files. Loading and aggregating...")
    
    # 2. Load and concatenate all data
    df_list = []
    for file in all_csv_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Could not read {file}: {e}")
            
    master_df = pd.concat(df_list, ignore_index=True)

    # 3. Clean the Data!
    # Drop rows where the score is exactly 0 (usually means baseline_diff was 0 / blank timestamp)
    clean_df = master_df[(master_df['Logit_Score'] != 0) & (master_df['Posteriors_Score'] != 0)].copy()
    
    # Clip extreme mathematical overshoots so they don't ruin our averages
    # We cap at 1.5 for overshoots, and -0.5 for undershoots
    clean_df['Logit_Score'] = clean_df['Logit_Score'].clip(-0.5, 1.5)
    
    # 4. Group by Component, Layer, and Relative Time
    # This averages across ALL reads and ALL valid timestamps!
    aggregated_df = clean_df.groupby(['Component', 'Metric_Mode', 'Layer', 'Time_Offset']).agg(
        Mean_Score=('Logit_Score', 'mean'),
        Variance=('Logit_Score', 'var'), # Helps find heads that act wildly different across reads
        Count=('Logit_Score', 'count')   # How many valid reads contributed to this pixel
    ).reset_index()

    # 5. Generate Master Heatmaps
    components = aggregated_df['Component'].unique()
    modes = aggregated_df['Metric_Mode'].unique()

    os.makedirs("aggregated_plots", exist_ok=True)

    for mode in modes:
        for comp in components:
            # Filter for specific component and mode (e.g., "head_2", "recovery")
            plot_df = aggregated_df[(aggregated_df['Component'] == comp) & (aggregated_df['Metric_Mode'] == mode)]
            
            if plot_df.empty:
                continue

            # Pivot to create the heatmap matrix
            heatmap_matrix = plot_df.pivot(index="Layer", columns="Time_Offset", values="Mean_Score")
            
            plt.figure(figsize=(10, 8))
            # vmin and vmax force the color scale to stay consistent across all plots!
            sns.heatmap(heatmap_matrix, cmap="RdBu_r", center=0.5, vmin=0.0, vmax=1.0) 
            
            plt.title(f"AGGREGATED {comp} - {mode}\n(Averaged across {plot_df['Count'].max()} read windows)")
            plt.xlabel("Relative Time Offset")
            plt.ylabel("Transformer Layer")
            
            plt.tight_layout()
            plt.savefig(f"aggregated_plots/Aggregated_{comp}_{mode}.png", dpi=300)
            plt.close()

    print("Aggregation complete! Check the 'aggregated_plots' folder.")

# Run the function
aggregate_and_plot()
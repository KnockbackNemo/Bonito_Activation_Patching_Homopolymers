import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from bonito.reader import Reader, read_chunks
import seaborn as sns
from pathlib import Path
import os
import re
import torch

# ==========================================
# CONFIGURATION & TOGGLES
# ==========================================
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

DIR_EXP = Path("./patch_results")           
DIR_CTRL = Path("./patch_results_control")  
OUTPUT_DIR = Path("./comparative_analysis_outputs")

GENERATE_ATTENTION_MAPS = False 
MODEL_PATH = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0" 

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_data_from_dir(component_name, metric_type, data_dir, group_label): ## TODO Add ctr/exp type
    if not data_dir.exists():
        return None
        
    search_pattern = f"*{component_name}*{metric_type}*.csv"
    csv_files = list(data_dir.rglob(search_pattern))

    if not csv_files:
        return None

    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        if temp_df.empty: continue
            
        temp_df['Logit_Score'] = pd.to_numeric(temp_df['Logit_Score'], errors='coerce')
        temp_df['Posteriors_Score'] = pd.to_numeric(temp_df['Posteriors_Score'], errors='coerce')
        
        temp_df['Group'] = group_label 
        temp_df['Component'] = component_name 
        
        # Extract Read ID and match to metadata
        match = re.search(r'R(\d+)r(\d+)', file.name)
        if match:
            read_num = match.group(1)
            row_num = int(match.group(2))
            temp_df['Read_ID'] = f"R{read_num}r{row_num}"
            
            # ---> METADATA RE-INTEGRATION <---
            # Note: Adjust this path if your metadata CSVs are stored elsewhere!
            meta_csv_path = Path(f"./data/pairs/Input_gen_results_read_{read_num}.csv")
            
            if meta_csv_path.exists():
                meta_df = pd.read_csv(meta_csv_path)
                
                # Make sure the row exists in the metadata file
                if row_num < len(meta_df):
                    meta_row = meta_df.iloc[row_num]
                    
                    # Using .get() allows us to safely grab the data even if your column names 
                    # use slightly different formatting (e.g. 'Base Letter' vs 'Base_Letter')
                    temp_df['Base Letter'] = meta_row.get('Base_Letter', meta_row.get('Base Letter', 'Unknown'))
                    temp_df['Clean H-er Length'] = meta_row.get('Clean_Length', meta_row.get('Clean H-er Length', np.nan))
                    temp_df['Error Type'] = meta_row.get('Error_Type', meta_row.get('Error Type', 'Unknown'))
                else:
                    temp_df['Base Letter'] = 'Unknown'
                    temp_df['Clean H-er Length'] = np.nan
                    temp_df['Error Type'] = 'Unknown'
            else:
                temp_df['Base Letter'] = 'Unknown'
                temp_df['Clean H-er Length'] = np.nan
                temp_df['Error Type'] = 'Unknown'
        else:
            # Fallback if filename format doesn't match
            temp_df['Read_ID'] = file.name.split('_')[0]
            temp_df['Base Letter'] = 'Unknown'
            temp_df['Clean H-er Length'] = np.nan
            temp_df['Error Type'] = 'Unknown'
            
        df_list.append(temp_df)
    
    if not df_list: return None
    return pd.concat(df_list, ignore_index=True)

def preprocess_layerwise_metrics(df):
    if df is None or df.empty: return None, None
    
    grouping_cols = ['Group', 'Component', 'Read_ID', 'Layer', 'Base Letter', 'Clean H-er Length', 'Error Type']
    
    # dropna=False ensures we don't lose data if metadata is missing
    aie_df = df.groupby(grouping_cols, dropna=False)[['Logit_Score', 'Posteriors_Score']].mean().reset_index()
    max_df = df.groupby(grouping_cols, dropna=False)[['Logit_Score', 'Posteriors_Score']].max().reset_index()
    
    return aie_df, max_df

# ==========================================
# 2. DATA EXPORT & PLOTTING (COMBINED)
# ==========================================
def export_summary_stats(df, filename_prefix, plot_dir, stat_type="AIE"):
    stats_logit = df.groupby(['Group', 'Component', 'Layer'])['Logit_Score'].agg(['mean', 'std', 'sem', 'count']).reset_index()
    stats_post = df.groupby(['Group', 'Component', 'Layer'])['Posteriors_Score'].agg(['mean', 'std', 'sem', 'count']).reset_index()
    
    merged_stats = pd.merge(stats_logit, stats_post, on=['Group', 'Component', 'Layer'], suffixes=('_Logit', '_Posteriors'))
    
    csv_path = plot_dir / f"{filename_prefix}_Numerical_{stat_type}_Stats.csv"
    merged_stats.to_csv(csv_path, index=False)

def plot_all_components_combined(df, score_column, metric_type, plot_dir, filter_suffix="All_Data", stat_type="AIE"):
    if df is None or df.empty: return
    
    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(data=df, x='Layer', y=score_column, hue='Component', 
                      style='Group', markers=True, dashes=True, errorbar=None)
    
    plt.title(f'All Components Combined: {stat_type} of {score_column} across Layers\n({metric_type.upper()} | {filter_suffix.replace("_", " ")})', fontsize=14)
    plt.ylabel(f'{stat_type} Score')
    plt.xlabel('Layer Index')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(plot_dir / f"Combined_{stat_type}_{score_column}_{filter_suffix}.png", dpi=300)
    plt.close()

# ==========================================
# 3. INDIVIDUAL COMPONENT PLOTTING & SUMMARY
# ==========================================
def generate_summary_report(df, component_name, metric_type, plot_dir):
    """Generates a text report detailing the breakdown of input data."""
    if df is None or df.empty: return
    
    report_path = plot_dir / f"{component_name}_{metric_type}_Summary_Report.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"=== SUMMARY REPORT: {component_name.upper()} | {metric_type.upper()} ===\n")
        f.write(f"Total Unique Reads: {df['Read_ID'].nunique()}\n\n")
        
        # Breakdown by Control vs Experiment
        for group in df['Group'].unique():
            group_df = df[df['Group'] == group]
            # Drop duplicates so we only count each read once (not once per layer)
            unique_reads_df = group_df.drop_duplicates(subset=['Read_ID'])
            
            f.write(f"--- Group: {group} ---\n")
            f.write(f"Reads in Group: {len(unique_reads_df)}\n\n")
            
            f.write("Base Letter Breakdown:\n")
            f.write(unique_reads_df['Base Letter'].value_counts(dropna=False).to_string() + "\n\n")
            
            f.write("Error Type Breakdown:\n")
            f.write(unique_reads_df['Error Type'].value_counts(dropna=False).to_string() + "\n\n")
            
            f.write("Clean H-er Length Breakdown:\n")
            f.write(unique_reads_df['Clean H-er Length'].value_counts(dropna=False).to_string() + "\n\n")
            f.write("-" * 40 + "\n\n")

def plot_individual_comparative_line(df, score_column, component_name, metric_type, plot_dir, stat_type):
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x='Layer', y=score_column, hue='Group', 
                      style='Group', markers=True, dashes=True, errorbar=('ci', 95))
    plt.title(f'{stat_type} {score_column} across Layers\n({component_name.upper()} | {metric_type.upper()})', fontsize=14)
    plt.ylabel(f'{stat_type} Score')
    plt.xlabel('Layer Index')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(plot_dir / f"{component_name}_{stat_type}_line_{score_column}.png", dpi=300)
    plt.close()

def plot_individual_comparative_bar(df, score_column, component_name, metric_type, plot_dir, stat_type):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Layer', y=score_column, hue='Group', errorbar='se', capsize=.1)
    plt.title(f'{stat_type} {score_column} by Layer (Bar)\n({component_name.upper()} | {metric_type.upper()})', fontsize=14)
    plt.ylabel(f'Mean {stat_type} Score (+/- SE)')
    plt.xlabel('Layer Index')
    plt.tight_layout()
    plt.savefig(plot_dir / f"{component_name}_{stat_type}_bar_{score_column}.png", dpi=300)
    plt.close()

def plot_individual_distribution_violin(df, score_column, component_name, metric_type, plot_dir, stat_type):
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Layer', y=score_column, hue='Group', split=True, inner="quartile", alpha=0.7)
    plt.title(f'Data Distribution: {stat_type} {score_column} across Layers\n({component_name.upper()} | {metric_type.upper()})')
    plt.ylabel(f'{stat_type} Score')
    plt.xlabel('Layer Index')
    plt.tight_layout()
    plt.savefig(plot_dir / f"{component_name}_{stat_type}_violin_{score_column}.png", dpi=300)
    plt.close()

def generate_individual_plots_and_summary(df, component_name, metric_type, base_dir, stat_type="AIE", generate_report=False):
    if df is None or df.empty: return
    
    plot_dir = base_dir / "Individual_Components"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the text report only once per component
    if generate_report:
        generate_summary_report(df, component_name, metric_type, plot_dir)
    
    for score in ['Logit_Score', 'Posteriors_Score']:
        plot_individual_comparative_line(df, score, component_name, metric_type, plot_dir, stat_type)
        plot_individual_comparative_bar(df, score, component_name, metric_type, plot_dir, stat_type)
        plot_individual_distribution_violin(df, score, component_name, metric_type, plot_dir, stat_type)

# ==========================================
# 4. FILTERING ENGINE
# ==========================================
def filter_and_plot_combined(df_aie, df_max, metric_type, plot_dir, base_type=None, min_length=None, max_length=None, error_type=None):
    filtered_aie = df_aie.copy()
    filtered_max = df_max.copy()
    filter_tags = []
    
    if base_type:
        filtered_aie = filtered_aie[filtered_aie['Base Letter'] == base_type]
        filtered_max = filtered_max[filtered_max['Base Letter'] == base_type]
        filter_tags.append(f"Base_{base_type}")
        
    if min_length is not None:
        filtered_aie = filtered_aie[filtered_aie['Clean H-er Length'] >= min_length]
        filtered_max = filtered_max[filtered_max['Clean H-er Length'] >= min_length]
        filter_tags.append(f"MinLen_{min_length}")
        
    if max_length is not None:
        filtered_aie = filtered_aie[filtered_aie['Clean H-er Length'] <= max_length]
        filtered_max = filtered_max[filtered_max['Clean H-er Length'] <= max_length]
        filter_tags.append(f"MaxLen_{max_length}")
        
    if error_type:
        filtered_aie = filtered_aie[filtered_aie['Error Type'] == error_type]
        filtered_max = filtered_max[filtered_max['Error Type'] == error_type]
        filter_tags.append(f"Error_{error_type}")
        
    if filtered_aie.empty or filtered_max.empty:
        print(f"⚠️ Filter {filter_tags} resulted in an empty dataframe. Skipping plot.")
        return
        
    suffix = "_".join(filter_tags) if filter_tags else "All_Data"
    
    for score in ['Logit_Score', 'Posteriors_Score']:
        plot_all_components_combined(filtered_aie, score, metric_type, plot_dir, filter_suffix=suffix, stat_type="AIE")
        plot_all_components_combined(filtered_max, score, metric_type, plot_dir, filter_suffix=suffix, stat_type="Max")
        
    print(f"📊 Generated combined AIE and Max plots for: {suffix}")

# ==========================================
# 5. ATTENTION MAP GENERATION
# ==========================================
def generate_attention_maps_for_first_read(model, clean_input_tensor, layer_idx=16, head_idx=5):
    print(f"Generating attention map for Layer {layer_idx}, Head {head_idx}...")
    d_model, n_heads = 512, 8
    head_dim = d_model // n_heads
    
    with model.trace(clean_input_tensor):
        attn_module = model.encoder.transformer_encoder.layers[layer_idx].self_attn
        hidden_states = attn_module.input[0][0].save() 

    x = hidden_states.value 
    if x.shape[0] != 1: x = x.transpose(0, 1) 
    x = x[0] 
    
    in_proj_weight = attn_module.in_proj_weight.detach().cpu()
    in_proj_bias = attn_module.in_proj_bias.detach().cpu() if attn_module.in_proj_bias is not None else 0
    
    q_weight, k_weight = in_proj_weight[:d_model, :], in_proj_weight[d_model:2*d_model, :]
    q_bias = in_proj_bias[:d_model] if torch.is_tensor(in_proj_bias) else 0
    k_bias = in_proj_bias[d_model:2*d_model] if torch.is_tensor(in_proj_bias) else 0
    
    Q = torch.matmul(x, q_weight.t()) + q_bias
    K = torch.matmul(x, k_weight.t()) + k_bias
    
    start_dim, end_dim = head_idx * head_dim, (head_idx + 1) * head_dim
    Q_head, K_head = Q[:, start_dim:end_dim], K[:, start_dim:end_dim]
    
    attention_scores = torch.matmul(Q_head, K_head.t()) / np.sqrt(head_dim)
    attention_probs = torch.softmax(attention_scores, dim=-1).numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_probs, cmap="viridis")
    plt.title(f"Attention Map: Layer {layer_idx}, Head {head_idx}")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"Attention_Map_L{layer_idx}_H{head_idx}.png", dpi=300)
    plt.close()
    print("Attention map saved.")

def load_tensor(read: int, row: int, csv_path: str, model):
    reader = Reader(csv_path)

    reads = reader.get_reads(
        csv_path, 
        do_trim=True,
        scaling_strategy=model.config.get("scaling"),
        norm_params=model.config.get("standardisation")
    )

    df_inputpairs = pd.read_csv(csv_path)
    
    for i in range(0, read):
        read_data = next(reader.reads()) 
    
    raw_stndrd_signal = read_data.signal

    def safe_parse(val, cast_type):
        if pd.isna(val): return None
        if isinstance(val, str): val = val.strip('[]')
        return cast_type(val)

    raw_start = safe_parse(row['raw start idx'], int)
    raw_end = safe_parse(row['raw end idx'], int)
    
    clean_signal = raw_stndrd_signal[raw_start:raw_end].copy()
    clean_input = torch.tensor(clean_signal, dtype=torch.float16).view(1, 1, -1).to(model.device)
    
    return clean_input
# ==========================================
# 6. MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    components = ["mlp", "attn", "layer"] + [f"head {h}" for h in range(8)]
    metric_types = ["noising", "denoising"]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for metric in metric_types:
        print(f"\n--- Processing Metric: {metric.upper()} ---")
        all_components_raw_df_list = []
        
        # 1. Load all components into one massive dataframe
        for comp in components:
            df_exp = load_data_from_dir(comp, metric, DIR_EXP, 'Experiment (Homopolymer)')
            df_ctrl = load_data_from_dir(comp, metric, DIR_CTRL, 'Control (Single Error)')
            
            if df_exp is not None: all_components_raw_df_list.append(df_exp)
            if df_ctrl is not None: all_components_raw_df_list.append(df_ctrl)
            
        if not all_components_raw_df_list:
            print(f"No data found for {metric}.")
            continue
            
        combined_raw_df = pd.concat(all_components_raw_df_list, ignore_index=True)
        
        # 2. Calculate both AIE (Mean) and Max metrics
        master_aie_df, master_max_df = preprocess_layerwise_metrics(combined_raw_df)
        
        # 3. Setup output directory
        metric_plot_dir = OUTPUT_DIR / f"{metric}_combined_analysis"
        metric_plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. Export Master Stats for both
        export_summary_stats(master_aie_df, f"Master_{metric}", metric_plot_dir, stat_type="AIE")
        export_summary_stats(master_max_df, f"Master_{metric}", metric_plot_dir, stat_type="Max")
        
        # 5. ---> GENERATE INDIVIDUAL COMPONENT PLOTS & SUMMARY REPORTS <---
        print("Generating individual component plots and summaries...")
        for comp in components:
            comp_aie_df = master_aie_df[master_aie_df['Component'] == comp]
            comp_max_df = master_max_df[master_max_df['Component'] == comp]
            
            # Use the AIE dataframe to generate the summary text report once per component
            if not comp_aie_df.empty:
                generate_individual_plots_and_summary(comp_aie_df, comp, metric, metric_plot_dir, stat_type="AIE", generate_report=True)
            if not comp_max_df.empty:
                generate_individual_plots_and_summary(comp_max_df, comp, metric, metric_plot_dir, stat_type="Max", generate_report=False)

        # 6. Generate the All-Components Plot (Unfiltered)
        print("Generating combined plots...")
        filter_and_plot_combined(master_aie_df, master_max_df, metric, metric_plot_dir)
        
        # 7. Generate Filtered Plots
        filter_and_plot_combined(master_aie_df, master_max_df, metric, metric_plot_dir, base_type='A')
        filter_and_plot_combined(master_aie_df, master_max_df, metric, metric_plot_dir, min_length=6)

    # Optional Attention Maps
    if GENERATE_ATTENTION_MAPS:
        from bonito import util
        from nnsight import NNsight
        
        print("Loading model for attention maps...")
        bonito_model = util.load_model(MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
        nnsight_model = NNsight(bonito_model._orig_mod)
        
        csv_path = "./data/pairs/Input_gen_results_read_1.csv"
        # Load the dummy input using the updated load_tensor function
        dummy_input = load_tensor(read=1, row=0, csv_path=csv_path, model=bonito_model) 
        generate_attention_maps_for_first_read(nnsight_model, dummy_input, layer_idx=16, head_idx=5)
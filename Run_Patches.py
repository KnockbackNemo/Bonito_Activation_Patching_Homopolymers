"""
Run activation patching on pairs of input data sets using input_gen csvs and raw
data to recreate the data pairs needed.
"""
import torch
import os
import gc
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bonito.reader import Reader
from pathlib import Path
import Code.helpers as helpers

##############################
######## CONFIG ##############
##############################

BONITO_MODEL_PATH = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
DATA_DIR = "./reads/"
IS_CONTROL = False      # Set True to run against negative controls

INPUT_DIR     = Path("./Intermediate_Data/clean_corrupt_pairs/selected/Nonhomopolymer/") if IS_CONTROL else Path("./Intermediate_Data/clean_corrupt_pairs/selected/Homopolymer/")
OUTPUT_PREFIX = "Results_and_Figures/Patching/Nonhomopolymer" if IS_CONTROL else "Results_and_Figures/Patching/Homopolymer"

CONTEXT_PADDING = 200
NOISE_SIZE = 6
NUM_T_LAYERS = 18
NUM_HEADS = 8

##############################
######## MODEL SETUP #########
##############################

model, bonito_model, model_dtype = helpers.Setup_Model(BONITO_MODEL_PATH)
file_name = Path(__file__).stem

##############################
########## FUNCTIONS #########
##############################

def run_patching_sweep(model, source_input, target_input, check_output_timestamps_list, patch_start, patch_end,
                       num_layers, component="mlp", head_idx=None, metric_mode="recovery"):
    """
    component options: "layer", "mlp", "attn", "head"
    """
    print(f"Sweeping {component} {metric_mode}...")

    results = []

    with model.trace(source_input):
        source_output_proxy = model.output.save()
    source_output = source_output_proxy.detach()

    with model.trace(target_input):
        target_output_proxy = model.output.save()
    target_output = target_output_proxy.detach()

    global_mse_baseline_diff = torch.nn.functional.mse_loss(source_output.to(torch.float32), target_output.to(torch.float32)).item()
    source_posteriors = bonito_model.seqdist.posteriors(source_output.to(torch.float32)) + 1e-8
    target_posteriors = bonito_model.seqdist.posteriors(target_output.to(torch.float32)) + 1e-8

    source_acts_cache = {}
    with model.trace(source_input):
        for layer_idx in range(num_layers):
            layer = model.encoder.transformer_encoder[layer_idx]
            if component == "mlp":
                source_acts_cache[layer_idx] = layer.ff.fc2.output.save()
            elif component == "attn":
                source_acts_cache[layer_idx] = layer.self_attn.output[0].save()
            elif component == "head":
                source_acts_cache[layer_idx] = layer.self_attn.out_proj.input[0].save()
            elif component == "layer":
                source_acts_cache[layer_idx] = layer.output[0].save()

    for k in source_acts_cache:
        source_acts_cache[k] = source_acts_cache[k].detach()

    for layer_idx in range(num_layers):
        layer = model.encoder.transformer_encoder[layer_idx]
        src_act = source_acts_cache[layer_idx]

        for time_offset, t in enumerate(range(patch_start, patch_end)):
            with model.trace(target_input):
                start = max(0, t - 1)
                seq_len, d_model = src_act.shape[-2:]
                assert seq_len != 1 and d_model != 1
                end = min(seq_len, t + 2)

                if component == "mlp":
                    target = layer.ff.fc2.output.clone()
                    assert target.numel() == seq_len * d_model
                    target[0, start:end, :] = src_act[0, start:end, :]
                    layer.ff.fc2.output = target

                elif component == "attn":
                    target = layer.self_attn.output[0].clone()
                    assert target.numel() == seq_len * d_model
                    target[start:end, :] = src_act[start:end, :]
                    layer.self_attn.output[0] = target

                elif component == "head":
                    target = layer.self_attn.out_proj.input[0]
                    assert target.numel() == seq_len * d_model
                    nhead, head_dim = 8, 64
                    target_reshaped = target.clone().reshape(-1, nhead, head_dim)
                    src_reshaped = src_act.reshape(-1, nhead, head_dim)
                    target_reshaped[start:end, head_idx, :] = src_reshaped[start:end, head_idx, :]
                    patched = target_reshaped.reshape(seq_len, d_model)
                    layer.self_attn.out_proj.input = patched

                elif component == "layer":
                    target = layer.output.clone()
                    target[0][start:end, :] = src_act[start:end, :]
                    layer.output = target

                patched_scores_proxy = model.output.save()

            patched_scores = patched_scores_proxy.detach()
            patched_posteriors = bonito_model.seqdist.posteriors(patched_scores.to(torch.float32)) + 1e-8

            for timestep in check_output_timestamps_list:
                target_transition_idx = torch.argmax(source_output[timestep, 0, :]).item()
                wrong_transition_idx  = torch.argmax(target_output[timestep, 0, :]).item()
                logit_diff_source = source_output[timestep, 0, target_transition_idx] - source_output[timestep, 0, wrong_transition_idx]
                logit_diff_target = target_output[timestep, 0, target_transition_idx] - target_output[timestep, 0, wrong_transition_idx]
                baseline_logit_diff = logit_diff_source - logit_diff_target

                if baseline_logit_diff == 0:
                    continue

                target_logit        = patched_scores[timestep, 0, target_transition_idx].item()
                wrong_logit         = patched_scores[timestep, 0, wrong_transition_idx].item()
                actual_max_pred_idx = torch.argmax(patched_scores[timestep, 0, :]).item()
                target_string  = helpers.transition_idx_to_str(bonito_model, target_transition_idx)
                wrong_string   = helpers.transition_idx_to_str(bonito_model, wrong_transition_idx)
                actual_string  = helpers.transition_idx_to_str(bonito_model, actual_max_pred_idx)

                reuse_logit_transitions = (torch.argmax(source_posteriors[timestep, 0, :]).item()
                                           == torch.argmax(target_posteriors[timestep, 0, :]).item())
                posteriors_target_transition_idx = target_transition_idx if reuse_logit_transitions else torch.argmax(source_posteriors[timestep, 0, :]).item()
                posteriors_wrong_transition_idx  = wrong_transition_idx  if reuse_logit_transitions else torch.argmax(target_posteriors[timestep, 0, :]).item()
                posteriors_actual_transition_idx = torch.argmax(patched_posteriors[timestep, 0, :]).item()

                target_state_prob = patched_posteriors[timestep, 0, posteriors_target_transition_idx]
                wrong_state_prob  = patched_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                actual_state_prob = patched_posteriors[timestep, 0, posteriors_actual_transition_idx]

                posteriors_target_string = helpers.transition_idx_to_str(bonito_model, posteriors_target_transition_idx)
                posteriors_wrong_string  = helpers.transition_idx_to_str(bonito_model, posteriors_wrong_transition_idx)
                posteriors_actual_string = helpers.transition_idx_to_str(bonito_model, posteriors_actual_transition_idx)

                patched_logit_diff = patched_scores[timestep, 0, target_transition_idx] - patched_scores[timestep, 0, wrong_transition_idx]

                if baseline_logit_diff == 0:
                    logit_score = 0
                elif metric_mode == "recovery":
                    logit_score = (1 - (logit_diff_source - patched_logit_diff) / baseline_logit_diff).item()
                elif metric_mode == "degradation":
                    logit_score = ((logit_diff_source - patched_logit_diff) / baseline_logit_diff).item()

                prob_diff_source   = source_posteriors[timestep, 0, posteriors_target_transition_idx] - source_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                prob_diff_target   = target_posteriors[timestep, 0, posteriors_target_transition_idx] - target_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                patched_prob_diff  = target_state_prob - wrong_state_prob
                baseline_prob_diff = prob_diff_source - prob_diff_target

                if baseline_prob_diff == 0:
                    posteriors_score = 0
                elif metric_mode == "recovery":
                    posteriors_score = (1 - (prob_diff_source - patched_prob_diff) / baseline_prob_diff).item()
                elif metric_mode == "degradation":
                    posteriors_score = ((prob_diff_source - patched_prob_diff) / baseline_prob_diff).item()

                is_target_logit_argmax = actual_max_pred_idx == target_transition_idx
                is_target_prob_argmax  = posteriors_actual_transition_idx == posteriors_target_transition_idx
                global_mse_diff = torch.nn.functional.mse_loss(source_output.to(torch.float32), patched_scores.to(torch.float32)).item()
                should_decode = not is_target_logit_argmax or not is_target_prob_argmax or (global_mse_diff > global_mse_baseline_diff)

                results.append({
                    "Layer": layer_idx,
                    "Time_Offset": time_offset,
                    "Target_Timestep": timestep,
                    "Component": component if head_idx is None else f"head_{head_idx}",
                    "Metric_Mode": metric_mode,
                    "Logit_Score": logit_score,
                    "Posteriors_Score": posteriors_score,
                    "Target_Transition_Prior": target_transition_idx,
                    "Target_String_Prior": target_string,
                    "Target_Logit": target_logit,
                    "Wrong_Transition_Prior": wrong_transition_idx,
                    "Wrong_String_Prior": wrong_string,
                    "Wrong_Logit": wrong_logit,
                    "Actual_Argmax_Prior": actual_max_pred_idx,
                    "Actual_String_Prior": actual_string,
                    "Actual_Argmax_Logit": patched_scores[timestep, 0, actual_max_pred_idx].item(),
                    "Target_Transition_Post": posteriors_target_transition_idx,
                    "Target_String_Post": posteriors_target_string,
                    "Target_Prob": target_state_prob.item(),
                    "Wrong_Transition_Post": posteriors_wrong_transition_idx,
                    "Wrong_String_Post": posteriors_wrong_string,
                    "Wrong_Prob": wrong_state_prob.item(),
                    "Actual_Transition_Post": posteriors_actual_transition_idx,
                    "actual_String_Post": posteriors_actual_string,
                    "actual_Prob": actual_state_prob.item(),
                    "Is_Target_Logit_Argmax": is_target_logit_argmax,
                    "Is_Target_Prob_Argmax": is_target_prob_argmax,
                    "Should_Decode": should_decode,
                    "Global_MSE_Change": global_mse_diff - global_mse_baseline_diff,
                    "New_String": None
                })

            del patched_scores_proxy, patched_scores

    if not results:
        return pd.DataFrame(columns=["Layer", "Time_Offset", "Target_Timestep", "Logit_Score"])
    return pd.DataFrame(results)


def plot_and_save_outputs(df, component="mlp", folder_name="default", index="none", plot=False):
    folder_path = f"{OUTPUT_PREFIX}/{file_name}/read_{folder_name}/row_{index}"
    os.makedirs(folder_path, exist_ok=True)
    csv_filename = f"R{folder_name}r{index}_{component}_data_step_{-1}.csv"

    if df.empty:
        print(f"Skipping plot and making empty save for {component} (No differences found & DataFrame empty.)")
        df.to_csv(os.path.join(folder_path, csv_filename), index=False)
        return

    for step in df['Target_Timestep'].unique():
        step_df = df[df['Target_Timestep'] == step]
        csv_filename = f"R{folder_name}r{index}_{component}_data_step_{step}.csv"
        csv_full_path = os.path.join(folder_path, csv_filename)

        if plot:
            for score_col, label in [("Logit_Score", "logit"), ("Posteriors_Score", "posteriors")]:
                png_filename = f"R{folder_name}r{index}_{component}_{label}_heatmap_results_stp_{step}.png"
                heatmap_matrix = step_df.pivot(index="Layer", columns="Time_Offset", values=score_col)
                plt.figure()
                sns.heatmap(heatmap_matrix)
                plt.title(f"{component} {label} heatmap results timestep {step}")
                plt.xlabel("Time ticks")
                plt.ylabel("Layer")
                plt.savefig(os.path.join(folder_path, png_filename))
                plt.close()

        step_df.to_csv(csv_full_path, index=False)


def check_if_run_exists(file_name, read_idx, row_idx, component, timestamps):
    """ Only looks at the read & row level, not the timestep level. If any timesteps have been run,
    it won't run again, so you may need to be careful about missing timesteps. """
    folder_path = f"{OUTPUT_PREFIX}/{file_name}/read_{read_idx}/row_{row_idx}"
    for step in timestamps:
        csv_filename = f"R{read_idx}r{row_idx}_{component}_data_step_{step}.csv"
        if os.path.exists(os.path.join(folder_path, csv_filename)):
            return True
    print(f"No files found for folder path {folder_path} {component}")
    return False


##############################
######## DATA LOADING ########
##############################

available_csvs = {}
for csv_path in INPUT_DIR.glob("*.csv"):
    match = re.search(r'read_(\d+)', csv_path.name)
    if match:
        available_csvs[int(match.group(1))] = csv_path

print(f"Found CSVs for reads: {list(available_csvs.keys())}, {list(available_csvs.values())}")

reader = Reader(DATA_DIR)
reads = reader.get_reads(
    DATA_DIR,
    do_trim=True,
    scaling_strategy=model.config.get("scaling"),
    norm_params=model.config.get("standardisation")
)

##############################
######## MAIN LOOP ###########
##############################

for current_read_idx, read_data in enumerate(reads, start=1):

    if current_read_idx not in available_csvs:
        continue

    print(f"\n--- Processing Read {current_read_idx} ---")
    csv_path = available_csvs[current_read_idx]

    if os.path.getsize(csv_path) == 0:
        print(f"Skipping {csv_path.name}: File is empty.")
        continue

    try:
        df_inputpairs = pd.read_csv(csv_path)
        if df_inputpairs.empty:
            print(f"Skipping {csv_path.name}: No data rows found.")
            continue
    except pd.errors.EmptyDataError:
        print(f"Skipping {csv_path.name}: EmptyDataError (file likely corrupted or empty).")
        continue

    has_control_cols = 'clean base' in df_inputpairs.columns
    raw_stndrd_signal = read_data.signal

    for index, row in df_inputpairs.iterrows():

        raw_start                 = helpers.safe_parse(row['raw start idx'], int)
        raw_end                   = helpers.safe_parse(row['raw end idx'], int)
        dampen_width              = helpers.safe_parse(row.get('Dampen width', None), float)
        scale_factor              = helpers.safe_parse(row.get('Scale Factor', None), float)
        noise_idx                 = helpers.safe_parse(row.get('Noise source idx', None), int)
        insert_idx                = helpers.safe_parse(row.get('Insert idx', None), int)
        h_recorded_begin_idx      = helpers.safe_parse(row['H Begin Idx'], int)
        clean_recorded_hmer_len   = helpers.safe_parse(row['Clean H-er Length'], int)
        corrupt_recorded_hmer_len = helpers.safe_parse(row['Corrupt H-er Length'], int)

        clean_signal, corrupt_signal = helpers.get_clean_corrupt_signals(
            raw_stndrd_signal, raw_start, raw_end,
            CONTEXT_PADDING, NOISE_SIZE, dampen_width, scale_factor,
            noise_idx, insert_idx
        )

        clean_input     = torch.tensor(clean_signal,   dtype=model_dtype).view(1, 1, -1).to(model.device)
        corrupted_input = torch.tensor(corrupt_signal, dtype=model_dtype).view(1, 1, -1).to(model.device)

        with model.trace(clean_input):
            clean_output_proxy = model.output.save()
        clean_output = clean_output_proxy.detach()
        del clean_output_proxy

        with model.trace(corrupted_input):
            corrupted_output_proxy = model.output.save()
        corrupted_output = corrupted_output_proxy.detach()
        del corrupted_output_proxy

        string_clean     = bonito_model.decode(clean_output[:, 0, :])
        string_corrupted = bonito_model.decode(corrupted_output[:, 0, :])
        print(f"Clean string: {string_clean}")
        print(f"Corrupted string: {string_corrupted}")

        max_time_idx           = clean_output.shape[0]
        score_window_start_idx = 0
        score_window_end_idx   = max_time_idx
        timestamps_to_score    = range(score_window_start_idx, score_window_end_idx)

        TRANSFORMER_BUFFER = 6
        PATCHING_SWEEP_WINDOW_START_IDX = max(0,
            helpers.output_to_transformer_idx(score_window_start_idx) - TRANSFORMER_BUFFER)
        PATCHING_SWEEP_WINDOW_END_IDX = min(
            helpers.time_to_transformer_idx(clean_input.shape[-1]) - 1,
            helpers.output_to_transformer_idx(score_window_end_idx) + TRANSFORMER_BUFFER)

        ##############################
        ###### PATCHING SWEEP ########
        ##############################

        # Denoising
        if not check_if_run_exists(file_name, current_read_idx, index, "layer_denoising", timestamps_to_score):
            layer_recovery = run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score,
                PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="layer", metric_mode="recovery")
            plot_and_save_outputs(layer_recovery, component="layer_denoising", folder_name=current_read_idx, index=index, plot=True)
        else:
            print(f"Skipping layer_denoising for Read {current_read_idx} Row {index}: Files already exist")

        if not check_if_run_exists(file_name, current_read_idx, index, "mlp_denoising", timestamps_to_score):
            mlp_recovery = run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score,
                PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="mlp", metric_mode="recovery")
            plot_and_save_outputs(mlp_recovery, component="mlp_denoising", folder_name=current_read_idx, index=index, plot=True)
        else:
            print(f"Skipping mlp_denoising for Read {current_read_idx} Row {index}: Files already exist")

        if not check_if_run_exists(file_name, current_read_idx, index, "attn_denoising", timestamps_to_score):
            attn_recovery = run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score,
                PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="attn", metric_mode="recovery")
            plot_and_save_outputs(attn_recovery, component="attn_denoising", folder_name=current_read_idx, index=index, plot=True)
        else:
            print(f"Skipping attn_denoising for Read {current_read_idx} Row {index}: Files already exist")

        for h in range(NUM_HEADS):
            if not check_if_run_exists(file_name, current_read_idx, index, f"head {h} denoising", timestamps_to_score):
                head_recovery = run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score,
                    PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="head", head_idx=h, metric_mode="recovery")
                plot_and_save_outputs(head_recovery, component=f"head {h} denoising", folder_name=current_read_idx, index=index)
            else:
                print(f"Skipping head {h} denoising for Read {current_read_idx} Row {index}: Files already exist")

        # Noising
        if not check_if_run_exists(file_name, current_read_idx, index, "layer_noising", timestamps_to_score):
            layer_degradation = run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score,
                PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="layer", metric_mode="degradation")
            plot_and_save_outputs(layer_degradation, component="layer_noising", folder_name=current_read_idx, index=index)
        else:
            print(f"Skipping layer_noising for Read {current_read_idx} Row {index}: Files already exist")

        if not check_if_run_exists(file_name, current_read_idx, index, "mlp_noising", timestamps_to_score):
            mlp_degradation = run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score,
                PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="mlp", metric_mode="degradation")
            plot_and_save_outputs(mlp_degradation, component="mlp_noising", folder_name=current_read_idx, index=index)
        else:
            print(f"Skipping mlp_noising for Read {current_read_idx} Row {index}: Files already exist")

        if not check_if_run_exists(file_name, current_read_idx, index, "attn_noising", timestamps_to_score):
            attn_degradation = run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score,
                PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="attn", metric_mode="degradation")
            plot_and_save_outputs(attn_degradation, component="attn_noising", folder_name=current_read_idx, index=index, plot=True)
        else:
            print(f"Skipping attn_noising for Read {current_read_idx} Row {index}: Files already exist")

        for h in range(NUM_HEADS):
            if not check_if_run_exists(file_name, current_read_idx, index, f"head {h} noising", timestamps_to_score):
                head_degradation = run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score,
                    PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="head", head_idx=h, metric_mode="degradation")
                plot_and_save_outputs(head_degradation, component=f"head {h} noising", folder_name=current_read_idx, index=index)
            else:
                print(f"Skipping head {h} noising for Read {current_read_idx} Row {index}: Files already exist")

        gc.collect()
        torch.cuda.empty_cache()

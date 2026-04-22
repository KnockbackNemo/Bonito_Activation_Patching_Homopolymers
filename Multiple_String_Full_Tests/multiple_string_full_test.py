# The goal of this file is to run a test but with more metrics - and more inputs. Ideally, this will loop over multiple inputs in a folder and then print metrics for them.

import torch
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bonito import util
from bonito.reader import Reader, read_chunks
from nnsight import NNsight
from dataclasses import dataclass
from pathlib import Path


##############################
######## MODEL SETUP #########
##############################

# Disabling the compiler because there are issues with mutating 
torch._dynamo.disable() 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")


model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0" ### The specific model being used!

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu") 

model = NNsight(bonito_model._orig_mod) # Use unoptimized model to avoid conflicts with dynamo
model_dtype = next(model.parameters()).dtype #torch.float16


##############################
########  FILE SETUP #########
##############################
file_name, extension = os.path.splitext(__file__)


##############################
########## FUNCTIONS #########
##############################

def run_patching_sweep(model, source_input, target_input, check_output_timestamps_list, patch_start, patch_end, 
                       num_layers, component="mlp", head_idx=None, metric_mode="recovery"):
    """
    component options: "mlp", "attn", "head"
    """

    # Each row is a patch (differ in layer #, component type, & timestep)
    results = []
    
    ######## Get baseline clean and corrupt outputs for comparison #######
    with model.trace(source_input):
        source_output_proxy = model.output.save()
    source_output = source_output_proxy.detach()

    with model.trace(target_input):
        target_output_proxy = model.output.save()
    target_output = target_output_proxy.detach()

    ## Calculate some stats
    global_mse_baseline_diff = torch.nn.functional.mse_loss(source_output.to(torch.float32), target_output.to(torch.float32)).item()

    ####### Run the actual patching ########################
    for layer_idx in range(num_layers):
        layer = model.encoder.transformer_encoder[layer_idx]

        ### Get source activations ###
        with model.trace(source_input):
            if component == "mlp":
                src_act_proxy = layer.ff.fc2.output.save()
            elif component == "attn":
                src_act_proxy = layer.self_attn.output[0].save()
            elif component == "head":
                src_act_proxy = layer.self_attn.out_proj.input[0].save()

        src_act = src_act_proxy.detach()
        del src_act_proxy

        ### Patch target ###
        for time_offset, t in enumerate(range(patch_start, patch_end)):
            with model.trace(target_input):
                # Calculate safe slice bounds for all components
                start = max(0, t - 1)
                
                # Need sequence length for the upper bound
                seq_len, d_model=src_act.shape[-2:]
                assert seq_len != 1 and d_model != 1
                
                end = min(seq_len, t + 2)

                if component == "mlp":
                    target = layer.ff.fc2.output.clone() # Shape: [1, seq_len, 512]
                    assert target.numel() == seq_len * d_model
                    # target[0, t-1:t+2, :] = src_act[0, t-1:t+2, :]
                    target[0, start:end, :] = src_act[0, start:end, :]
                    layer.ff.fc2.output = target

                elif component == "attn":
                    target = layer.self_attn.output[0].clone() # Shape: [seq_len, 512]
                    assert target.numel() == seq_len * d_model
                    # target[t-1:t+2, :] = src_act[t-1:t+2, :]
                    target[start:end, :] = src_act[start:end, :]
                    layer.self_attn.output[0] = target
                
                elif component == "head":
                    target = layer.self_attn.out_proj.input[0] # Shape: [seq_len, 512]
                    assert target.numel() == seq_len * d_model
                    # seq_len, d_model = target_act.shape
                    # 8 heads -> 64 = size of each head
                    nhead, head_dim = 8, 64
                    target_reshaped = target.clone().reshape(-1, nhead, head_dim)
                    src_reshaped = src_act.reshape(-1, nhead, head_dim)
                    
                    target_reshaped[start:end, head_idx, :] = src_reshaped[start:end, head_idx, :]

                    patched = target_reshaped.reshape(seq_len, d_model)
                    
                    # Reshape and put back into activations
                    layer.self_attn.out_proj.input = patched 

                patched_scores_proxy = model.output.save()

            patched_scores = patched_scores_proxy.detach()

            ####### Calculate metrics ##########
            for timestep in check_output_timestamps_list:
                target_transition_idx = torch.argmax(source_output[timestep, 0, :]).item() # Note that "target" now means correct instead of "corrupted string"
                wrong_transition_idx = torch.argmax(target_output[timestep, 0, :]).item() 
                logit_diff_source = source_output[timestep,0,target_transition_idx] - source_output[timestep,0,wrong_transition_idx]
                logit_diff_target = target_output[timestep,0,target_transition_idx] - target_output[timestep,0,wrong_transition_idx]
                baseline_logit_diff = logit_diff_source - logit_diff_target


                ### Get additional metrics ###
                # Logits and strings
                target_logit = patched_scores[timestep, 0, target_transition_idx].item()
                wrong_logit = patched_scores[timestep, 0, wrong_transition_idx].item()
                actual_max_pred_idx = torch.argmax(patched_scores[timestep, 0, :]).item()
                target_string = transition_idx_to_str(target_transition_idx)
                wrong_string = transition_idx_to_str(wrong_transition_idx)
                actual_string = transition_idx_to_str(actual_max_pred_idx)

                # Posteriors and strings - this includes global most likely path calculations #TODO There is definitely a cleaner way to do this
                source_posteriors = bonito_model.seqdist.posteriors(source_output.to(torch.float32)) + 1e-8
                target_posteriors = bonito_model.seqdist.posteriors(target_output.to(torch.float32)) + 1e-8
                patched_posteriors = bonito_model.seqdist.posteriors(patched_scores.to(torch.float32)) + 1e-8
                posteriors_target_transition_idx = torch.argmax(source_posteriors[timestep, 0, :]).item()
                posteriors_wrong_transition_idx = torch.argmax(target_posteriors[timestep, 0, :]).item()
                posteriors_actual_transition_idx = torch.argmax(patched_posteriors[timestep, 0, :]).item()
                target_state_prob = patched_posteriors[timestep, 0, posteriors_target_transition_idx]
                wrong_state_prob = patched_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                actual_state_prob = patched_posteriors[timestep, 0, posteriors_actual_transition_idx]
                posteriors_target_string = transition_idx_to_str(posteriors_target_transition_idx)
                posteriors_wrong_string = transition_idx_to_str(posteriors_wrong_transition_idx)
                posteriors_actual_string = transition_idx_to_str(posteriors_actual_transition_idx) 

                   
                
                # Logit calculations
                patched_logit_diff = patched_scores[timestep, 0, target_transition_idx] - patched_scores[timestep, 0, wrong_transition_idx]

                if baseline_logit_diff == 0:
                    logit_score = 0
                elif metric_mode == "recovery":
                    logit_score = 1 - (logit_diff_source - patched_logit_diff) / baseline_logit_diff # 1 means full recovery, 0 means no change
                    logit_score = logit_score.item()
                elif metric_mode == "degradation":
                    logit_score = (logit_diff_source - patched_logit_diff) / baseline_logit_diff # 1 means full degradation, 0 means no change
                    logit_score = logit_score.item()
             

                # Probability calculations (posteriors)
                
                patched_prob_diff = target_state_prob - wrong_state_prob
                prob_diff_source = source_posteriors[timestep, 0, posteriors_target_transition_idx] - source_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                prob_diff_target = target_posteriors[timestep, 0, posteriors_target_transition_idx] - target_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                baseline_prob_diff = prob_diff_source - prob_diff_target 
                
                if baseline_prob_diff == 0:
                    posteriors_score = 0
                elif metric_mode == "recovery":
                    posteriors_score = 1 - (prob_diff_source - patched_prob_diff) / baseline_prob_diff # 1 means full recovery, 0 means no change
                    posteriors_score = posteriors_score.item()
                elif metric_mode == "degradation":
                    posteriors_score = (prob_diff_source - patched_prob_diff) / baseline_prob_diff # 1 means full degradation, 0 means no change
                    posteriors_score = posteriors_score.item()


                # If the MSE is greater than clean vs corrupt or the target prediction for this particular timestep isn't correct, decode and save the string
                is_target_logit_argmax = actual_max_pred_idx == target_transition_idx
                is_target_prob_argmax = posteriors_actual_transition_idx == posteriors_target_transition_idx
                global_mse_diff = torch.nn.functional.mse_loss(source_output.to(torch.float32), patched_scores.to(torch.float32)).item()
                should_decode = not is_target_logit_argmax or not is_target_prob_argmax or (global_mse_diff > global_mse_baseline_diff)

                string = None
                if should_decode:
                    string = bonito_model.decode(patched_scores[:, 0, :])
                
                #### Save everything to a dictionary
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
                    "New_String": string
                })

            del patched_scores_proxy, patched_scores

    return pd.DataFrame(results)

    
# Plot results
def plot_and_save_outputs(df, component="mlp", folder_name="default", index="none"):
    
        # Plot and save each timestep individually
        for step in df['Target_Timestep'].unique():

            step_df = df[df['Target_Timestep'] == step]

            folder_path = f"patch_results/{file_name}/read_{folder_name}/row_{index}"
            
            csv_filename = (f"R{folder_name}r{index}_{component}_data_step_{step}.csv")
            csv_full_path = os.path.join(folder_path, csv_filename) 

            os.makedirs(folder_path, exist_ok=True)

            # Logit heatmaps
            png_filename = (f"R{folder_name}r{index}_{component}_logit_heatmap_results_stp_{step}.png")
            png_full_path = os.path.join(folder_path, png_filename) 
            heatmap_matrix = step_df.pivot(index="Layer", columns="Time_Offset", values="Logit_Score")
            plt.figure()
            sns.heatmap(heatmap_matrix)
            plt.title(f"{component} logit heatmap results timestep {step}")
            plt.xlabel(f"Time ticks")
            plt.ylabel(f"Layer")
            plt.savefig(png_full_path)
            plt.close()

            # Probabilitiy heatmaps
            png_filename = (f"{component}_posteriors_heatmap_results_stp_{step}.png")
            png_full_path = os.path.join(folder_path, png_filename) 
            heatmap_matrix = step_df.pivot(index="Layer", columns="Time_Offset", values="Posteriors_Score")
            plt.figure()
            sns.heatmap(heatmap_matrix)
            plt.title(f"{component} posteriors heatmap results timestep {step}")
            plt.xlabel(f"Time ticks")
            plt.ylabel(f"Layer")
            plt.savefig(png_full_path)
            plt.close()
          
            step_df.to_csv(csv_full_path, index=False)

def transition_idx_to_str(transition_idx: int):
    alphabet = model.seqdist.alphabet
    n_states = len(alphabet)
    len_window = model.seqdist.state_len
    n_base = n_states - 1

    # Get the transition (which includes blanks)
    next_base = transition_idx % n_states
    new_idx = transition_idx // n_states

    # Work backwards to get the most to least recent base in the transition
    past_bases_rev = alphabet[next_base]
    # Use mod to get each state (decoded) and append to a string
    for i in range(len_window):
        prev_base = alphabet[(new_idx % n_base) + 1] # No blanks in the context
        new_idx = new_idx // n_base

        past_bases_rev += prev_base

    past_bases = past_bases_rev[::-1]

    return past_bases


def get_clean_corrupt_signals(raw_standrd_signal, raw_start, raw_end, context_padding, noise_size, dampen_width_percent=None, scale_factor=None, noise_idx=None, insert_idx=None):
    
    CONTEXT_PADDING = context_padding
    NOISE_SIZE = noise_size
    
    chunk_start = max(0, raw_start - CONTEXT_PADDING) # Beginning or 0 if too short
    chunk_end = min(len(raw_standrd_signal), raw_end + CONTEXT_PADDING)

    clean_chunk = raw_standrd_signal[chunk_start:chunk_end].copy()

    rel_homo_start = raw_start - chunk_start
    rel_homo_end = raw_end - chunk_start
    homo_len = rel_homo_end - rel_homo_start # In raw input ticks
    midpoint = (rel_homo_start + rel_homo_end) // 2

    corrupt_chunk = clean_chunk.copy()
    
    assert not np.isnan([raw_start, chunk_start, homo_len, dampen_width_percent]).any()

    # If the dampen_width_percent/scale factor are used, apply them to the chunk
    if (dampen_width_percent is not None and not np.isnan(dampen_width_percent) and 
        scale_factor is not None and not np.isnan(scale_factor)):
        dampen_radius = int((homo_len / 2) * dampen_width_percent)
        local_mean = np.mean(corrupt_chunk[midpoint - dampen_radius : midpoint + dampen_radius])
        corrupt_chunk[midpoint - dampen_radius : midpoint + dampen_radius] = ( 
            (corrupt_chunk[midpoint - dampen_radius : midpoint + dampen_radius] * scale_factor) \
            + (local_mean * (1-scale_factor)))
    

    # If the noise offset and source look valid, apply those
    if (insert_idx is not None and not np.isnan(insert_idx) and
        noise_idx is not None and not np.isnan(noise_idx)):
        corrupt_chunk[insert_idx : insert_idx + NOISE_SIZE] = corrupt_chunk[noise_idx : noise_idx + noise_size]

    return clean_chunk, corrupt_chunk

def time_to_output_idx(x) -> int:
    return x // 6 # CNN has stride of 3, 2, and 2, linear upsample has scale factor of 2

def time_to_transformer_idx(x) -> int:
    return x // 12 # CNN has stride of 3, 2, and 2

def output_to_transformer_idx(x) -> int:
    return x // 2 # CNN has stride of 3, 2, and 2


##############################
######## DATA CREATION ####### #TODO: We will want to load in reads generated from somewhere else instead probably
##############################

### NOTE: You must manually filter and add the output tick indices for 'Score window start idx' and 'Score window end idx'
# in the CSVs that you want to read

# Load in data
data_dir = "../data/reads/"

reader = Reader(data_dir)

reads = reader.get_reads(
    data_dir, 
    do_trim=True,
    scaling_strategy=model.config.get("scaling"),
    norm_params=model.config.get("standardisation")
)

CONTEXT_PADDING = 200 # Hardcoded value from other file #TODO make more modular and independent
NOISE_SIZE = 6

for NUM_READ in range(1, 2):

    # Grab the very first read
    first_read = next(reads)
    raw_stndrd_signal = first_read.signal

    # Get the corresponding csv
    csv_path = (f"./Inputs_Tagged/Input_gen_indel_results_read_{NUM_READ}.csv") ## Change this to match the name
    df_inputpairs = pd.read_csv(csv_path)

    for index, row in df_inputpairs.iterrows():
        try:
            raw_start = int(row['raw start idx'])
            raw_end = int(row['raw end idx'])
            noise_idx = int(row.get('Noise idx', default=None))
            insert_idx = int(row.get('Insert idx', default=None))
            dampen_width = float(row.get('Dampen width', default=None))
            scale_factor = float(row.get('Scale Factor', default=None))
            score_window_start_idx = int(row['Score window start idx'])
            score_window_end_idx = int(row['Score window end idx'])
        
        except (ValueError, KeyError): # Some of the files still have brackets in strings
            base = row.loc['base']
            raw_start = int(row.loc['raw start idx'].strip('[]'))
            raw_end = int(row.loc['raw end idx'].strip('[]'))
            num_bases = int(row.loc['num_bases'].strip('[]'))
            noise_idx = int(row.get('Noise idx', default=None).strip('[]'))
            insert_idx = int(row.get('Insert idx', default=None).strip('[]'))
            dampen_width = float(row.get('Dampen width', default=None).strip('[]'))
            scale_factor = float(row.get('Scale Factor', default=None).strip('[]'))
            score_window_start_idx = int(row.loc['Score window start idx'].strip('[]'))
            score_window_end_idx = int(row.loc['Score window end idx'].strip('[]'))

    
        clean_signal, corrupt_signal = get_clean_corrupt_signals(raw_stndrd_signal, raw_start, raw_end, 
                                                                 CONTEXT_PADDING, NOISE_SIZE, dampen_width, scale_factor, 
                                                                 noise_idx, insert_idx)


        # Move signal to tensors and GPU
        clean_input = torch.tensor(clean_signal, dtype=model_dtype).view(1, 1, -1).to(model.device)
        corrupted_input = torch.tensor(corrupt_signal, dtype=model_dtype).view(1, 1, -1).to(model.device)
                
        # Save clean and corrupted outputs
        with model.trace(clean_input):
            clean_output_proxy = model.output.save()
        clean_output = clean_output_proxy.detach()
        del clean_output_proxy

        with model.trace(corrupted_input):
            corrupted_output_proxy = model.output.save()
        corrupted_output = corrupted_output_proxy.detach()
        del corrupted_output_proxy


        string_clean = bonito_model.decode(clean_output[:, 0, :]) # Need to convert to a numpy array in memory
        string_corrupted = bonito_model.decode(corrupted_output[:, 0, :]) 

        print(f"Clean string: {string_clean}")
        print(f"Corrupted string: {string_corrupted}")


        # Run sweep of all layers using a timestep of 1
        NUM_T_LAYERS = 18
        TRANSFORMER_SCORE_WINDOW_PATCH_BUFFER = 6 # Patch the area surrounding the score window (score window = around where we patched)
        PATCHING_SWEEP_WINDOW_START_IDX = max(0, output_to_transformer_idx(score_window_start_idx) - TRANSFORMER_SCORE_WINDOW_PATCH_BUFFER) 
        PATCHING_SWEEP_WINDOW_END_IDX = min(time_to_transformer_idx(clean_input.shape[-1]), output_to_transformer_idx(score_window_end_idx) + TRANSFORMER_SCORE_WINDOW_PATCH_BUFFER)

        NUM_TIMESTEP_SWEEPS = int((PATCHING_SWEEP_WINDOW_END_IDX - PATCHING_SWEEP_WINDOW_START_IDX) / 1) # TODO Later: divide by timestep patch size


        ##############################
        ###### PATCHING SWEEP ########
        ##############################

        timestamps_to_score = range(score_window_start_idx, score_window_end_idx)
        # Call patching sweep function and get heatmap_data things to plot

        # Denoising
        mlp_recovery = run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score, PATCHING_SWEEP_WINDOW_START_IDX, 
            PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="mlp", metric_mode="recovery")
        attn_recovery = run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score, PATCHING_SWEEP_WINDOW_START_IDX, 
            PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="attn", metric_mode="recovery")

        head_recoveries = []
        NUM_HEADS = 8
        for h in range(NUM_HEADS):
            print(f"Sweeping head {h} denoising...")
            head_recoveries.append(run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score, PATCHING_SWEEP_WINDOW_START_IDX, 
                PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="head", head_idx=h, metric_mode="recovery"))
            plot_and_save_outputs(head_recoveries[h], component=f"head {h} denoising", folder_name=NUM_READ, index=index)

            
        # Noising
        mlp_degradation = run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score, PATCHING_SWEEP_WINDOW_START_IDX, 
            PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="mlp", metric_mode="degradation")
        attn_degradation = run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score, PATCHING_SWEEP_WINDOW_START_IDX, 
            PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="attn", metric_mode="degradation")

        head_degradations = []
        NUM_HEADS = 8
        for h in range(NUM_HEADS):
            print(f"Sweeping head {h} noising...")
            head_degradations.append(run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score, PATCHING_SWEEP_WINDOW_START_IDX, 
                PATCHING_SWEEP_WINDOW_END_IDX, NUM_T_LAYERS, component="head", head_idx=h, metric_mode="degradation"))
            plot_and_save_outputs(head_degradations[h], component=f"head {h} noising", folder_name=NUM_READ, index=index)


        # Plot denoising
        plot_and_save_outputs(mlp_recovery, component="mlp_denoising", folder_name=NUM_READ, index=index)
        plot_and_save_outputs(attn_recovery, component="attn_denoising", folder_name=NUM_READ, index=index)
        ## plot_and_save_outputs(head_recoveries, component="head_denoising")

        # Plot noising
        plot_and_save_outputs(mlp_degradation, component="mlp_noising", folder_name=NUM_READ, index=index)
        plot_and_save_outputs(attn_degradation, component="attn_noising", folder_name=NUM_READ, index=index)
        ### plot_and_save_outputs(head_degradations, component="head_noising")

        # (End of row loop)
        gc.collect()
        torch.cuda.empty_cache()


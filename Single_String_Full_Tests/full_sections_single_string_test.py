# The goal of this file is to run a test but with more metrics - and more inputs. Ideally, this will loop over multiple inputs in a folder and then print metrics for them.

import torch
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bonito import util
# from bonito import util
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

@dataclass
class check_output_timestamps_obj:
    timestamp: int
    target_transition: any #TODO not sure what type this is
    wrong_transition: any

def run_patching_sweep(model, source_input, target_input, check_output_timestamps_list, 
                       component="mlp", head_idx=None, metric_mode="recovery", global_baseline_diff=None):
    """
    component options: "mlp", "attn", "head"
    """

    # Each row is a patch (differ in layer #, component type, & timestep)
    results = []

    # heatmap_data = [np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS)) for _ in range(3)]
    
    ######## Get baseline clean and corrupt outputs for comparison #######
    with model.trace(source_input):
        source_output_proxy = model.output.save()
    source_output = source_output_proxy.detach() #TODO What does proxy and detach do?

    with model.trace(target_input):
        target_output_proxy = model.output.save()
    target_output = target_output_proxy.detach()

    global_mse_baseline_diff = torch.nn.functional.mse_loss(source_output.to(torch.float32), target_output.to(torch.float32)).item()

    ####### Run the actual patching ########################
    for layer_idx in range(NUM_T_LAYERS):
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
        for time_offset, t in enumerate(range(PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX)):
            with model.trace(target_input):
                # Calculate safe slice bounds for all components
                start = max(0, t - 1)
                
                # Need sequence length for the upper bound
                seq_len, d_model=src_act.shape[-2:]
                
                end = min(seq_len, t + 2)

                if component == "mlp":
                    target = layer.ff.fc2.output.clone() # Shape: [1, seq_len, 512]
                    # target[0, t-1:t+2, :] = src_act[0, t-1:t+2, :]
                    target[start:end, :] = src_act[start:end, :]
                    layer.ff.fc2.output = target

                elif component == "attn":
                    target = layer.self_attn.output[0].clone() # Shape: [seq_len, 512]
                    # target[t-1:t+2, :] = src_act[t-1:t+2, :]
                    target[start:end, :] = src_act[start:end, :]
                    layer.self_attn.output[0] = target
                
                elif component == "head":
                    target = layer.self_attn.out_proj.input[0] # Shape: [seq_len, 512]

                    # seq_len, d_model = target_act.shape
                    # 8 heads -> 64 = size of each head
                    nhead, head_dim = 8, 64
                    target_reshaped = target.clone().reshape(-1, nhead, head_dim)
                    src_reshaped = src_act.reshape(-1, nhead, head_dim)
                    
                    # start = max(0, t - 1)
                    # end = min(seq_len, t + 2)

                    target_reshaped[start:end, head_idx, :] = src_reshaped[start:end, head_idx, :]

                    patched = target_reshaped.reshape(seq_len, d_model)
                    
                    # Reshape and put back into activations
                    layer.self_attn.out_proj.input = patched 

                patched_scores_proxy = model.output.save()

            patched_scores = patched_scores_proxy.detach()

            ####### Calculate metrics ##########
            for i in range(len(check_output_timestamps_list)):
                time_idx = check_output_timestamps_list[i].timestamp
                target_idx = check_output_timestamps_list[i].target_transition
                wrong_idx = check_output_timestamps_list[i].wrong_transition

                patched_diff = patched_scores[time_idx, 0, target_idx] - patched_scores[time_idx, 0, wrong_idx]
                if baseline_diffs[i] == 0:
                    score = 0
                if metric_mode == "recovery":
                    score = 1 - (logit_diffs_clean[i] - patched_diff) / baseline_diffs[i]
                    score = score.item()
                elif metric_mode == "degradation":
                    score = (logit_diffs_clean[i] - patched_diff) / baseline_diffs[i]
                    score = score.item()

                # Get additional metrics
                target_logit = patched_scores[time_idx, 0, target_idx].item()
                wrong_logit = patched_scores[time_idx, 0, wrong_idx].item()
                global_mse_diff = torch.nn.functional.mse_loss(source_output.to(torch.float32), patched_scores.to(torch.float32)).item()
                actual_max_pred = torch.argmax(patched_scores[time_idx, 0, :]).item()
                
                # If the MSE is greater than clean vs corrupt or the target prediction for this particular timestep isn't correct, decode and save the string
                is_target_argmax = actual_max_pred == target_idx
                should_decode = is_target_argmax or (global_mse_diff > global_mse_baseline_diff)

                string = None
                if should_decode:
                    string = bonito_model.decode(patched_scores[:, 0, :])
                
                #### Save everything to a dictionary
                results.append({
                    "Layer:": layer_idx,
                    "Time_Offset": time_offset,
                    "Target_Timestep": time_idx,
                    "Component": component if head_idx is None else f"head_{head_idx}",
                    "Metric_Mode": metric_mode,
                    "Score": score,
                    "Target_Logit": target_logit,
                    "Wrong_Logit": wrong_logit,
                    "Actual_Argmax": actual_max_pred,
                    "Is_Target_Argmax": is_target_argmax,
                    "Should_Decode": should_decode,
                    "Global_MSE_Diff": global_mse_diff,
                    "New_String": string
                })

            del patched_scores_proxy, patched_scores

    return pd.DataFrame(results)


    
    
# Plot results
def plot_and_save_outputs(df, component="mlp"):
    
        # Plot and save each timestep individually
        for step in df['Target_Timestep'].unique():

            step_df = df[df['Target_Step'] == step]

            heatmap_matrix = step_df.pivot(index="Layer", columns="Time_Offset", values="Score")

            plt.figure()
            sns.heatmap(heatmap_matrix)
            plt.title(f"{file_name} {component} heatmap_results_stp_{step}")
            plt.xlabel(f"Time ticks")
            plt.ylabel(f"Layer")
            plt.savefig(f"{file_name}_{component}_heatmap_results_stp_{step}.png")
            plt.close()

            folder_path = f"patch_results/{file_name}"
            filename = (f"{component}_data_step_{step}.csv")

            os.makedirs(folder_path, exist_ok=True)

            full_path = os.path.join(folder_path, file_name)           
            step_df.to_csv(full_path, index=False)

##############################
######## DATA CREATION ####### #TODO: We will want to load in reads generated from somewhere else instead probably
##############################

# Load in data
data_dir = "../data/reads/"

reader = Reader(data_dir)

reads = reader.get_reads(
    data_dir, 
    do_trim=True,
    scaling_strategy=model.config.get("scaling"),
    norm_params=model.config.get("standardisation")
)

# Grab the very first read
first_read = next(reads)


# Chop data
start_idx = 54250 # Skip beginning noise
LENGTH = 250
end_idx = start_idx + LENGTH
raw_stndrd_signal = first_read.signal
chopped_signal = raw_stndrd_signal[start_idx:end_idx]

clean_signal = chopped_signal.copy()
def time_to_output_idx(x) -> int:
    return x // 6 # CNN has stride of 3, 2, and 2, linear upsample has scale factor of 2

def time_to_transformer_idx(x) -> int:
    return x // 12 # CNN has stride of 3, 2, and 2

# Make fake signal spike
REPLACE_START = 130
steal_base_len = 6 # Must be even
steal_base_spike_start = 100

part_for_spike = clean_signal[steal_base_spike_start:steal_base_spike_start + steal_base_len]

# Modify data to add a spike

corrupt_signal = clean_signal.copy()
corrupt_signal[REPLACE_START : REPLACE_START + steal_base_len] = part_for_spike


# Move signal to tensors and GPU
clean_input = torch.tensor(clean_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
corrupted_input = torch.tensor(corrupt_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
first_input = torch.tensor(chopped_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)

# Save clean and corrupted outputs
with model.trace(clean_input):
    clean_output_proxy = model.output.save()
clean_output = clean_output_proxy.detach()
del clean_output_proxy

with model.trace(corrupted_input):
    corrupted_output_proxy = model.output.save()
corrupted_output = corrupted_output_proxy.detach()
del corrupted_output_proxy

# Calculate baseline MSE at the time
# The homopolymer we want has Cs starting at 14, 15, 16, 20, and 22 with the next A starting at 24 and ending at 26
# Format of the scores is [timesteps, batch, transitions]
score_window_start_idx = 23 #REPLACE_START - 10 # I think this is the wrong index
score_window_end_idx = 25
INDEX_C = 2
INDEX_BLANK = 0


# Run sweep of all layers using a timestep of 1
NUM_T_LAYERS = 18
PATCHING_SWEEP_WINDOW_START_IDX = 8#time_to_transformer_idx(REPLACE_START - 30) # 30 timestamps before spike
PATCHING_SWEEP_WINDOW_END_IDX = 16#time_to_transformer_idx(LENGTH)#REPLACE_START + steal_base_len + 30) # 30 timestamps after end of spike

NUM_TIMESTEP_SWEEPS = int((PATCHING_SWEEP_WINDOW_END_IDX - PATCHING_SWEEP_WINDOW_START_IDX) / 1) # TODO Later: divide by timestep patch size


# Metrics to look at:
# Correct transition vs wrong transition
# Sum of C states - sum of blank states
# later - Sum of A states?
#       - Sum of blank states?
###### Things because I can't think right now but I 100% need to fix this (hardcode 22-24)
TARGET_TRANSITIONS = [torch.argmax(clean_output[22, 0, :]).item(), torch.argmax(clean_output[23, 0, :]).item(), torch.argmax(clean_output[24, 0, :]).item()]

WRONG_TRANSITIONS = [torch.argmax(corrupted_output[22, 0, :]).item(), torch.argmax(corrupted_output[23, 0, :]).item(), torch.argmax(corrupted_output[24, 0, :]).item()]

# heatmap_data_sum_diff_23 = np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS)) # test all three
# heatmap_data_sum_diff_24 = np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS))
# heatmap_data_sum_diff_22 = np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS))
# heatmap_data_argmax_diffs = [np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS)), np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS)), np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS))]

patching_strings = {}
######################################################################

# Logit difference timestep constants
logit_diffs_clean= []
logit_diffs_corrupt= []
for i in range(0, len(TARGET_TRANSITIONS)):
    logit_diffs_clean.append(clean_output[22 + i, 0, TARGET_TRANSITIONS[i]] - clean_output[22 + i, 0, WRONG_TRANSITIONS[i]])
    logit_diffs_corrupt.append(corrupted_output[22 + i, 0, TARGET_TRANSITIONS[i]] - corrupted_output[22 + i, 0, WRONG_TRANSITIONS[i]])

baseline_diffs = []
for i in range(0, len(TARGET_TRANSITIONS)): # Positive if clean is closer to the target than wrong, negative if corrupt is closer to target than wrong, 0 if same
    baseline_diffs.append(logit_diffs_clean[i].to(torch.float32) - logit_diffs_corrupt[i].to(torch.float32))
print(f"Output shape: {clean_output.shape}")
print(f"baseline diff: {baseline_diffs}")

##############################
###### PATCHING SWEEP ########
##############################

# Call patching sweep function and get heatmap_data things to plot

# Denoising
mlp_recovery = run_patching_sweep(model, clean_input, corrupted_input, component="mlp", metric_mode="recovery")
attn_recovery = run_patching_sweep(model, clean_input, corrupted_input, component="attn", metric_mode="recovery")

head_recoveries = []
NUM_HEADS = 8
for h in range(NUM_HEADS):
    print(f"Sweeping head {h} denoising...")
    head_recoveries.append(run_patching_sweep(model, clean_input, corrupted_input, component="head", head_idx=h, metric_mode="recovery"))
    plot_and_save_outputs(head_recoveries[h], component=f"head {h} denoising")

    
# Noising
mlp_degradation = run_patching_sweep(model, corrupted_input, clean_input, component="mlp", metric_mode="degradation")
attn_degradation = run_patching_sweep(model, corrupted_input, clean_input, component="attn", metric_mode="degradation")

head_degradations = []
NUM_HEADS = 8
for h in range(NUM_HEADS):
    print(f"Sweeping head {h} noising...")
    head_degradations.append(run_patching_sweep(model, corrupted_input, clean_input, component="head", head_idx=h, metric_mode="degradation"))
    plot_and_save_outputs(head_degradations[h], component=f"head {h} noising")


# Plot denoising
plot_and_save_outputs(mlp_recovery, component="mlp_denoising")
plot_and_save_outputs(attn_recovery, component="attn_denoising")
## plot_and_save_outputs(head_recoveries, component="head_denoising")

# Plot noising
plot_and_save_outputs(mlp_degradation, component="mlp_noising")
plot_and_save_outputs(attn_degradation, component="attn_noising")
### plot_and_save_outputs(head_degradations, component="head_noising")


string_clean = bonito_model.decode(clean_output[:, 0, :]) # Need to convert to a numpy array in memory
string_corrupted = bonito_model.decode(corrupted_output[:, 0, :]) 
# string_patched = bonito_model.decode(patched_output[:, 0, :]) 

print(f"Clean string: {string_clean}")
print(f"Corrupted string: {string_corrupted}")


# with open(f"{file_name}_strings.txt", "w") as f:
#     print(f"Clean output string: {string_clean}", file=f)
#     print(f"Corrupt output strn: {string_corrupted}", file=f)
#     print()

#     # Loop through heatmap and print each string
#     for layer_idx in range(NUM_T_LAYERS):
#         print(f"--- Patched Strings for Layer {layer_idx} ---", file=f)
#         for time_offset, t in enumerate(range(SWEEP_WINDOW_START, SWEEP_WINDOW_END)):
#             print(f"Time {time_offset} : {patching_strings[layer_idx, time_offset]} : scores: {heatmap_data_argmax_diffs[0][layer_idx, time_offset]}, {heatmap_data_argmax_diffs[1][layer_idx, time_offset]}, {heatmap_data_argmax_diffs[2][layer_idx, time_offset]}", file=f)


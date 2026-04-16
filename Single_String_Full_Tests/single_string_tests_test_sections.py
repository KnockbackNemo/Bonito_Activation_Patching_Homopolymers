# The goal of this file is to run a test but with more metrics - and more inputs. Ideally, this will loop over multiple inputs in a folder and then print metrics for them.

import torch
import os
import gc
import pod5
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bonito import util
from bonito import util
from bonito.reader import Reader, read_chunks
from nnsight import NNsight


##############################
######## MODEL SETUP #########
##############################

# Disabling the compiler because there are issues with mutating 
torch._dynamo.disable() 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")


model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu") 

model = NNsight(bonito_model._orig_mod) # Use unoptimized model to avoid conflicts with dynamo
model_dtype = next(model.parameters()).dtype #torch.float16



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

logit_diff_clean = clean_output[score_window_start_idx, 0, INDEX_C] - clean_output[score_window_start_idx, 0, INDEX_BLANK]
logit_diff_corrupt = corrupted_output[score_window_start_idx, 0, INDEX_C] - corrupted_output[score_window_start_idx, 0, INDEX_BLANK]

baseline_diff = logit_diff_clean.to(torch.float32) - logit_diff_corrupt.to(torch.float32)
print(f"Output shape: {clean_output.shape()}")
print(f"baseline diff: {baseline_diff}")

##############################
###### PATCHING SWEEP ########
##############################

# Run sweep of all layers using a timestep of 1
NUM_T_LAYERS = 18
SWEEP_WINDOW_START = 0#time_to_transformer_idx(REPLACE_START - 30) # 30 timestamps before spike
SWEEP_WINDOW_END = time_to_transformer_idx(LENGTH)#REPLACE_START + steal_base_len + 30) # 30 timestamps after end of spike

NUM_TIMESTEP_SWEEPS = int((SWEEP_WINDOW_END - SWEEP_WINDOW_START) / 1) # TODO Later: divide by timestep patch size
heatmap_data = np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS))
patching_strings = {}

# Logit difference timestep constants

for layer_idx in range(NUM_T_LAYERS):
    print(f"Sweeping layer {layer_idx}")

    with model.trace(clean_input):
        clean_activation_proxy = model.encoder.transformer_encoder[layer_idx].output[0].save()
    clean_activation = clean_activation_proxy.detach()
    del clean_activation_proxy

    for time_offset, t in enumerate(range(SWEEP_WINDOW_START, SWEEP_WINDOW_END)):
        print(f"Sweeping transformer timestep {t} : offset {time_offset} -> output {2*t}")

        with model.trace(corrupted_input):
            # print(f"Shape: {clean_activation.shape}")
            model.encoder.transformer_encoder[layer_idx].output[0][t-1:t+2, :] = clean_activation[t-1:t+2, :]
            patched_scores_proxy = model.output.save()

        patched_scores = patched_scores_proxy.detach()

        # This is higher if C is more likely
        patched_logit_diff = patched_scores[score_window_start_idx, 0, INDEX_C] - patched_scores[score_window_start_idx, 0, INDEX_BLANK]
        patched_diff = logit_diff_clean.to(torch.float32) - patched_logit_diff.to(torch.float32)

        recovery_score = 1.0 - (patched_diff / baseline_diff)

        heatmap_data[layer_idx, time_offset] = recovery_score
        patching_strings[(layer_idx, time_offset)] = bonito_model.decode(patched_scores[:, 0, :].to(torch.float32) )

        del patched_scores_proxy
        del patched_scores
        
    del clean_activation
    gc.collect() 
    torch.cuda.empty_cache()


# Write data to a file
file_name, extension = os.path.splitext(__file__)

# Plot results
plt.figure()
sns.heatmap(heatmap_data)
plt.savefig(f"{file_name}_heatmap_results.png")
np.save(f"{file_name}_heatmap_data.npy", heatmap_data)


string_clean = bonito_model.decode(clean_output[:, 0, :]) # Need to convert to a numpy array in memory
string_corrupted = bonito_model.decode(corrupted_output[:, 0, :]) 
# string_patched = bonito_model.decode(patched_output[:, 0, :]) 


with open(f"{file_name}_strings.txt", "w") as f:
    print(f"Clean output string: {string_clean}", file=f)
    print(f"Corrupt output strn: {string_corrupted}", file=f)
    print()

    # Loop through heatmap and print each string
    for layer_idx in range(NUM_T_LAYERS):
        print(f"--- Patched Strings for Layer {layer_idx} ---", file=f)
        for time_offset, t in enumerate(range(SWEEP_WINDOW_START, SWEEP_WINDOW_END)):
            print(f"Time {time_offset} : {patching_strings[layer_idx, time_offset]} : score: {heatmap_data[layer_idx, time_offset]}", file=f)


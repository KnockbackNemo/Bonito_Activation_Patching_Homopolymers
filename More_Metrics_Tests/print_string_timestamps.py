# This file takes in a read and prints out the timestamps for each base

import os
import pod5
import torch
from bonito import util
from bonito.reader import Reader, read_chunks

##############################
######## MODEL SETUP #########
##############################

# Disabling the compiler because there are issues with mutating 
torch._dynamo.disable() 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")


model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0" # Need to find model path

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

model = bonito_model._orig_mod # Use unoptimized model to avoid conflicts with dynamo
model_dtype = next(model.parameters()).dtype #torch.float16
#print(model_dtype)


##############################
######## DATA LOADING ########
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
start_idx = 54250 # Skip beginning noise 53000
LENGTH = 250 # Somewhere in here is a nice C 5-mer string
end_idx = start_idx + LENGTH
raw_stndrd_signal = first_read.signal
chopped_signal = raw_stndrd_signal[start_idx:end_idx]

input = torch.tensor(chopped_signal, dtype=model_dtype).view(1, 1, LENGTH).to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    output = bonito_model(input)


v_path = model.seqdist.viterbi(output.to(dtype=torch.float32)) #  Only float32 supported

alphabet = {1: 'A', 2: 'C', 3: 'G', 4: 'T'}

string_clean = bonito_model.decode(output[:, 0, :].to(dtype=torch.float32)) # Need to convert to a numpy array in memory

print("--- Full string ---")
print(string_clean)
print("--- Sequence timestamps ---")

current_state = v_path[0]
start_t = 0

idx = 0

for t, state in enumerate(v_path):
    current_state = state.item()
    if current_state != 0: # skip blanks
        base = alphabet.get(current_state, f"Unknown({current_state})") # Get state or print unknown if unsuccessful
        orig_char = string_clean[idx] if idx < len(string_clean) else "END"
        print(f"Timesteps {t:3d} : {base} : Orig : {orig_char}")
        idx = idx + 1
  


if current_state != 0:
    base = alphabet.get(current_state, f"Unknown({current_state})")
    print(f"Timesteps {start_t:3d} to {len(v_path) - 1:3d} : {base}")


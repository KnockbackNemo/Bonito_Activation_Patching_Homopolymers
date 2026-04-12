# The goal of this file is to look for the place where a spike in the signal is found
# as a proof of concept


import torch
import os
import pod5
import matplotlib.pyplot as plt
import numpy as np
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

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu") # Will eventually run this on LRC machines- may have gpus?

model = NNsight(bonito_model._orig_mod) # Use unoptimized model to avoid conflicts with dynamo
model_dtype = next(model.parameters()).dtype #torch.float16
#print(model_dtype)


##############################
######## DATA CREATION #######
##############################


# Load in data
data_dir = "./data/reads/"

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


# Make fake signal spike
# SPIKE_LEN = 8 # Even
# SPIKE_MIDPOINT = 240
# SPIKE_SURROUND_WIDTH = 50
# NUM_TO_REPLACE = 0
REPLACE_START = 130
# steal_base_start = 150 
steal_base_len = 6 # Must be even
steal_base_spike_start = 100

# part_to_copy = clean_signal[steal_base_start:steal_base_start + steal_base_len]
# part_to_copy = part_to_copy - 0.75*part_to_copy + part_to_copy.mean()
part_for_spike = clean_signal[steal_base_spike_start:steal_base_spike_start + steal_base_len]

# Flatten the data around the spike area to make it look like a plateau


# Add spike
# spike = np.linspace(-3, 3, SPIKE_LEN)
# spike = -np.exp(-0.5 * spike**2) * .50
# corrupt_signal[SPIKE_MIDPOINT - int(SPIKE_LEN/2) : SPIKE_MIDPOINT + int(SPIKE_LEN/2)] += spike
# corrupt_signal[SPIKE_MIDPOINT - int(steal_base_len/2): SPIKE_MIDPOINT + int(steal_base_len/2)] = clean_signal[steal_base_start:steal_base_start+steal_base_len]

# Modify data to add a homopolymer and a spike

# clean_signal[REPLACE_START:REPLACE_START + NUM_TO_REPLACE*steal_base_len] = np.tile(part_to_copy, NUM_TO_REPLACE)
corrupt_signal = clean_signal.copy()
corrupt_signal[REPLACE_START : REPLACE_START + steal_base_len] = part_for_spike


# Move signal to tensors and GPU
clean_input = torch.tensor(clean_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
corrupted_input = torch.tensor(corrupt_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
first_input = torch.tensor(chopped_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)

# Let's see if the output suggests A->C and A->G homopolymers
with model.trace(clean_input):
    clean_activation = model.encoder.transformer_encoder[0].output[0].save()
    clean_output = model.output.save()

with model.trace(corrupted_input):
    corrupted_output = model.output.save()

with model.trace(corrupted_input):
    model.encoder.transformer_encoder[0].output[0] = clean_activation
    patched_output = model.output.save()


with model.trace(first_input):
    first_output = model.output.save()


file_name, extension = os.path.splitext(__file__)



string_clean = bonito_model.decode(clean_output[:, 0, :]) # Need to convert to a numpy array in memory
string_corrupted = bonito_model.decode(corrupted_output[:, 0, :]) 
string_patched = bonito_model.decode(patched_output[:, 0, :]) 
string_orig = bonito_model.decode(first_output[:, 0, :]) 



with open(f"{file_name}_strings.txt", "w") as f:
    print(f"Clean output string: {string_clean}", file=f)
    print(f"Corrupt output strn: {string_corrupted}", file=f)
    print(f"Patched output strn: {string_patched}", file=f)
    print(f"Origin output strng: {string_orig}", file=f)

# Check signal
plt.figure()
plt.plot(chopped_signal, label="Chopped signal", color='green')
plt.plot(clean_signal, label="Clean signal", color='blue')
plt.plot(corrupt_signal, label="Corrupted signal", color='red')
plt.show()

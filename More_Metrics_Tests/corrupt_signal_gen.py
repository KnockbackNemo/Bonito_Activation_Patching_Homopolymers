# This script reads in a signal and produces a corrupted signal
# Plots both signals and the string so we can see it, and
# Saves the data as an array so other scripts can use them.

# This file takes in a read and prints out the timestamps for each base

import os
import torch
import matplotlib as plt
import pandas as pd
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

NUM_READ = 8 # Change this to get a different read in the POD5 file

for i in range(1,NUM_READ):
    first_read = next(reads)

raw_stndrd_signal = first_read.signal

# Chop data 
# ##############
# # THIS IS WHERE THE FOUND KMER PARAMS COME IN
# ##############
RAW_START =
INPUT_LENGTH = 
INPUT_BEGIN_PADDING_LENGTH = 80
INPUT_END_PADDING_LENGTH = 40
EXTEND = 1 # 1 for extend by 1, 0 for shorten by 1
SECOND_TO_LAST_HMER_TOK_START = 
LAST_HMER_TOK_START =
NEXT_TOK_START = 
print(f"Length: {LENGTH} for read")

# Chop out a window around the desired signal
raw_stndrd_signal = raw_stndrd_signal[RAW_START - INPUT_BEGIN_PADDING_LENGTH:RAW_START + INPUT_LENGTH + INPUT_END_PADDING_LENGTH]
input = torch.tensor(raw_stndrd_signal, dtype=model_dtype).view(1, 1, LENGTH).to("cuda" if torch.cuda.is_available() else "cpu")

# Make a corrupt signal that pastes the token over last homopolymer or extends the second to last homopolymer token out
corrupt_input = input.copy()
if EXTEND: # Turn CGAAA*AAT*TC into CGAAAA*AAT*C #TODO I actually need to know when the next token ends... I think.? Or we say we don't really care if we corrupt the string afterwards and just paste a few ticks in. We can't be all that precise here anyways.
    signal_to_cpy = input[SECOND_TO_LAST_HMER_TOK_START:NEXT_TOK_START + 15] #10-15 is about the length of one base
    corrupt_input[LAST_HMER_TOK_START:LAST_HMER_TOK_START + len(signal_to_cpy)] = signal_to_cpy
else: # Shorten string
    signal_to_cpy = input[LAST_HMER_TOK_START:NEXT_TOK_START + 15] #10-15 is about the length of one base
    corrupt_input[SECOND_TO_LAST_HMER_TOK_START:SECOND_TO_LAST_HMER_TOK_START + len(signal_to_cpy)] = signal_to_cpy

# Check signal
plt.figure()
plt.plot(input, label="Clean signal", color='blue')
plt.plot(corrupt_input, label="Corrupted signal", color='red')
plt.show()

with torch.no_grad():
    output = bonito_model(input)


v_path = model.seqdist.viterbi(output.to(dtype=torch.float32)) #  Only float32 supported

alphabet = {1: 'A', 2: 'C', 3: 'G', 4: 'T'}

''' We want to save
    - The start timestamp
    - The timestamps of each base called in the middle
    - The end timestamp (when the next different base appears)
    - The length (in time)
    - The length (in number of bases) - k-mer k
    - Which base it is
    We'll keep this all in the same read so that standardization isn't a problem (if it ever would be)
'''

def output_to_abs_in_idx(output_idx):
    ''' Returns the timestamp in the original read where the index occured
    '''
    return output_idx * 6 # Output is downsampled by 6

K_MIN = 4 # We'll save homopolymers of 4 or more bases


v_path_list = v_path.flatten().tolist()


for i, state in enumerate(v_path_list):
    if state != 0 and state != len(v_path_list) - 1:
        print(f"{alphabet[state]}", end="")

# Iterate through all bases. For each one, keep track of the last base and slide a window to count the length of the homopolymer    
for t, current_state in enumerate(v_path_list):
    if current_state == 0: # skip blanks
        continue

    if current_state == last_state: # continue counting window
        slice_length = slice_length + 1
        kmer_stamps_input.append(output_to_abs_in_idx(t))
        
    else: # End window counting: reset vars and save output if it meets requirements
        if slice_length >= K_MIN:

            # Assumes last state is valid (not the initial -1)
            base =  alphabet.get(last_state, f"Unknown({current_state})") # Get state or print unknown if unsuccessful
    
            
            # Save slice_timestamp_start, t, length (time), # bases, base, kmer_stamps
            records.append({'base' : [base],
                            'num_bases' : [slice_length],
                            'duration_viterbi' : [(t - slice_timestamp_start)],
                            'raw start idx' : [output_to_abs_in_idx(slice_timestamp_start)],
                            'raw end idx' : [output_to_abs_in_idx(t)], # Start of next base; should be excluded
                            'input_base_timestamps' : list(kmer_stamps_input)
                            })

            print(f"Found {base} {slice_length}-mer at viterbi timestamps {slice_timestamp_start}-{t} ")
        
        # Reset variables
        slice_timestamp_start = t
        kmer_stamps_input = [output_to_abs_in_idx(t)]
        slice_length = 1

        
    
    last_state = current_state
  

# Save data to a csv
df = pd.DataFrame(records)

output_fil_name = 'kmers_data_read_' + str(NUM_READ)
if not df.empty:
    df.to_csv(output_fil_name + '.csv', index=False)
    print(f"saved to {output_fil_name}.csv")
else:
    print("nothing found")
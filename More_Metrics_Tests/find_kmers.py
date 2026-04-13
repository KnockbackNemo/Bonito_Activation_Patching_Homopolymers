# This file takes in a read and prints out the timestamps for each base

import os
import pod5
import torch
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
raw_stndrd_signal = first_read.signal

# Chop data
START_IDX = 54250 # Skip beginning noise 50000
MAX_LENGTH = 250
LENGTH = min(MAX_LENGTH, len(raw_stndrd_signal))

if LENGTH <=0:
    raise ValueError(f"START_IDX is larger than the read leength! Please decrease START_IDX to less than {len(first_read)}")

raw_stndrd_signal = raw_stndrd_signal[START_IDX:START_IDX + LENGTH]

input = torch.tensor(raw_stndrd_signal, dtype=model_dtype).view(1, 1, LENGTH).to("cuda" if torch.cuda.is_available() else "cpu")

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
    return output_idx * 6 + START_IDX # Output is downsampled by 6 and relative to START_IDX

K_MIN = 4 # We'll save homopolymers of 4 or more bases


v_path_list = v_path.flatten().tolist()
v_path_list.append(-1)

last_state = -1
records = [] # Save all data
slice_timestamp_start = 0
slice_length = 0
kmer_stamps_input = [] # Save each homopolymer timestamp


print(v_path)

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

if not df.empty:
    df.to_csv('kmers_data.csv', index=False)
    print(f"saved to kmers_data.csv")
else:
    print("nothing found")
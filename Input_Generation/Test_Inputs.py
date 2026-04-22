# The goal of this file is to look for the place where a spike in the signal is found
# as a proof of concept


import torch
import os
import difflib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def get_string_difference(str1, str2):
    matcher = difflib.SequenceMatcher(None, str1, str2)
    diff_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            diff_count += max(i2 - i1, j2 - j1)
    return diff_count

def is_homopolymer_single_indel(str1, str2):
    matcher = difflib.SequenceMatcher(None, str1, str2)
    diff_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            return False
        elif tag in ('insert', 'delete'):
            diff_count += max(i2 - i1, j2 - j1)
    return diff_count

##############################
######## DATA CREATION #######
##############################

NUM_READ = 4

# Load in data
data_dir = "../data/reads/"
csv_path = (f"../More_Metrics_Tests/kmers_data_read_{NUM_READ}.csv") ## Change this to match the name
df_kmers = pd.read_csv(csv_path)

reader = Reader(data_dir)

reads = reader.get_reads(
    data_dir, 
    do_trim=True,
    scaling_strategy=model.config.get("scaling"),
    norm_params=model.config.get("standardisation")
)

# Grab the very first read
for i in range(0, NUM_READ):
    first_read = next(reads)

raw_stndrd_signal = first_read.signal

CONTEXT_PADDING = 200

inputs = []

for index, row in df_kmers.iterrows():
    # base, raw_start, raw_end, num_bases
    try:
        base = row['base']
        raw_start = int(row['raw start idx'])
        raw_end = int(row['raw end idx'])
        num_bases = int(row['num_bases'])
    except: # Some of the files still have brackets in strings
        base = row.loc['base']
        raw_start = int(row.loc['raw start idx'].strip('[]'))
        raw_end = int(row.loc['raw end idx'].strip('[]'))
        num_bases = int(row.loc['num_bases'].strip('[]'))

    print(f"Processing row {index}: {num_bases}{base} hpolymer at raw idex {raw_start}-{raw_end}")

    # Slice window
    chunk_start = max(0, raw_start - CONTEXT_PADDING)
    chunk_end = min(len(raw_stndrd_signal), raw_end + CONTEXT_PADDING)
    LENGTH = chunk_end - chunk_start

    clean_chunk = raw_stndrd_signal[chunk_start:chunk_end].copy()

    # Corrupt signal
    rel_homo_start = raw_start - chunk_start
    rel_homo_end = raw_end - chunk_start
    homo_len = rel_homo_end - rel_homo_start
    midpoint = (rel_homo_start + rel_homo_end) // 2

    

    dampen_widths_percent = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    scale_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for dampen_width_percent in dampen_widths_percent:
        for scale_factor in scale_factors:

            corrupt_chunk = clean_chunk.copy()
            
            dampen_radius = int((homo_len / 2) * dampen_width_percent)
            local_mean = np.mean(corrupt_chunk[midpoint - dampen_radius : midpoint + dampen_radius])
            corrupt_chunk[midpoint - dampen_radius : midpoint + dampen_radius] = ( 
                (corrupt_chunk[midpoint - dampen_radius : midpoint + dampen_radius] * scale_factor) \
                + (local_mean * (1-scale_factor)))
    



            # Move signal to tensors and GPU
            clean_input = torch.tensor(clean_chunk, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
            corrupted_input = torch.tensor(corrupt_chunk, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)

            # Let's see if the output suggests A->C and A->G homopolymers
            with model.trace(clean_input):
                clean_output_proxy = model.output.save()

            with model.trace(corrupted_input):
                corrupted_output_proxy = model.output.save()

            clean_str = bonito_model.decode(clean_output_proxy.detach()[:, 0, :])
            corrupt_str = bonito_model.decode(corrupted_output_proxy.detach()[:, 0, :])

            del clean_output_proxy, corrupted_output_proxy

            if not is_homopolymer_single_indel(clean_str, corrupt_str):
                print(f"Skipping: corruption with width {dampen_width_percent} and scale {scale_factor} changed more than one base.")
                # print(f"Clean:  {clean_str}")
                # print(f"Corrupt:{corrupt_str}")
                continue
            
            print(f"Possible success with width {dampen_width_percent} and scale {scale_factor}")
            print(f"Clean:  {clean_str}")
            print(f"Corrupt:{corrupt_str}")

            min_strings = ['AAAA', 'CCCC', 'GGGG', 'TTTT']
            for minstr in min_strings:
                if minstr in clean_str and minstr in corrupt_str and (len(clean_str) - len(corrupt_str)):
            
                    print(f"Possible success with noise idx {noise_idx} at relative index {i}")
                    print(f"Clean:  {clean_str}")
                    print(f"Corrupt:{corrupt_str}")

                    inputs.append({
                        **row.to_dict(),
                        "Dampen width": dampen_width_percent,
                        "Scale Factor": scale_factor,
                        "Clean string": clean_str,
                        "Corrupt_string": corrupt_str
                    })

                    break
        

            
            
        

    noise_size = 6
    noise_offset = -50
    # Try adding some noise

    for i in range(rel_homo_start, rel_homo_end - noise_size):
        corrupt_chunk = clean_chunk.copy()

        noise_idx = max(0, i - noise_offset)
        
        corrupt_chunk[i : i + 6] = corrupt_chunk[noise_idx : noise_idx + noise_size]

        # Move signal to tensors and GPU
        clean_input = torch.tensor(clean_chunk, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
        corrupted_input = torch.tensor(corrupt_chunk, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)

        # Let's see if the output suggests A->C and A->G homopolymers
        with model.trace(clean_input):
            clean_output_proxy = model.output.save()

        with model.trace(corrupted_input):
            corrupted_output_proxy = model.output.save()

        clean_str = bonito_model.decode(clean_output_proxy.detach()[:, 0, :])
        corrupt_str = bonito_model.decode(corrupted_output_proxy.detach()[:, 0, :])

        del clean_output_proxy, corrupted_output_proxy

        if not is_homopolymer_single_indel(clean_str, corrupt_str):
            print(f"Skipping: corruption with noise idx {noise_idx} at relative index {i} changed more than one base.")
            # print(f"Clean:  {clean_str}")
            # print(f"Corrupt:{corrupt_str}")
            continue

        min_strings = ['AAAA', 'CCCC', 'GGGG', 'TTTT']
        for minstr in min_strings:
            if minstr in clean_str and minstr in corrupt_str:
        
                print(f"Possible success with noise idx {noise_idx} at relative index {i}")
                print(f"Clean:  {clean_str}")
                print(f"Corrupt:{corrupt_str}")

                inputs.append({
                    **row.to_dict(),
                    "Noise idx": noise_idx,
                    "Insert idx": i,
                    "Clean string": clean_str,
                    "Corrupt_string": corrupt_str
                })

                break
        
    

        
    

df = pd.DataFrame(inputs)
df.to_csv(f"Input_gen_results_read_{NUM_READ}.csv", index=False)
# file_name, extension = os.path.splitext(__file__)



# string_clean = bonito_model.decode(clean_output[:, 0, :]) # Need to convert to a numpy array in memory
# string_clean_viterbi = bonito_model.seqdist.viterbi(clean_output.to(dtype=torch.float32))
# string_corrupted = bonito_model.decode(corrupted_output[:, 0, :]) 
# string_corrupted_viterbi = bonito_model.seqdist.viterbi(corrupted_output.to(dtype=torch.float32))
# string_patched = bonito_model.decode(patched_output[:, 0, :]) 
# string_patched_viterbi = bonito_model.seqdist.viterbi(patched_output.to(dtype=torch.float32))



# def viterbi_to_string(path):
#     alphabet = ['N', 'A', 'C', 'G', 'T']
#     str = ""
#     for state in path:
#         str += alphabet[state.item()]
#     return str
    

# # print(f"Clean output string: {string_clean}", file=f)
# # print(f"Corrupt output strn: {string_corrupted}", file=f)
# # print(f"Patched output strn: {string_patched}", file=f)
# # print(f"Origin output strng: {string_orig}", file=f)
# print(f"string_clean:     {string_clean}")
# print(f"string_corrupted: {string_corrupted}")
# print(f"string_patched:   {string_patched}")
# print(f"string_clean_viterbi: {viterbi_to_string(string_clean_viterbi)}")
# print(f"string_corrupted_viterbi: {viterbi_to_string(string_corrupted_viterbi)}")
# print(f"string_patched_viterbi: {viterbi_to_string(string_patched_viterbi)}")

# # # Check signal
# # plt.figure()
# # plt.plot(chopped_corrupt_signal, label="Chopped signal", color='green')
# # plt.plot(clean_signal, label="Clean signal", color='blue')
# # plt.plot(corrupt_signal, label="Corrupted signal", color='red')
# # plt.show()



# raw_stndrd_signal = first_read.signal
# chopped_clean_signal = raw_stndrd_signal.copy()
# chopped_corrupt_signal = raw_stndrd_signal.copy()
# # chopped_corrupt_signal[58175:58203] = chopped_corrupt_signal[69095:69123]# Umm we will ignore that the previous base might be different
# chopped_corrupt_signal[69094:69124] = chopped_corrupt_signal[58174:58204]
# # chopped_corrupt_signal = chopped_corrupt_signal[58000:58300]
# # chopped_clean_signal = chopped_clean_signal[58000:58300]
# chopped_corrupt_signal = chopped_corrupt_signal[69000:69300]
# chopped_clean_signal = chopped_clean_signal[69000:69300]

# clean_signal = chopped_clean_signal
# corrupt_signal = chopped_corrupt_signal
# LENGTH = 58300 - 58000
# # 1,A,4,10,58164,58224,"[58164, 58188, 58200, 58212]"
# # 1,A,5,10,69084,69144,"[69084, 69090, 69096, 69108, 69132]"
# # Maybe try these two


# # # Try 1,G,5,13,35214,35292,"[35214, 35232, 35268, 35274, 35280]"?
# # # Try swapping into a different plateau from an otherwise similar area?


# # # Chop data
# # start_idx = 35150 # Skip beginning noise
# # LENGTH = 250
# # end_idx = start_idx + LENGTH
# # raw_stndrd_signal = first_read.signal
# # chopped_signal = raw_stndrd_signal[start_idx:end_idx]

# # clean_signal = chopped_signal.copy()


# # # Make fake signal spike
# # # SPIKE_LEN = 8 # Even
# # # SPIKE_MIDPOINT = 240
# # # SPIKE_SURROUND_WIDTH = 50
# # # NUM_TO_REPLACE = 0
# # REPLACE_START = 35295 - start_idx
# # # steal_base_start = 150 
# # steal_base_len = 6 # Must be even
# # steal_base_spike_start = 50

# # # part_to_copy = clean_signal[steal_base_start:steal_base_start + steal_base_len]
# # # part_to_copy = part_to_copy - 0.75*part_to_copy + part_to_copy.mean()
# # part_for_spike = clean_signal[steal_base_spike_start:steal_base_spike_start + steal_base_len]

# # Flatten the data around the spike area to make it look like a plateau


# # Add spike
# # spike = np.linspace(-3, 3, SPIKE_LEN)
# # spike = -np.exp(-0.5 * spike**2) * .50
# # corrupt_signal[SPIKE_MIDPOINT - int(SPIKE_LEN/2) : SPIKE_MIDPOINT + int(SPIKE_LEN/2)] += spike
# # corrupt_signal[SPIKE_MIDPOINT - int(steal_base_len/2): SPIKE_MIDPOINT + int(steal_base_len/2)] = clean_signal[steal_base_start:steal_base_start+steal_base_len]

# # Modify data to add a homopolymer and a spike

# # clean_signal[REPLACE_START:REPLACE_START + NUM_TO_REPLACE*steal_base_len] = np.tile(part_to_copy, NUM_TO_REPLACE)
# # corrupt_signal = clean_signal.copy()
# # corrupt_signal[REPLACE_START : REPLACE_START + steal_base_len] = part_for_spike

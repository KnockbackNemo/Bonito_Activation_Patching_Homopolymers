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


def get_homopolymer_len(string, idx):
    ''' Get the number of repeating characters at the given idx for that string
    and return the length, starting idx, and ending idx '''
    begin = end = idx
    char = string[idx]
    length = 1

    while (begin > 0 and string[begin - 1] == char):
        length = length + 1
        begin = begin - 1
    
    while (end < len(string) - 1) and string[end + 1] == char:
        length = length + 1
        end = end + 1

    return (length, begin, end)


def is_homopolymer_single_indel(str1, str2):
    matcher = difflib.SequenceMatcher(None, str1, str2)
    diff_count = 0
    in_homopolymer = 0
    base_l = ""
    hbegin_idx, str1_hlen, str2_hlen = 0, 0, 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            return (False, 0, 0, "", 0)
        elif tag in ('insert', 'delete'):
            # Check if this happened in a homopolymer
            if tag == 'delete':
                str1_hlen, str1_hbegin, str1_hend = get_homopolymer_len(str1, i1)
                if (str1_hlen >= 5):
                    if str1_hbegin >= len(str2):
                        str2_hlen, str2_hbegin, str2_hend = 0, 0, 0
                    else:
                        str2_hlen, str2_hbegin, str2_hend = get_homopolymer_len(str2, str1_hbegin)
                    
                    in_homopolymer = 1
                    base_l = str1[i1]

            if tag == 'insert':
                str2_hlen, str2_hbegin, str2_hend = get_homopolymer_len(str2, j1) 
                if (str2_hlen >= 5):
                    if str2_hbegin >= len(str1):
                        str1_hlen, str1_hbegin, str1_hend = 0, 0, 0
                    else:
                        str1_hlen, str1_hbegin, str1_hend = get_homopolymer_len(str1, str2_hbegin)
                    
                    in_homopolymer = 1
                    base_l = str2[j1]
                        
            diff_count += max(i2 - i1, j2 - j1)

    if in_homopolymer and diff_count == 1:
        return (True, str1_hlen, str2_hlen, base_l, str1_hbegin) # homo_len is the length of the longer version
    
    return (False, 0, 0, "", 0)

def safe_parse(val, cast_type):
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.strip('[]')
    return cast_type(val)

##############################
######## DATA CREATION #######
##############################

NUM_READ = 3

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
    base = safe_parse(row['base'], str)
    raw_start = safe_parse(row['raw start idx'], int)
    raw_end = safe_parse(row['raw end idx'], int)
    num_bases = safe_parse(row['num_bases'], int)

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
        success = False
        for scale_factor in scale_factors:

            corrupt_chunk = clean_chunk.copy()
            
            dampen_radius = int((homo_len / 2) * dampen_width_percent)
            dampen_radius = max(1, dampen_radius)
            local_mean = np.mean(corrupt_chunk[midpoint - dampen_radius : midpoint + dampen_radius])
            corruption_start = midpoint - dampen_radius;
            corruption_end = midpoint + dampen_radius
            corrupt_chunk[corruption_start:corruption_end] = ( 
                (corrupt_chunk[corruption_start :  midpoint + dampen_radius] * scale_factor) \
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

            is_valid, str1_hlen, str2_hlen, base_l, hbegin_idx = is_homopolymer_single_indel(clean_str, corrupt_str)

            if (not is_valid): # Check if it counts or not
                print(f"Skipping: corruption with width {dampen_width_percent} and scale {scale_factor} did not make a valid indel corruption.")
                continue
            
            # Should count
            print(f"Possible success with width {dampen_width_percent} and scale {scale_factor}")
            print(f"Clean:  {clean_str}")
            print(f"Corrupt:{corrupt_str}")

            
            inputs.append({
                **row.to_dict(),
                "Dampen width": dampen_width_percent,
                "Scale Factor": scale_factor,
                "Clean string": clean_str,
                "Corrupt_string": corrupt_str,
                "H Begin Idx": hbegin_idx,
                "Base Letter": base_l,
                "Clean H-er Length": str1_hlen,
                "Corrupt H-er Length": str2_hlen,
                "Type": ("Insertion" if str2_hlen > str1_hlen else "Deletion"),
                "Corruption Start Raw": chunk_start + corruption_start,
                "Corruption End Raw": chunk_start + corruption_end
            })
            success = True
            break
        if success:
                break


            
            
        

    noise_size = 6
    noise_offset = -50
    # Try adding some noise

    success = False

    for i in range(rel_homo_start, rel_homo_end - noise_size):
        corrupt_chunk = clean_chunk.copy()

        noise_idx = max(0, i - noise_offset)
        
        corruption_start = i
        corruption_end = i + 6
        corrupt_chunk[corruption_start : corruption_end] = corrupt_chunk[noise_idx : noise_idx + noise_size]

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

        is_valid, str1_hlen, str2_hlen, base_l, hbegin_idx = is_homopolymer_single_indel(clean_str, corrupt_str)
        if not is_valid:
            print(f"Skipping: corruption with noise idx {noise_idx} at relative index {i} did not make a valid indel corruption.")
            continue

      
        print(f"Possible success with noise idx {noise_idx} at relative index {i}")
        print(f"Clean:  {clean_str}")
        print(f"Corrupt:{corrupt_str}")

        inputs.append({
            **row.to_dict(),
            "Noise source idx": noise_idx,
            "Insert idx": i,
            "Clean string": clean_str,
            "Corrupt_string": corrupt_str,
            "H Begin Idx": hbegin_idx,
            "Base Letter": base_l,
            "Clean H-er Length": str1_hlen,
            "Corrupt H-er Length": str2_hlen,
            "Type": ("Insertion" if str2_hlen > str1_hlen else "Deletion"),
            "Corruption Start Raw": chunk_start + corruption_start,
            "Corruption End Raw": chunk_start + corruption_end
        })
        success = True
        break

        
    

df = pd.DataFrame(inputs)
df.to_csv(f"Input_gen_results_read_{NUM_READ}.csv", index=False)

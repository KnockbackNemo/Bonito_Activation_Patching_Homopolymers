# Overview #

This repo contains the code and data used in the engineering undergraduate thesis "An Exploration of the Oxford Nanopore Technologies Bonito Model Through Activation Patching on Homopolymer Reads," which can be read here:
https://drive.google.com/file/d/1cqdEWf3xB1rUcSQ5CsdD_FrkICirZYSM/view

This includes
- Scripts (find_kmers.py, Test_Inputs.py, Test_Inputs_control.py, Run_Patches.py, Analyze_Results_with_Summaries.py, helpers.py)
- Input data
- Experimental results and intermediate data

## How to use this repo ##

### Setup notes ###

This experiment was run in WSL on an Ubuntu 20.04 distro within a Python 3.12 conda environment.

### Current contents ###

The graphs and data produced in this experiment are already present in the repo and can be viewed without running additional scripts.

If you want to look at data, see
- *reads* for input reads

If you want to look at results graphs and csvs, see
- *Results_and_Figures/comparative_analysis_outputs* for a comparison of homopolymer vs non-homopolymer error results
- *Results_and_Figures/comparative_analysis_outputs_experimental_only* to see only the homopolymer runs (can be easier to compare components)
- Within each of these folders, you can find a breakdown of results by denoising/noising in the *denoising_combined_analysis* and *noising_combined_analysis* folders
    - Combined graphs and stats are within these folders, and additional selections of component comparisons can be found in *heads_analysis*, *Individual_Components*, and *MLP_atn_layer_analysis* for less busy visuals

### Rerunning the experiment ###

If you want to look at intermediate scripts or rerun the whole process, the order is

1. Run find_kmers.py to find homopolymer regions.
    - There are 10 reads in the /reads/....pod5 file. Change READ_NUM at the top to search through different ones (the experiment uses reads 1-8).
    - This produces csvs in the kmers_data_reads folder

2. Run Test_Inputs to generate input pairs.
    - This applies various corruptions to the signals listed in kmers csvs to create clean/corrupt input pairs (added to Intermediate_Data/clean_corrupt_pairs/all/Homopolymer)
    - Change READ_NUM to choose which read to use
    - This script takes a long time to run
    - Note: A large unsolved issue is that decoding is done using Bonito's decode model rather than the viterbi algorithm used in find_kmers.py. This means that not all data ends up being used since decode may return a different string. This is a limitation that is slated to be addressed but has not been addressed yet.

3. Run Test_Inputs_control.py to generate comparison input pairs
    - Change NUM_READ to choose which read to use
    - We acknowledge that this is not a true control given the current data size and distribution. This is another fix intended to be included in future work.
    - Output files will be in Intermediate_Data/clean_corrupt_pairs/all/Nonhomopolymer

4. Run Run_Patches.py to perform activation patching.
    - Toggle IS_CONTROL to run the script on homopolymer (False) vs nonhomopolymer (True) reads.
    - This script will run layer, MLP, attention, and head-level sweeps on all input pairs.
    - Due to time/computational power constraints, not all pairs were used in the experiment. The pairs used are in the Intermediate_Data/clean_corrupt_pairs/selected folder.
    - Results will be placed in Results_and_Figures/Patching/Homopolymer or Results_and_Figures/Patching/Nonhomopolymer

5. Run Analyze_Results_with_Summaries.py to create graphs and aggregated data csvs
    - This will generate aggregated data figures, input summaries, and master csvs with result data
    - Output is written to Results_and_Figures/comparative_analysis_outputs or comparative_analysis_outputs_experimental_only
    - Toggle GENERATE_EXPERIMENTAL_ONLY = True to skip plotting comparison (nonhomopolymer) data
    - Toggle GENERATE_MLP_ATN_ONLY = True to plot graphs comparing just the MLP and ATN components
    - Toggle GENERATE_HEADS_ONLY = True to plot graphs comparing just the attention heads

Acknowledgement: These scripts were generated and refactored with AI assistance.

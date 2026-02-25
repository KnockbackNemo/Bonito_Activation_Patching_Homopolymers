import h5py

# Open a Fast5 file
with h5py.File('5210_N128870_20170307_FN2002033683_MN19691_mux_scan_Klebs_Ecoli_HI_barcode_98141_ch204_read46_strand.fast5', 'r') as h5_file:
    # List top-level groups (e.g., 'Raw', 'Analyses')
    print(list(h5_file.keys()))
    
    # Access a specific read's raw data
    read_id = list(h5_file['Raw'].keys())[0]
    raw_signal = h5_file[f'Raw/{read_id}/Signal'][()]
    
    print(f"Signal shape: {raw_signal.shape}")

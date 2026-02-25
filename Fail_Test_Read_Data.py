""" From https://pypi.org/project/ont-fast5-api/ """

from ont_fast5_api.fast5_interface import get_fast5_file

def print_all_raw_data():
    fast5_filepath = "5210_N128870_20170307_FN2002033683_MN19691_mux_scan_Klebs_Ecoli_HI_barcode_98141_ch204_read46_strand.fast5" # This can be a single- or multi-read file
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data()
            print(read.read_id, raw_data)

    #pkg_resources deprecated

print_all_raw_data()
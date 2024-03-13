from enum import Enum

Usage = Enum("Usage", ["SINGLE_CELL", "MULTI_CELL"])

### WARNING: DO NOT EDIT THIS FILE ABOVE THIS LINE! ###

usage = Usage.MULTI_CELL
extract_traces = True
acquisition_frequency = 50
pacing_frequency = 1
apply_photo_bleach_correction = False
beginning_frames_removed = 0
max_pacing_deviation = 0.1
good_snr = True
quiet = False

# videos_directory = "E:/06-12-2023_PGP1WT-RGECO-aActGFP_PFRPMI_afi_acute/dmso"
# videos_directory = "D:/30-11-20 ISO WT/3um"
videos_directory = "sample_videos/multi_cell"
# videos_directory = "sample_videos/dataset1_RGECO_SingleCell"
# videos_directory = "sample_videos/notworkinglol"

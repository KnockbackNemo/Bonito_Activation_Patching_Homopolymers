
# Metrics I want to look at/questions to answew/things I have concerns about
# Can I add a negative control, and if so, what would be the best way to do that? Just take a non-homopolymer string and patch it?
# Some components both degrade and recovery depending on the timestamp, so I might want to take the absolute effect (is it AIE?) or someting like that
# What about the center of gravity metric?
# How might the results change based on the length, error type (insertion / deletion), and base that was used?
# If I don't know what timestamp is the most important, should I take the max score? (Especially if I don't want the varying patching windows to affect results)
# If I want to know if maybe the effect depends on the part of the homopolymer (for example recover is boosted at the beginning but suppressed at the end), how can I take into account not know the exact timestamp
# but also caring about the position in relation to the homopolymer? Maybe I can look at the gradient across time?
# Are there any other important metrics I'm missing?
# What about how the logits change sometimes, but the posteriors don't? I wonder what I can make of that?
# And should I compare the fact that the decode string may or may not change?
# I can at least plot relative to the window and mark the corruption point

# TODO
# Find negative control data
# Run negative control sweep
# Look at max score
# Have component over time (relative to window), marking the corruption point
        # note some runs might be invalid if the homopolymer in the window is different
# Track both logits and posteriors?
# Track whether or not the string flipped
# Bar charts for stuff
# One big heatmap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("./patch_results")

def aggregate_and_plot(component_name, metric_type):

    search_pattern = f"*{component_name_{metric}*.csv}"
    csv_files = list(OUTPUT_DIR.rglob(search_pattern))

    if not csv_files:
        print(f"No files found for {component_name} {metric_type}")
        return

    print(f"Found {len(csv_files)} files for {component_name} {metric_type}. Aggregating...")

    all_scores = []

    # Load files
    for file in csv_files:
        df = pd.read_csv(file)

        scores = df.loc[:, 'Layer'].values
        all_scores.append(scores) ## TODO: read in scores and stuff




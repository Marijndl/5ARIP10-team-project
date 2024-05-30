import os
import pandas as pd

# Directory containing CSV files
directory = "Segments_CSV"

# List to store DataFrame objects
dfs = []

# Iterate through each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read CSV file into a DataFrame and append to the list
        df = pd.read_csv(filepath)
        dfs.append(df)

# Concatenate all DataFrames in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Output combined DataFrame to a single CSV file
output_file = "combined_segment_points.csv"
combined_df.to_csv(output_file, index=False)

print("Combined CSV file saved as:", output_file)

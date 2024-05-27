import os
import re

# Directory containing the CSV files
directory = f'.\\CTA data\\Segments renamed\\'

# Regular expression pattern to match filenames like "SegmentPoints_1_0.csv"
pattern = r'SegmentPoints_(\d+)_(\d+)\.csv'

# Iterate through files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file and matches the pattern
    if filename.endswith('.csv') and re.match(pattern, filename):
        # Extract the numbers from the filename
        match = re.match(pattern, filename)
        num1 = match.group(1)
        num2 = match.group(2)

        # Pad the numbers with leading zeros
        num1_padded = num1.zfill(4)

        # Pad
        num2_padded = num2.zfill(2)

        # Construct the new filename
        new_filename = f'SegmentPoints_{num1_padded}_{num2_padded}.csv'

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        print(f'Renamed {filename} to {new_filename}')
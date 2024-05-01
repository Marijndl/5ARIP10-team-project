import os
import glob

# Define the directory where your files are located
directory = "D:\\CTA data\\Segments original\\"

# Use glob to get a list of files ending with ".schema"
files_to_delete = glob.glob(os.path.join(directory, "*.schema.csv"))

# Iterate over the files and delete them
for file_path in files_to_delete:
    os.remove(file_path)
    print(f"Deleted file: {file_path}")

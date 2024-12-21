import os
import shutil

# Define the base directory and the target directory for merged files
base_dir = r"D:\MIGLAB\MRI_labled_new\OneDrive_2024-10-22\Converted MRI images by 3D Slicer"  # Replace with your actual path
target_dir = r"D:\NEU\CS5330\mini_proj_10\MRI\MRI_ds\axial_mask"  # Replace with your desired target path

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Walk through each subfolder in the base directory
for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Define the path to the 'original/axial_png' folder inside each subfolder
        axial_png_path = os.path.join(subfolder_path, 'mask', 'axial_png')
        
        if os.path.isdir(axial_png_path):
            # Iterate through each PNG file in the 'axial_png' folder
            for filename in os.listdir(axial_png_path):
                if filename.endswith('.png'):
                    # Construct the source file path
                    source_file = os.path.join(axial_png_path, filename)
                    
                    # Define the new filename based on the grandparent folder name
                    new_filename = f"{subfolder}_{filename}"
                    
                    # Construct the destination file path in the target directory
                    dest_file = os.path.join(target_dir, new_filename)
                    
                    # Copy the file to the target directory with the new name
                    shutil.copy2(source_file, dest_file)

print("All axial PNG files have been merged into the target directory.")

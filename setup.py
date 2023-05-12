import os
import sys
import shutil
import glob

# Get the path of the current virtual environment
venv_dir = sys.prefix
site_packages_dir = ""

# Build path to paste into
if sys.platform.startswith('win'):
    site_packages_dir = os.path.join("./.venv", 'Lib', 'site-packages')
else:
    site_packages_dir = os.path.join("./.venv", 'lib', 'python{}'.format(sys.version[:4]), 'site-packages')

# Find build to copy
search_directory = './cuda_kernels'
search_pattern = '*cuda_kernels*'

print(glob.glob(search_directory + '/' + search_pattern))
build_file_path = glob.glob(search_directory + '/' + search_pattern)[0]

# Copy build to correct location
print("Copying ", build_file_path, " to ", site_packages_dir)
shutil.copy2(build_file_path, site_packages_dir)
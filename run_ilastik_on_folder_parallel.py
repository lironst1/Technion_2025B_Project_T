import os
from glob import glob
import numpy as np
import subprocess
import tempfile
import h5py
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tifffile


def run_ilastik_single_image(args):
	ilastik_path, project_path, input_path, save_dir, save_format = args

	with tempfile.TemporaryDirectory() as temp_dir:
		output_file = os.path.join(temp_dir, "output.h5")

		command = [
			ilastik_path,
			f"--headless",
			f"--project={project_path}",
			f"--export_source=probabilities",
			f"--output_format=hdf5",
			f"--output_filename_format={output_file}",
			input_path
		]

		subprocess.run(command, check=True)

		with h5py.File(output_file, "r") as f:
			dataset_keys = list(f.keys())
			data = f[dataset_keys[0]][:]

	# Save output
	base_filename = os.path.splitext(os.path.basename(input_path))[0]
	if save_dir is not None:
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f"{base_filename}.{save_format}")
		if save_format == "npy":
			np.save(save_path, data)
		elif save_format == "tif":
			tifffile.imwrite(save_path, data.astype(np.float32))  # or np.uint8 if needed
		else:
			raise ValueError(f"Unknown save_format: {save_format}")

	return data


def run_ilastik_on_folder_parallel(ilastik_path, project_path, folder_path, file_extension="tif",
		num_workers=None, save_dir=None, save_format="npy"):
	"""
	Run ilastik headless on all images in a specified folder using parallel processing, providing
	capabilities for automatic saving and progress tracking. This function enables processing of
	multiple images in parallel with user-specified configurations such as the number of workers,
	file extensions, and output formats.

	Parameters
	----------
	ilastik_path : str
	    Path to the Ilastik's 'run_ilastik.bat' script used for headless processing.
	project_path : str
	    Path to the Ilastik project file (.ilp) for loading the pre-trained model.
	folder_path : str
	    Path to the directory containing the images to process.
	file_extension : str, optional
	    File extension of image files to process (default is `"tif"`).
	num_workers : int, optional
	    Number of parallel processes to spawn for concurrent processing; defaults to using
	    all available CPUs minus one.
	save_dir : str, optional
	    Directory path for saving the output files. If not specified, results are not saved.
	save_format : str, optional
	    Format for saving output files, either `"npy"` or `"tif"` (default is `"npy"`).

	Returns
	-------
	list of np.ndarray
	    List of processed output arrays corresponding to the images in the specified folder.
	"""
	search_pattern = os.path.join(folder_path, f"*.{file_extension}")
	input_paths = sorted(glob(search_pattern))

	if not input_paths:
		raise ValueError(f"No .{file_extension} files found in {folder_path}")

	print(f"Found {len(input_paths)} images. Running ilastik in parallel...")

	args_list = [(ilastik_path, project_path, input_path, save_dir, save_format) for input_path in input_paths]

	if num_workers is None:
		num_workers = max(1, cpu_count() - 1)

	print(f"Using {num_workers} worker(s)")

	outputs = []
	with Pool(processes=num_workers) as pool:
		for result in tqdm(pool.imap(run_ilastik_single_image, args_list), total=len(args_list)):
			outputs.append(result)

	print("Finished processing all images.")
	return outputs


ilastik_path = r"C:\path\to\ilastik-1.4.0b21\run_ilastik.bat"
project_path = r"C:\path\to\your_project.ilp"
folder_path = r"C:\path\to\your_folder_of_images"
save_dir = r"C:\path\to\where_you_want_outputs"

outputs = run_ilastik_on_folder_parallel(
		ilastik_path,
		project_path,
		folder_path,
		save_dir=save_dir,
		save_format="npy",  # or "tif"
		num_workers=4
)

for idx, arr in enumerate(outputs):
	print(f"Image {idx}: shape {arr.shape}, dtype {arr.dtype}")

import os
import subprocess
import numpy as np
import tempfile
from tqdm import tqdm

from __cfg__ import PATH_ILASTIK_EXE, IMAGE_EXTENSIONS, logger
import tests


def run_ilastik(path_project, dir_root, image_ext=".tif"):
	if not os.path.exists(path_project):
		raise ValueError(f"Project file not found at {path_project}")
	if not os.path.isdir(dir_root):
		raise ValueError(f"Path is not a directory: {dir_root}")
	if " " in path_project or " " in dir_root:  # Ilastik doesn't like spaces in the path_project
		tests.symlink_admin_priv()
	if image_ext not in IMAGE_EXTENSIONS:
		raise ValueError(f"Invalid image extension: {image_ext}. Supported extensions are: {IMAGE_EXTENSIONS}")

	dir_out = os.path.join(dir_root, "output")
	os.makedirs(dir_out, exist_ok=True)

	with tempfile.TemporaryDirectory() as temp_dir:
		if " " in path_project:  # Ilastik doesn't like spaces in the path_project
			tmp = path_project
			path_project = os.path.join(temp_dir, os.path.basename(tmp))
			os.symlink(src=tmp, dst=path_project, target_is_directory=False)
			logger.info(f"Created symlink to {path_project}")

		if " " in dir_root:
			tmp = dir_root
			dir_root = os.path.join(temp_dir, os.path.basename(tmp))
			os.symlink(src=tmp, dst=dir_root, target_is_directory=True)
			logger.info(f"Created symlink to {dir_root}")

			tmp = dir_out
			dir_out = os.path.join(temp_dir, os.path.basename(tmp))
			os.symlink(src=tmp, dst=dir_out, target_is_directory=True)
			logger.info(f"Created symlink to {dir_out}")

		filenames_format = os.path.join(dir_root, f"*{image_ext}")
		output_filename_format = os.path.join(dir_out, "{nickname}.npy")

		command = [
			PATH_ILASTIK_EXE,
			f'--headless',
			f'--readonly',
			f'--input-axes=zyx',
			# f'--stack_along="c"',
			f'--export_source=probabilities',
			f'--project={path_project}',
			f'--output_format=numpy',
			f'--output_filename_format={output_filename_format}',
			f'{filenames_format}'
		]

		subprocess.run(command, check=True)

		"""
		TODO: Current error:
		ilastik.applets.base.applet.DatasetConstraintError: Constraint of 'Pixel Classification' applet was violated: All input images must have the same dimensionality.  Your new image has 4 dimensions (including channel), but your other images have 3 dimensions.
		ERROR 2025-05-04 01:26:14,752 log_exception 33984 23564 Project could not be loaded due to the exception shown above.
		Aborting Project Open Action
		"""
		pass


# 	with h5py.File(output_file, "r") as f:
# 		dataset_keys = list(f.keys())
# 		data = f[dataset_keys[0]][:]
#
# # Save output
# base_filename = os.path.splitext(os.path.basename(input_path))[0]
# if save_dir is not None:
# 	os.makedirs(save_dir, exist_ok=True)
# 	save_path = os.path.join(save_dir, f"{base_filename}.{save_format}")
# 	if save_format == "npy":
# 		np.save(save_path, data)
# 	elif save_format == "tif":
# 		tifffile.imwrite(save_path, data.astype(np.float32))  # or np.uint8 if needed
# 	else:
# 		raise ValueError(f"Unknown save_format: {save_format}")


def run_ilastik_parallel(
		path_project,
		filenames,
		dir_target=None,
		save_format="npy"
):
	"""
	Run ilastik headless on all images in a specified folder using parallel processing, providing
	capabilities for automatic saving and progress tracking. This function enables processing of
	multiple images in parallel with user-specified configurations such as the number of workers,
	file extensions, and output formats.

	Parameters
	----------
	path_project :      str
	    Path to the Ilastik project file (.ilp) for loading the pre-trained model.
	filenames :         list[str]
	    List of image filenames to be processed. If a directory is provided, all images in the directory will be processed.
	dir_target :        str, optional
	    Directory path for saving the output files. If not specified, results are not saved.
	save_format :       str, optional
	    Format for saving output files, either `"npy"` or `"tif"` (default is `"npy"`).

	Returns
	-------
	list[np.ndarray]
	    List of processed output arrays corresponding to the images in the specified folder.
	"""
	if not os.path.exists(path_project):
		raise ValueError(f"Project file not found at {path_project}")
	if save_format not in ["npy", "tif"]:
		raise ValueError(f"Invalid save format: {save_format}. Use 'npy' or 'tif'.")

	args_list = [(PATH_ILASTIK_EXE, path_project, f, dir_target, save_format) for f in filenames]

	# outputs = parallel_threading(
	# 		func=run_ilastik,
	# 		iterable=args_list,
	# 		)
	outputs = []
	# with Pool(processes=n_workers) as pool:
	# 	for result in tqdm(pool.imap(run_ilastik, args_list), total=len(args_list)):
	# 		outputs.append(result)
	for args in tqdm(args_list):
		outputs.append(run_ilastik(*args))

	logger.info("Finished processing all images.")
	return outputs

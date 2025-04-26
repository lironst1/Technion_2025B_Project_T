import os
import random
import shutil


# def move_and_rename_files(root_dir):
# 	for dirpath, _, filenames in os.walk(root_dir, topdown=False):
# 		for filename in filenames:
# 			# Get the relative path from the root directory
# 			relative_path = os.path.relpath(dirpath, root_dir)
# 			# Create the new name based on the original subdirectory structure
# 			new_name = f"{relative_path.replace(os.sep, '_')}_{filename}"
# 			# Get the full path of the current file
# 			current_file_path = os.path.join(dirpath, filename)
# 			# Move the file to the parent directory with the new name
# 			shutil.move(current_file_path, os.path.join(root_dir, new_name))
# 			if filename == filenames[1]:
# 				print(f"Moved: {current_file_path} -> {os.path.join(root_dir, new_name)}")
#
#
# if __name__ == "__main__":
# 	# Replace this with the path to your target directory
# 	root_directory = r"C:\Users\liron\Downloads\Data"  # Current directory
# 	move_and_rename_files(root_directory)


def select_and_move_images(source_dir, training_dir, N):
	# Get all files in the source directory
	all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

	# Filter the list to only include image files (optional)
	image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']
	image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]

	# Ensure N is not greater than the number of images in the source directory
	N = min(N, len(image_files))

	# Randomly select N images
	selected_images = random.sample(image_files, N)

	# Create the training folder if it does not exist
	if not os.path.exists(training_dir):
		os.makedirs(training_dir)

	# Move selected images to the training folder
	for image in selected_images:
		source_path = os.path.join(source_dir, image)
		dest_path = os.path.join(training_dir, image)
		shutil.move(source_path, dest_path)
		print(f"Moved {image} to {training_dir}")


# Example usage
source_directory = r'C:\Users\liron\Downloads\Data'
training_directory = r'C:\Users\liron\Downloads\Data\train'
num_images_to_move = 300

select_and_move_images(source_directory, training_directory, num_images_to_move)

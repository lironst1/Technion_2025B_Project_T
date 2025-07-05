import argparse

from pathlib import Path


def tuple_type(strings):
	strings = strings.replace("(", "").replace(")", "")
	mapped_int = map(int, strings.split(","))
	return tuple(mapped_int)


def parse_args():
	"""
	Parse command line arguments.
	Returns
	-------
	argparse.Namespace: Parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description=globals()["__doc__"])

	parser.add_argument(
			"-d",
			"--dir",
			type=str,
			default=".",
			help="Directory containing the images",
	)

	parser.add_argument(
			"-dl",
			"--directories_list",
			type=str,
			default=None,
			help="A file with a list of directories to process",
	)

	return parser.parse_args()


def main():
	r"""
	Main function to run the script.
	Parse command line arguments and execute the main logic.

	Examples
	--------
	python main.py --dir /path/to/images --output /path/to/output

	Returns
	-------

	"""

	args = parse_args()

	raise NotImplementedError


if __name__ == "__main__":
	main()

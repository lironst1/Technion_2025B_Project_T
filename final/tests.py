import os
import tempfile
import pandas as pd

from __cfg__ import logger, DATA_TYPES


def file_exist(path):
	"""Checks if the file exists at the given path."""
	if not os.path.exists(path):
		raise FileNotFoundError(f"File not found: {path}")


def dir_exist(path):
	"""Checks if the directory exists at the given path."""
	if not os.path.isdir(path):
		raise NotADirectoryError(f"Directory not found: {path}")


def symlink_admin_priv():
	"""Checks if the given path is a symlink and if it requires admin privileges to access."""
	with tempfile.TemporaryDirectory() as temp_dir:
		# Create a symlink in the temporary directory
		try:
			os.symlink(os.getcwd(), os.path.join(temp_dir, "symlink"), target_is_directory=True)
			print("hello")
		except OSError as e:
			if e.errno == 22:
				logger.error(f"Failed to create symlink as admin privileges are required."
				             f"Run the following command in an admin PowerShell:\n"
				             f'reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v "AllowDevelopmentWithoutDevLicense" /d "1"')
			raise


def excel_permissions(path_excel):
	"""Tests if the user has permissions to access Excel files."""
	try:
		excel_data = pd.read_excel(path_excel)
	except PermissionError:
		logger.error(f"Permission error. Make sure to close the Excel file before running the script.")
		raise


def data_type_valid(data_type):
	"""Checks if given data type is valid"""
	if data_type not in DATA_TYPES:
		msg = f"Invalid data type '{data_type}'. Valid types are: {', '.join(DATA_TYPES.keys())}."
		logger.error(msg)
		raise ValueError(msg)

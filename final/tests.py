import os
import tempfile

from __cfg__ import logger


def test_symlink_admin_priv():
	"""Tests if the given path is a symlink and if it requires admin privileges to access."""
	with tempfile.TemporaryDirectory() as temp_dir:
		# Create a symlink in the temporary directory
		symlink_path = os.path.join(temp_dir, "symlink")

		try:
			os.symlink(os.getcwd(), os.path.join(temp_dir, "symlink"), target_is_directory=True)
			print("hello")
		except OSError as e:
			if e.errno == 22:
				logger.error(f"Failed to create symlink as admin privileges are required."
				             f"Run the following command in an admin PowerShell:\n"
				             f'reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v "AllowDevelopmentWithoutDevLicense" /d "1"')
			raise

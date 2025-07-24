import re
from datetime import datetime
from pathlib import Path

PATH_INIT = Path("__init__.py")  # <-- adjust this to your module


def update_version():
    today = datetime.now().strftime("%Y.%m.%d")
    content = PATH_INIT.read_text()
    new_content, n = re.subn(
            r'__version__\s*=\s*["\']\d{4}\.\d{2}\.\d{2}["\']',
            f'__version__ = "{today}"',
            content
    )
    if n:
        PATH_INIT.write_text(new_content)
        print(f"Updated version to {today}")
    else:
        print("No version string found or already up to date.")


if __name__ == "__main__":
    update_version()

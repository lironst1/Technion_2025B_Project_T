# Exit if any command fails
$ErrorActionPreference = "Stop"

# Define variables
$DIR_ROOT = "C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\original"
$PATH_EXCEL = "C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\betacatenin_head.xlsx"
$DATE = "2025-03-05"
$POS = 1

# Run Python script with arguments
python main.py --dir "$DIR_ROOT" --excel "$PATH_EXCEL" --date "$DATE" --pos $POS --segment_manually
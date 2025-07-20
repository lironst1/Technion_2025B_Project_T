#!/bin/bash

# Exit if any command fails
set -e

DIR_ROOT="C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\original"
PATH_EXCEL="C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\betacatenin_foot.xlsx"
DATE="2025-02-27,2025-03-05"
POS=2,12

python main.py --dir "$DIR_ROOT" --excel "$PATH_EXCEL" --date "$DATE" --pos $POS

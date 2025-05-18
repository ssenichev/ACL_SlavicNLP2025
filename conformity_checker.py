import argparse
import re
from config import VALID_LABELS

import warnings
warnings.filterwarnings('always')
warnings.simplefilter('always')

def check_subtask_1_format(filepath):
    issue_detected = False
    with open(filepath, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            line = line.strip()  # Remove leading/trailing whitespace
            parts = line.split("\t")
            
            # Check column count
            if len(parts) != 4:
                print(f"Line {i} in Subtask 1 file is invalid: Incorrect number of columns (expected 4, got {len(parts)}) in line: {line}")
                issue_detected = True
                continue
            
            doc_id, start, end, persuasion_flag = parts

            # Check document ID format
            if not re.match(r".*\.txt$", doc_id):
                print(f"Line {i} in Subtask 1 file has an invalid document ID: {doc_id} in line: {line}")
                issue_detected = True

            # Check start and end offsets
            if not (start.isdigit() and end.isdigit()):
                print(f"Line {i} in Subtask 1 file has invalid start or end offsets: start={start}, end={end} in line: {line}")
                issue_detected = True
                continue  # Skip the next check if offsets are invalid
            if int(start) > int(end):
                print(f"Line {i} in Subtask 1 file has start greater than end: start={start}, end={end} in line: {line}")
                issue_detected = True

            # Check persuasion flag (case-insensitive, handles whitespace)
            if persuasion_flag.lower().strip() not in {"true", "false"}:
                print(f"Line {i} in Subtask 1 file has an invalid persuasion_flag: {persuasion_flag} in line: {line}")
                issue_detected = True
    if issue_detected:
        raise Exception("Format check completed. Issues detected in Subtask 1 file. Invalid submission file.")
    else:
        print("Format check completed without errors. Subtask 1 file.")


def check_subtask_2_format(filepath):
    issue_detected = False
    with open(filepath, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            line = line.strip()  # Remove leading/trailing whitespace
            parts = line.split("\t")
            
            # Check column count (at least 3 for doc_id, start, and end; labels can be empty)
            if len(parts) < 3:
                print(f"Line {i} in Subtask 2 file is invalid: Fewer than 3 columns (expected at least 3, got {len(parts)}) in line: {line}")
                issue_detected = True
                continue
            
            doc_id, start, end, *labels = parts

            # Check document ID format
            if not re.match(r".*\.txt$", doc_id):
                print(f"Line {i} in Subtask 2 file has an invalid document ID: {doc_id} in line: {line}")
                issue_detected = True

            # Check start and end offsets
            if not (start.isdigit() and end.isdigit()):
                print(f"Line {i} in Subtask 2 file has invalid start or end offsets: start={start}, end={end} in line: {line}")
                issue_detected = True
                continue
            if int(start) > int(end):
                print(f"Line {i} in Subtask 2 file has start greater than end: start={start}, end={end} in line: {line}")
                issue_detected = True
                continue

            # Check each label for validity
            for label in labels:
                if label and label not in VALID_LABELS:
                    print(f"Line {i} in Subtask 2 file contains an invalid label: {label} in line: {line}")
                    issue_detected = True
    if issue_detected:
        raise Exception("Format check completed. Issues detected in Subtask 2 file. Invalid submission file.")
    else:
        print("Format check completed without errors. Subtask 2 file.")

def main():
    parser = argparse.ArgumentParser(description="Check submission conformity")
    parser.add_argument("subtask", choices=["subtask1", "subtask2"], help="Which subtask to check")
    parser.add_argument("submission_file", type=str, help="Path to the submission file")
    args = parser.parse_args()

    if args.subtask == "subtask1":
        check_subtask_1_format(args.submission_file)
    else:
        check_subtask_2_format(args.submission_file)

if __name__ == "__main__":
    main()
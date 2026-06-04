#!/usr/bin/env python3
"""
Script to extract anthropometric data from Excel files for all participants.

This script reads the 'antro' (or 'anthro') Excel files from each participant's 
directory and extracts weight and height measurements from the first measurement 
column.
"""

import pandas as pd
from pathlib import Path
import os
import glob
from jhamon_training.pathutils import RESULTS_TRAINING_PATH, dame_participants
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def find_anthropometric_file(participant_path):
    """
    Find the anthropometric Excel file in the participant's directory.
    
    Args:
        participant_path (Path): Path to the participant's directory
        
    Returns:
        Path or None: Path to the anthropometric file if found, None otherwise
    """
    # Common variations of the anthropometric file name
    possible_names = [
        "antro.xlsx",
        "anthro.xlsx", 
        "antro.xls",
        "anthro.xls",
        "*antro*.xlsx",
        "*anthro*.xlsx",
        "*antro*.xls",
        "*anthro*.xls"
    ]
    
    for pattern in possible_names:
        files = list(participant_path.glob(pattern))
        if files:
            return files[0]  # Return the first match
    
    return None


def extract_anthropometric_data(file_path):
    """
    Extract weight and height from the anthropometric Excel file.
    
    Args:
        file_path (Path): Path to the Excel file
        
    Returns:
        dict: Dictionary with weight and height values from first measurement
    """
    try:
        # Try to read the Excel file
        df = pd.read_excel(file_path, header=None)
        
        # Initialize result dictionary
        result = {
            'weight': None,
            'height': None,
            'measurement_date': None
        }
        
        # Look for weight and height rows
        for idx, row in df.iterrows():
            if pd.isna(row.iloc[0]):
                continue
                
            row_label = str(row.iloc[0]).lower().strip()
            
            # Check if this is a date row (first row)
            if idx == 0 and len(row) > 1:
                # Try to extract the first measurement date
                for col_idx in range(1, len(row)):
                    if not pd.isna(row.iloc[col_idx]):
                        result['measurement_date'] = row.iloc[col_idx]
                        break
            
            # Check for weight
            if 'weight' in row_label or 'peso' in row_label:
                # Get the first non-null value after the label
                for col_idx in range(1, len(row)):
                    if not pd.isna(row.iloc[col_idx]):
                        weight_val = row.iloc[col_idx]
                        # Handle comma as decimal separator
                        if isinstance(weight_val, str):
                            weight_val = weight_val.replace(',', '.')
                        try:
                            result['weight'] = float(weight_val)
                        except (ValueError, TypeError):
                            pass
                        break
            
            # Check for height
            elif 'height' in row_label or 'altura' in row_label or 'talla' in row_label:
                # Get the first non-null value after the label
                for col_idx in range(1, len(row)):
                    if not pd.isna(row.iloc[col_idx]):
                        height_val = row.iloc[col_idx]
                        # Handle comma as decimal separator
                        if isinstance(height_val, str):
                            height_val = height_val.replace(',', '.')
                        try:
                            result['height'] = float(height_val)
                        except (ValueError, TypeError):
                            pass
                        break
        
        return result
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def create_anthropometric_dataset(path_to_data, output_path):
    """
    Create a complete anthropometric dataset for all participants.
    
    Args:
        path_to_data (Path): Path to the main data directory
        output_path (Path): Path where to save the results
    """
    # Get all participants
    all_participants = dame_participants()
    
    # Determine training groups based on pathutils.py
    nht_participants = [
        "jhamon01", "jhamon02", "jhamon03", "jhamon04", "jhamon05", "jhamon06",
        "jhamon09", "jhamon10", "jhamon11", "jhamon12", "jhamon14", "jhamon15",
        "jhamon16"
    ]
    
    ik_participants = [
        "jhamon18", "jhamon20", "jhamon22", "jhamon23", "jhamon24", "jhamon25",
        "jhamon26", "jhamon28", "jhamon29", "jhamon30", "jhamon31", "jhamon32",
        "jhamon33", "jhamon34"
    ]
    
    # Create results list
    anthropometric_data = []
    
    print(f"Processing {len(all_participants)} participants...")
    
    for participant in all_participants:
        participant_path = path_to_data / participant
        
        # Determine training group
        if participant in nht_participants:
            training_group = "NH"
        elif participant in ik_participants:
            training_group = "IK"
        else:
            training_group = "Unknown"
        
        print(f"Processing {participant} ({training_group})...")
        
        if not participant_path.exists():
            print(f"  Warning: Directory not found for {participant}")
            continue
        
        # Find anthropometric file
        anthro_file = find_anthropometric_file(participant_path)
        
        if anthro_file is None:
            print(f"  Warning: No anthropometric file found for {participant}")
            # Add participant with missing data
            anthropometric_data.append({
                'participant_id': participant,
                'training_group': training_group,
                'weight': None,
                'height': None,
                'measurement_date': None,
                'file_found': False
            })
            continue
        
        print(f"  Found file: {anthro_file.name}")
        
        # Extract data
        data = extract_anthropometric_data(anthro_file)
        
        if data is None:
            print(f"  Warning: Could not extract data from {anthro_file}")
            anthropometric_data.append({
                'participant_id': participant,
                'training_group': training_group,
                'weight': None,
                'height': None,
                'measurement_date': None,
                'file_found': True
            })
            continue
        
        # Add to results
        anthropometric_data.append({
            'participant_id': participant,
            'training_group': training_group,
            'weight': data['weight'],
            'height': data['height'],
            'measurement_date': data['measurement_date'],
            'file_found': True
        })
        
        print(f"  Weight: {data['weight']} kg, Height: {data['height']} cm")
    
    # Create DataFrame
    anthro_df = pd.DataFrame(anthropometric_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save results
    csv_file = output_path / "anthropometric_data.csv"
    excel_file = output_path / "anthropometric_data.xlsx"
    feather_file = output_path / "anthropometric_data.feather"
    
    anthro_df.to_csv(csv_file, index=False)
    anthro_df.to_excel(excel_file, index=False)
    anthro_df.to_feather(feather_file)
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_file}")
    print(f"  Excel: {excel_file}")
    print(f"  Feather: {feather_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total participants: {len(anthro_df)}")
    print(f"Files found: {anthro_df['file_found'].sum()}")
    print(f"Valid weight data: {anthro_df['weight'].notna().sum()}")
    print(f"Valid height data: {anthro_df['height'].notna().sum()}")
    print(f"\nBy training group:")
    print(anthro_df.groupby('training_group').agg({
        'participant_id': 'count',
        'weight': lambda x: x.notna().sum(),
        'height': lambda x: x.notna().sum()
    }).rename(columns={
        'participant_id': 'total_participants',
        'weight': 'valid_weight',
        'height': 'valid_height'
    }))
    
    return anthro_df


def main():
    """Main function to run the anthropometric data extraction."""
    # Define paths (same as in the main training script)
    path_to_data = Path("/Volumes/AJMA/")
    results_output_path = RESULTS_TRAINING_PATH
    
    print("Extracting anthropometric data from Excel files...")
    print(f"Data path: {path_to_data}")
    print(f"Output path: {results_output_path}")
    
    if not path_to_data.exists():
        print(f"Error: Data path {path_to_data} does not exist!")
        print("Please check the path and try again.")
        print("\nAlternatively, you can modify the path_to_data variable in this script.")
        return
    
    # Create the anthropometric dataset
    anthro_df = create_anthropometric_dataset(path_to_data, results_output_path)
    
    print("\nAnthropometric data extraction completed!")
    
    return anthro_df


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract annotations from words.txt based on folder names in train/validation/test files.
Creates three separate files with all annotations belonging to those folders.
"""

def load_folder_ids(filepath):
    """Load folder IDs from train/validation/test files."""
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def extract_annotations(words_file, folder_ids):
    """Extract annotations from words.txt that match the folder IDs."""
    annotations = []
    
    with open(words_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 9:
                continue
            
            word_id = parts[0]
            # Extract base folder ID (e.g., a01-000u from a01-000u-00-00)
            folder_id = '-'.join(word_id.split('-')[:2])
            
            if folder_id in folder_ids:
                annotations.append(line)
    
    return annotations

def save_annotations(annotations, output_file, split_name):
    """Save annotations to output file."""
    with open(output_file, 'w') as f:
        f.write(f"# {split_name.upper()} SET ANNOTATIONS\n")
        f.write(f"# Total: {len(annotations)} entries\n")
        f.write('#' + '='*78 + '\n\n')
        for line in annotations:
            f.write(line + '\n')
    print(f"Created {output_file} with {len(annotations)} entries")

def main():
    base_dir = '/mnt/d/HTR/SimpleHTRNew/data/aachen word'
    
    # Load folder IDs from split files
    print("Loading folder IDs from split files...")
    train_folders = load_folder_ids(f'{base_dir}/train.txt')
    validation_folders = load_folder_ids(f'{base_dir}/validation.txt')
    test_folders = load_folder_ids(f'{base_dir}/test.txt')
    
    print(f"Train folders: {len(train_folders)}")
    print(f"Validation folders: {len(validation_folders)}")
    print(f"Test folders: {len(test_folders)}")
    
    words_file = f'{base_dir}/words.txt'
    
    # Extract annotations for each split
    print("\nExtracting train annotations...")
    train_annotations = extract_annotations(words_file, train_folders)
    save_annotations(train_annotations, f'{base_dir}/train_annotations.txt', 'train')
    
    print("Extracting validation annotations...")
    val_annotations = extract_annotations(words_file, validation_folders)
    save_annotations(val_annotations, f'{base_dir}/validation_annotations.txt', 'validation')
    
    print("Extracting test annotations...")
    test_annotations = extract_annotations(words_file, test_folders)
    save_annotations(test_annotations, f'{base_dir}/test_annotations.txt', 'test')
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Train annotations: {len(train_annotations)}")
    print(f"Validation annotations: {len(val_annotations)}")
    print(f"Test annotations: {len(test_annotations)}")
    print(f"Total: {len(train_annotations) + len(val_annotations) + len(test_annotations)}")

if __name__ == '__main__':
    main()

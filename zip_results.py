#!/usr/bin/env python3
import os
import zipfile
import argparse

def zip_directory_exclude_pt(source_dir, output_zip):
    """
    Zip the contents of source_dir into output_zip, excluding all .pt files
    
    Parameters:
    - source_dir: Directory to zip
    - output_zip: Name of the zip file to create
    """
   
    abs_source = os.path.abspath(source_dir)
    
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
       
        for root, dirs, files in os.walk(abs_source):
            for file in files:
                if file.endswith('.pt'):
                    print(f"Skipping {file} (excluded file type)")
                    continue
                
                abs_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file_path, abs_source)
                print(f"Adding {rel_path}")
                zipf.write(abs_file_path, rel_path)

def main():
    parser = argparse.ArgumentParser(description='Zip a directory excluding .pt files')
    parser.add_argument('--directory', help='Directory to zip')
    parser.add_argument('--output', '-o', default='output.zip', help='Output zip filename (default: output.zip)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    print(f"Creating zip archive {args.output} from {args.directory} (excluding .pt files)...")
    zip_directory_exclude_pt(args.directory, args.output)
    print(f"Archive created successfully: {args.output}")

if __name__ == "__main__":
    main()

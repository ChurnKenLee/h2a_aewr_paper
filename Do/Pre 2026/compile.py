import os

def compile_source_files(target_directory=".", output_filename="compiled_code.txt"):
    # Define the extensions we are looking for
    extensions = ('.do', '.R')
    
    # Get a list of all files in the directory and sort them alphabetically
    files = sorted([f for f in os.listdir(target_directory) if f.endswith(extensions)])
    
    if not files:
        print(f"No .do or .R files found in {os.path.abspath(target_directory)}")
        return

    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for filename in files:
                file_path = os.path.join(target_directory, filename)
                
                # Write a clear header for each file
                outfile.write("=" * 80 + "\n")
                outfile.write(f" FILE: {filename}\n")
                outfile.write("=" * 80 + "\n\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"ERROR: Could not read file {filename}. Reason: {e}\n")
                
                # Add spacing between files
                outfile.write("\n\n" + "-" * 80 + "\n\n")
                
        print(f"Successfully compiled {len(files)} files into '{output_filename}'")

    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")

if __name__ == "__main__":
    # You can change "." to a specific folder path if needed
    compile_source_files(target_directory=".")
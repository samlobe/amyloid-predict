#%%
from tqdm import tqdm

# Generalized function to fragment sequences into specified fragment lengths with a sliding window
def fragment_fasta(input_file, output_file, fragment_length=6, sliding_window=3):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        current_header = None
        current_sequence = []
        for line in infile:
            if line.startswith(">"):
                # If a new header is found, process the previous sequence (if any)
                if current_header and current_sequence:
                    print(current_header)
                    # Join the sequence lines into one string
                    full_sequence = ''.join(current_sequence).strip()
                    # Extract start and end residue numbers from the header
                    protein_id = current_header.split('|')[1].strip()
                    sequence_name = current_header.split('|')[0].strip()
                    start_residue = int(sequence_name.split('_')[1])
                    end_residue = int(sequence_name.split('_')[2])
                    
                    # Create fragments, handling sequences shorter than the fragment length
                    if len(full_sequence) < fragment_length:
                        fragment_start = start_residue
                        fragment_end = start_residue + len(full_sequence) - 1
                        new_header = f">{fragment_start}-{fragment_end}|{current_header.strip()}"
                        outfile.write(new_header + "\n")
                        outfile.write(full_sequence + "\n")
                    else:
                        for i in range(0, len(full_sequence) - fragment_length + 1, sliding_window):
                            fragment = full_sequence[i:i + fragment_length]
                            fragment_start = start_residue + i
                            fragment_end = fragment_start + fragment_length - 1
                            new_header = f">{fragment_start}-{fragment_end}|{current_header.strip()}"
                            outfile.write(new_header + "\n")
                            outfile.write(fragment + "\n")
                            
                # Update the current header
                current_header = line.strip()[1:]  # Remove the '>' from the header
                current_sequence = []
            else:
                # Append sequence lines
                current_sequence.append(line.strip())
        
        # Process the last sequence
        if current_header and current_sequence:
            print(current_header)
            full_sequence = ''.join(current_sequence).strip()
            protein_id = current_header.split('|')[1].strip()
            sequence_name = current_header.split('|')[0].strip()
            start_residue = int(sequence_name.split('_')[1])
            end_residue = int(sequence_name.split('_')[2])
            
            # Handle sequences shorter than the fragment length
            if len(full_sequence) < fragment_length:
                fragment_start = start_residue
                fragment_end = start_residue + len(full_sequence) - 1
                new_header = f">{fragment_start}-{fragment_end}|{current_header.strip()}"
                outfile.write(new_header + "\n")
                outfile.write(full_sequence + "\n")
            else:
                for i in range(0, len(full_sequence) - fragment_length + 1, sliding_window):
                    fragment = full_sequence[i:i + fragment_length]
                    fragment_start = start_residue + i
                    fragment_end = fragment_start + fragment_length - 1
                    new_header = f">{fragment_start}-{fragment_end}|{current_header.strip()}"
                    outfile.write(new_header + "\n")
                    outfile.write(fragment + "\n")

# Example usage
file_path = 'IDRs.fasta'
output_file_path = 'IDRs_6aa_frag_1sw.fasta'
fragment_length = 6
sliding_window = 1

# Run the generalized function with the specified fragment length and sliding window
fragment_fasta(file_path, output_file_path, fragment_length, sliding_window)

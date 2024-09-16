import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fragment sequences in a FASTA file into specified lengths with a sliding window (to make overlapping fragments).')
    parser.add_argument('input_file', help='Path to the input FASTA file.')
    parser.add_argument('output_file', help='Path to the output FASTA file.')
    parser.add_argument('fragment_length', type=int, help='Length of the fragments.')
    parser.add_argument('sliding_window', type=int, help='Size of the sliding window to make overlapping fragments.')

    args = parser.parse_args()

    # Run the generalized function with the specified fragment length and sliding window
    fragment_fasta(args.input_file, args.output_file, args.fragment_length, args.sliding_window)

# Example usage:
# python fragment_fasta.py IDRs.fasta IDRs_6aa_frag_1sw.fasta 6 1
# python fragment_fasta.py IDRs.fasta IDRs_10aa_frag_2sw.fasta 10 2
# python fragment_fasta.py IDRs.fasta IDRs_15aa_frag_3sw.fasta 15 3
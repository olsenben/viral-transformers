from Bio import SeqIO
import pandas as pd


def load_wt_aa_sequence(filepath):

    '''load fasta file of wildtype codon sequence and translate it to amino acid sequence.
        returns str of aa sequence.
    '''
    #skip the first line
    wt_nuc_seq = next(SeqIO.parse(filepath, "fasta")).seq

    #translate codon sequence to amino acid sequence
    wt_aa_seq = wt_nuc_seq.translate(to_stop=True)

    return str(wt_aa_seq)

def apply_mutations(wildtype_aa_sequence, mutations_str):
    """applies amino acid mutations at designated position in wt aa sequence.
        
        input:
            wildtype_aa_sequence (str):  wildtype amino acid sequence. Note that the aas are indexed starting at 1
            mutations_str (str): string containing all point mutations in format 'S29H'
            (S to H at position 29). Multiple mutations in one string 'C6F S29H S36D' seperated by a space.

        returns:
            (str): string with wildtype aa sequence with mutations applied. 
    """
    mutations = mutations_str.split(' ')

    seq = list(wildtype_aa_sequence)

    for mut in mutations:
        from_aa = mut[0]
        to_aa = mut[-1]
        pos = int(mut[1:-1]) - 1 # the mutations are indexed starting at 1 not 0
        
        if seq[pos] != from_aa: 
            print(f"error: expected {from_aa} at position {pos +1 }, got {seq[pos]}")
            continue

        seq[pos] = to_aa

    return ''.join(seq)

def process_codon_file(csv_path, wt_sequence_path, output_path):
    """
    processes mutations and applies to wildtype sequence. builds training data based on binding affinity.
    """
    #load data
    df = pd.read_csv(csv_path)
    wt_aa_seq = load_wt_aa_sequence(wt_sequence_path)

    df['mutated_seq'] = df['aa_substitutions'].apply(lambda x: apply_mutations(wt_aa_seq, x))

    output_cols = ["target", "barcode", "aa_subsitutions", 'mutated_seq', "n_aa_substitutions"]

    df[output_cols].to_csv(output_path, index=False)
    print (f"Saved processed data to {output_path}")

def merge_variant_binding_scores(variants_filepath, binding_filepath, output_path):
    """
    Merge varient sequences df with binding df
    """

    variants_df = pd.read_csv(variants_filepath)

    binding_df = pd.read_csv(binding_filepath)

    merged_df = pd.merge(
        variants_df,
        binding_df,
        on=['target', 'barcode']
    )

    merged_df = merged_df.dropna(subset="log10Ka")

    merged_df.to_csv(output_path)
    print(f"saved merged data to {output_path}")
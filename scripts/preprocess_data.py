from src.data import *
import os

"""
Proprocess training data
"""

if __name__ =="__main__":

    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data', 'processed','results')
    codon_counts_path = os.path.join(data_dir, 'variants','codon_variant_table_Wuhan_Hu_1.csv')
    variant_scores_path = os.path.join(data_dir, 'final_variant_scores', 'final_variant_scores.csv')
    wt_sequence_path = os.path.join(current_dir, 'data', 'raw','wildtype_sequence.fasta')
    output_path = os.path.join(current_dir, 'data', 'processed','results', 'variants_with_sequences_Wuhan_Hu_1.csv')

    process_codon_file(codon_counts_path, wt_sequence_path, output_path)

    


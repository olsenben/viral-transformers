from data import process_codon_file, merge_variant_binding_scores
import os

"""
Proprocess training data. After review, I see I only need two files to make this work.
"""

if __name__ =="__main__":

    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data', 'processed','results')
    codon_counts_path = os.path.join(data_dir, 'variants','codon_variant_table_Wuhan_Hu_1.csv')
    #variant_scores_path = os.path.join(data_dir, 'final_variant_scores', 'final_variant_scores.csv')
    wt_sequence_path = os.path.join(current_dir, 'data', 'raw','wildtype_sequence.fasta')
    codon_output_path = os.path.join(current_dir, 'data', 'processed','results', 'variants_with_sequences_Wuhan_Hu_1.csv')
    bc_binding_file = os.path.join(data_dir, 'binding_Kd', 'bc_binding.csv')
    merged_binding_file = os.path.join(current_dir, 'data', 'processed','results', 'variant_binding_sequences_Wuhan_Hu_1.csv')

    #preprocesses and saves the data
    process_codon_file(codon_counts_path, wt_sequence_path, codon_output_path)
    merge_variant_binding_scores(codon_output_path, bc_binding_file, merged_binding_file)




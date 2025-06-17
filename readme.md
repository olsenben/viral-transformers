## Can we predict whether a mutation in the SARS-CoV-2 spike protein is likely to:

- Increase ACE2 binding affinity?

- Escape neutralizing antibodies?

- Persist in the population (i.e., be evolutionarily fit)?

## Data Sources

- GISAID

- ProteinGym

- EMBL EBI PDBe-KB, NExtstrain, or COV-GLUE

## Approach

- Preprocess protien sequences of spike proteins into 3-mer or 6-mer tokens

- Fine-tune ESM-1b or ProtBERT using 
    - classification (beneficial vs neutral vs deleterious mutation)
    - regression (fitness scores, binding energy)

- Optional: Include position-aware embeddings for mutations (attention to RBD)

- Evaluate against CNN or MLP baselines

- Visualize embeddings or attention heatmaps to identify important residues
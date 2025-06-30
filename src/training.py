import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def highlight_rbd_and_ace2(ax, tokens, seq_offset=0, rbd_sites=None, ace2_sites=None):
    for i, token in enumerate(tokens):
        site_num = i + seq_offset  # adjust if your sequence starts at e.g., position 331
        if ace2_sites is not None and site_num in ace2_sites:
            ax.get_xticklabels()[i].set_color("red")
            ax.get_yticklabels()[i].set_color("red")
        elif rbd_sites is not None and site_num in rbd_sites:
            ax.get_xticklabels()[i].set_color("blue")
            ax.get_yticklabels()[i].set_color("blue")

def save_attention_heatmaps(model, tokenizer, batch, output_dir, annot_path, epoch_num=0, max_samples=2):

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    input_ids = batch["input_ids"][:max_samples].to(device)
    attention_mask = batch["attention_mask"][:max_samples].to(device)
    raw = batch.get("raw", [{}] * max_samples)

    # Load annotation
    rbd_sites, ace2_sites = load_annotations(annot_path)

    
    with torch.no_grad():
        outputs = model.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=False,
        )
        attentions = outputs.attentions  # List of [batch, heads, q, k]

    # Use last layer, average across heads
    last_layer_attn = attentions[-1]  # [B, H, L, L]
    avg_attn = last_layer_attn.mean(dim=1)  # [B, L, L]

    for i in range(avg_attn.size(0)):
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        valid_token_mask = [not t.startswith("<") and t != "[PAD]" for t in all_tokens]

        seq_len = attention_mask[i].sum().item()
        valid_indices = [j for j in range(seq_len) if valid_token_mask[j]]
        
        tokens = [all_tokens[i] for i in valid_indices]
        attn_matrix = avg_attn[i][valid_indices, :][:,valid_indices].cpu().numpy()

        if len(tokens) != attn_matrix.shape[0]:
            tokens = tokens[:attn_matrix.shape[0]]


        # Plotting
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
        highlight_rbd_and_ace2(ax, tokens, seq_offset=331, rbd_sites=rbd_sites, ace2_sites=ace2_sites)

        plt.title(f"Attention Heatmap (Epoch {epoch_num}, Sample {i})")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"attention_epoch{epoch_num}_sample{i}.png")
        plt.savefig(fig_path)
        plt.close()

        # Save raw CSV
        attn_df = pd.DataFrame(attn_matrix, index=tokens, columns=tokens)
        csv_path = os.path.join(output_dir, f"attention_epoch{epoch_num}_sample{i}.csv")
        attn_df.to_csv(csv_path)

        # Save summary info
        mutated_seq = raw[i].get("mutated_seq", "")
        log10Ka = raw[i].get("log10Ka", "NA")

        # Example simple sum over attention to RBD/ACE2 tokens
        total_rbd_attention = sum(attn_matrix[j].sum() for j in rbd_sites if j < seq_len)
        total_ace2_attention = sum(attn_matrix[j].sum() for j in ace2_sites if j < seq_len)

        summary_path = os.path.join(output_dir, "attention_summary_all.csv")
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, "a") as f:
            if not file_exists:
                f.write("sample,epoch,total_rbd_attention,total_ace2_attention,log10Ka,mutated_seq\n")
            f.write(f"{i},{epoch_num},{total_rbd_attention},{total_ace2_attention},{log10Ka},{mutated_seq}\n")


def load_annotations(ANNOT_PATH):
    annot_df = pd.read_csv(ANNOT_PATH)

    #extract annotation sides
    rbd_sites = annot_df['site'].to_list()
    ace2_contact_sites = annot_df.loc[annot_df['SARS2_ACE2_contact'] == True, 'site']

    return rbd_sites, ace2_contact_sites

def sum_attention_over_sites(attn_matrix, seq_offset, site_list):
    total_attention = 0
    for i in range(attn_matrix.shape[0]):
        site_i = i + seq_offset
        if site_i in site_list:
            total_attention += attn_matrix[i].sum()
    return total_attention
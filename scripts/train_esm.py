import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import EsmTokenizer, EsmModel
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from model import ESMRegressor
from training import save_attention_heatmaps
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



# tokenization function:
def tokenize_fn(example):
    return tokenizer(
        example['mutated_seq'],
        truncation=True,
        padding='max_length',
        max_length = 1024,
    )

# data collator
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    targets = torch.tensor([item['log10Ka'] for item in batch], dtype=torch.float32)
    return {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask,
        "targets" : targets,
        'raw' : batch
    }

if __name__=="__main__":
    
    # === CONFIG === #
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    DATA_PATH = os.path.join(os.getcwd(), 'data', 'processed','results', 'variant_binding_sequences_Wuhan_Hu_1.csv')
    ANNOT_PATH = os.path.join(os.getcwd(), 'data', 'raw','RBD_sites.csv')
    OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
    EPOCHS = 20
    BATCH_SIZE = 8
    PATIENCE = 5
    CHECKPOINT_PATH = os.path.join(os.getcwd(),'models',"best_model.pt")
    UNFREEZE_LAYERS = ["layers.32", "layers.33"]  # last 2 ESM layers

    #load data
    df = pd.read_csv(DATA_PATH)

    #split data 
    train_df, val_df = train_test_split(df, test_size=.15, random_state=42)

    #convert to hugging face datasets
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    #load model and tokenizer
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    base_model = EsmModel.from_pretrained(MODEL_NAME)

    #tokenize datasets
    train_ds = train_ds.map(tokenize_fn, batched=False)
    val_ds = val_ds.map(tokenize_fn, batched=False)


    #TESTING DATASET
    train_ds =  Subset(train_ds, range(8))
    val_ds =  Subset(val_ds, range(8))

    #data loaders
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)    
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)  

    # === PARTIAL FINE-TUNING === #
    for name, param in base_model.named_parameters():
        param.requires_grad = any(unfreeze in name for unfreeze in UNFREEZE_LAYERS)

    #instantiate model 
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')  
    print("Using device:", device)
    model = ESMRegressor(base_model).to(device)

    #optimizer + loss fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()

    # === TRAINING LOOP === #
    train_losses, val_losses, val_rs = [], [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device).unsqueeze(1)

            preds = model(input_ids, attention_mask)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device).unsqueeze(1)

                preds = model(input_ids, attention_mask)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        val_mse = mean_squared_error(val_targets, val_preds)
        val_r = pearsonr(val_targets.squeeze(), val_preds.squeeze())[0]
        val_losses.append(val_mse)
        val_rs.append(val_r)

        print(f"[Epoch {epoch +1 }] Train MSE: {avg_train_loss:.4f} | Val MSE: {val_mse:.4f} | r: {val_r:.4f}")
    
        # early stopping + checkpointing
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved new best model to {CHECKPOINT_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stoppage triggered at {epoch+1}")
                break
        
        #save attention heatmaps
        for val_batch in val_loader:
            save_attention_heatmaps(model, tokenizer, batch, OUTPUT_DIR, ANNOT_PATH, epoch_num=0, max_samples=2)
            break
            


    # === FINAL PLOTS === #
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_rs, label="Val Pearson r", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("r")
    plt.title("Validation Pearson Correlation")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/training_metrics.png")
    plt.show()

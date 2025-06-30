import torch.nn as nn


class ESMRegressor(nn.Module):
    def __init__(self, base_model, hidden_size=128):
        super().__init__()
        self.base = base_model
        self.regressor = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:,0, :] #[CLS] token
        return self.regressor(cls_embedding)
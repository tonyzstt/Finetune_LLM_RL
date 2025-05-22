import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_layer = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        output = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = output.hidden_states[-1][:, -1, :]
        reward = self.reward_layer(hidden).squeeze(-1)
        return reward

    # Bradley-Terry reward objective
    def loss(self, chosen_reward, rejected_reward):
        assert chosen_reward.shape == rejected_reward.shape
        diff = chosen_reward - rejected_reward
        BTloss = -F.logsigmoid(diff).mean()
        return BTloss
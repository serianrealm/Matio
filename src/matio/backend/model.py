import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def load(self, pretrained, adapter=None):
        pass

    def chat(self):
        pass
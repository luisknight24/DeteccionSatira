import torch
import torch.nn as nn


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_extra_features):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(bert_model.config.hidden_size + num_extra_features, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #cls_output = outputs.pooler_output
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = torch.cat((cls_output, extra_features), dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.softmax(self.fc2(x))
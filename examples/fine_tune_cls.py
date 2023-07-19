from torch.utils.data import Dataset, DataLoader
from models import build_transformer_model
from layers.tokenizers import Tokenizer
from layers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import json
import time
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# fixed seed
pl.seed_everything(3407)

learning_rate = 2e-5
epochs = 50
max_len = 32
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() > 0 else "cpu")

# load model
root_model_path = "./"
vocab_path = root_model_path + "bert-base-chinese/vocab.txt"
config_path = root_model_path + "bert-base-chinese/config.json"
checkpoint_path = root_model_path + 'bert-base-chinese/pytorch_model.bin'


def load_data(filename):
    """
    load data
    each row is (text, label id)
    """
    texts = []
    labels = []
    with open(filename, encoding='utf8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text = l['sentence']
            label = l['label']
            texts.append(text)
            label_int = int(label)
            # transform label to format [0, class_num-1]
            if label_int <= 104:
                labels.append(label_int - 100)
            elif 104 < label_int <= 110:
                labels.append(label_int - 101)
            else:
                labels.append(label_int - 102)
    return texts, labels


# load dataset by specific path
X_train, y_train = load_data('./tnews_public/train.json')
X_test, y_test = load_data('./tnews_public/dev.json')
# setup tokenizer
tokenizer = Tokenizer(vocab_path)


class MyDataset(Dataset):
    '''
    Handle the input data.
    Use the tokenizer to encode sentences into token ids and segment ids.
    '''
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sentence = self.X[index]
        label = self.y[index]
        tokens_ids, segments_ids = tokenizer.encode(sentence, max_len=max_len)
        tokens_ids = tokens_ids + (max_len - len(tokens_ids)) * [0]
        segments_ids = segments_ids + (max_len - len(segments_ids)) * [0]
        tokens_ids_tensor = torch.tensor(tokens_ids)
        segment_ids_tensor = torch.tensor(segments_ids)
        return tokens_ids_tensor, segment_ids_tensor, label


class Model(nn.Module):
    '''
    Setup model by loading weights from function build_transformer_model(config, checkpoint), with_pool=True
    The last layer of the model is a linear layer with output size of 15, indicating a 15-class classification problem.
    '''
    def __init__(self, config, checkpoint):
        super(Model, self).__init__()
        self.model = build_transformer_model(config, checkpoint, with_pool=True)
        for param in self.model.parameters():        # train all layers
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(768, 15)         # pooler


    def forward(self, token_ids, segment_ids):
        encoded_layers, pooled_output = self.model(token_ids, segment_ids)
        # take the CLS token (first position) of the final layer's output
        cls_rep = self.dropout(encoded_layers[:, 0])
        out = self.fc(cls_rep)
        return out
# setup dataset
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
# setup dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# instantiate model
model = Model(config_path, checkpoint_path).to(device)
# define loss function
critertion = nn.CrossEntropyLoss()
# 权重衰减，layernorm层，以及每一层的bias不进行权重衰减
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'layerNorm']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
# AdamW optimizer with weight decay
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# Use a learning rate scheduler with warmup
num_training_steps = (len(train_dataloader) + 1) * epochs
num_warmup_steps = num_training_steps * 0.05
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

total_step = len(train_dataloader)
loss_list = []
test_acc_list = []

best_acc = 0.0
model.train()
# In each epoch, the model is trained on the entire training dataset and then evaluated on the entire test dataset
for epoch in range(epochs):
    start = time.time()
    for i, (token_ids, segment_ids, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids = token_ids.to(device)
        segment_ids = segment_ids.to(device)
        labels = labels.to(device)
        outputs = model(token_ids, segment_ids)
        loss = critertion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.item())
        if (i % 100) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, spend time: {:.4f}'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(), time.time() - start))
            start = time.time()
    model.eval()
    print(len(loss))
    print('end evaluation')
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (token_ids, segment_ids, labels) in enumerate(test_dataloader):
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)
            outputs = model(token_ids, segment_ids)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = correct / total
        test_acc_list.append(test_acc)

        print('Epoch [{}/{}], train_acc: {:.6f}'
              .format(epoch + 1, epochs, test_acc))
    model.train()
print(len(loss))

# Assuming loss_list is a list of losses recorded each epoch or minibatch
plt.plot(loss_list)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch' if len(loss_list) == epochs else 'Minibatch')
plt.show()
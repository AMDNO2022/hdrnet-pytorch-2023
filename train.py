import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from network import FullNet 
from dataset import TrainDataset

print("------------------------------------")
print("* torch.__version__: ", torch.__version__, "*")
print("------------------------------------")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fullnet = FullNet(channel=3)

fullnet.to(device)

rand_loader = DataLoader(dataset=TrainDataset("dataset/train/"), batch_size=1, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(fullnet.parameters(), lr=0.0001)
epoch_num = 100

for epoch in range(epoch_num):
    for i, batch_data in enumerate(rand_loader, 0):
        full, low, gt = list(map(lambda x : x.squeeze(0), batch_data))
        full, low, gt = full.to(device), low.to(device), gt.to(device)

        fullnet.zero_grad()
        output = fullnet(full, low)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()

        if i % 10 == 9:
            print('[epoch: %d / %5d, batch: %5d / %5d] loss: %.4f' % (epoch + 1, epoch_num, i + 1, len(rand_loader), loss.item()))

torch.save({'model_state_dict': fullnet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, 'ckpt/epoch_' + str(epoch_num) + '.tar')

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

if torch.cuda.device_count() > 1:
  print("Use", torch.cuda.device_count(), "GPUs")
  model = nn.DataParallel(model)

fullnet.to(device)

rand_loader = DataLoader(dataset=TrainDataset("dataset/train/"), batch_size=1, shuffle=True)


criterion = nn.MSELoss()
optimizer = optim.Adam(fullnet.parameters(), lr=0.0001)

for epoch in range(2):
    for i, batch_data in enumerate(rand_loader, 0):
        running_loss = 0.0
        for mini in range(batch_data[0].shape[0]):
            full = batch_data[0][mini].squeeze(0)
            low = batch_data[1][mini].squeeze(0)
            gt = batch_data[2][mini].squeeze(0)
            full, low, gt = full.to(device), low.to(device), gt.to(device)

            fullnet.zero_grad()
            output = fullnet(full, low, device)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if i % 200 == 199:
            print('[epoch: %d, batch: %5d / %5d] loss: %.3f' % (epoch + 1, i + 1, len(rand_loader), running_loss / 200))
    torch.save({'epoch': epoch, 'model_state_dict': fullnet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, 'epoch_' + str(epoch + 1) + '.tar')

import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 

class TrainDataset(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_name = os.listdir(root_dir + "gt")
		self.transf = transforms.ToTensor()

	def __len__(self):
		return len(self.img_name) * 5

	def __getitem__(self, idx):
		origin_full = cv2.imread(self.root_dir + "full/" + self.img_name[idx // 5])
		origin_gt = cv2.imread(self.root_dir + "gt/" + self.img_name[idx // 5])

		# random flip & rotate
		flip = random.randint(-2, 1)
		rotate = random.randint(-1, 2)
		if flip != -2:
			origin_full = cv2.flip(origin_full, flip)
			origin_gt = cv2.flip(origin_gt, flip)
		if rotate != -1:
			origin_full = cv2.rotate(origin_full, rotate)
			origin_gt = cv2.rotate(origin_gt, rotate)

		# random crop
		up_left_y = random.randint(0, len(origin_full) - 1024)
		up_left_x = random.randint(0, len(origin_full[0]) - 1024)

		full_img = origin_full[up_left_y : up_left_y + 1024, up_left_x : up_left_x + 1024]
		low_img = cv2.resize(full_img, (256, 256))
		gt_img = origin_gt[up_left_y : up_left_y + 1024, up_left_x : up_left_x + 1024]

		return self.transf(full_img), self.transf(low_img), self.transf(gt_img)



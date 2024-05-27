import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Default values for dataset path and image resizing 
DATASET_PATH = "../dataset"
INPUT_IMAGE_SIZE = (572,572)


# Return the loaded dataset
def load_dataset(batch_size, num_workers=8,shuffle=True,path=DATASET_PATH,size=INPUT_IMAGE_SIZE):
	transform = transforms.Compose([
		transforms.Resize(size),
		transforms.ToTensor(),
	]) # Resize images and convert to tensor

	dataset = ImageFolder(path,transform)
	return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=True)


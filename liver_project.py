import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

# local imports
from utils import model
from img_loader import *

# pytorch imports
import torch
from torch.autograd import Variable
from torchvision import transforms


def train(train_data_dir, val_data_dir, out_weights_dir):
    ct_dir = os.path.join(train_data_dir,'ct')
    seg_dir = os.path.join(train_data_dir,'seg')
    ct_val_dir = os.path.join(val_data_dir,'ct')
    seg_val_dir = os.path.join(val_data_dir,'seg')

    transform = transforms.Compose([ToTensor()])
    train_data = UnetDataset(images_dir=ct_dir, masks_dir=seg_dir,
                             input_size=512, transform=transform)
    trainloader = DataLoader(train_data, batch_size=6, shuffle=True, num_workers=4)

    val_data = UnetDataset(images_dir=ct_val_dir, masks_dir=seg_val_dir,
                           input_size=512, transform=transform)
    valloader = DataLoader(val_data, batch_size=6, shuffle=True, num_workers=4)

    my_model = model(model_name='UNet', args=3)  # in out UNet args mean out class
    my_model.fit(lr=0.001)
    my_model.print_summary(input_size=(1, 512, 512))
    my_model.train(num_epochs=30, trainloader=trainloader, valloader=valloader,
                   epochs_per_save=1, save_dir=out_weights_dir)


def predict(test_data_dir, weights_path, out_seg_dir):
    load_model = model(model_name='UNet', model_path=weights_path, args=3)
    load_model.model.eval()
    all_images = [x for x in sorted(os.listdir(test_data_dir)) if x[-4:] == '.png']  # Read all the images
    transform = transforms.Compose([ToTensor()])
    test_data = UnetDataset(images_dir=test_data_dir, transform=transform)
    loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    os.makedirs(out_seg_dir, exist_ok=True)
    for name, sample in zip(all_images, loader):
        output = load_model.model(Variable(sample['input']).cuda())
        output = output.data.cpu().numpy()
        max_output = np.argmax(output, axis=1) * 127
        seg = max_output.transpose((1, 2, 0))
        cv2.imwrite(out_seg_dir + '\{}'.format(name), seg)
    print('Done')

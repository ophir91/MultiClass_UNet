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
        if torch.cuda.is_available():
            output = load_model.model(Variable(sample['input']).cuda())
        else:
            output = load_model.model(Variable(sample['input']))
        output = output.data.cpu().numpy()
        max_output = np.around(np.argmax(output, axis=1) * 127.3)  # to get 127 for class 1 and 255 for classss 2
        seg = max_output.transpose((1, 2, 0))
        cv2.imwrite(os.path.join(out_seg_dir, name), seg)
        print("saved {}".format(name))
    print('Done')





def calc_accuracy(val_seg_dir, val_pred_dir, label):
    labels = [x for x in sorted(os.listdir(val_seg_dir)) if x[-4:] == '.png']
    preds = [x for x in sorted(os.listdir(val_pred_dir)) if 'seg_'+x[-10:] in labels]
    labels = [x for x in labels if 'ct_'+x[-10:] in preds]
    dc = []
    precision = []
    recall=[]
    for i,(lbl,prd) in enumerate(zip(labels, preds)):
        print(i)
        y_true = cv2.imread(val_seg_dir + '//' + lbl)
        y_true = cv2.cvtColor(y_true, cv2.COLOR_BGR2GRAY)
        y_pred = cv2.imread(val_pred_dir + '//' + prd)
        y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2GRAY)
        y_true_liver = np.zeros_like(y_true)
        y_true_liver[np.where(y_true ==label)]=1
        y_pred_liver=np.zeros_like(y_pred)
        y_pred_liver[np.where(y_pred ==label)]=1
        y_pred_liver =y_pred_liver.flatten()
        y_true_liver = y_true_liver.flatten()

        # # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        # TP = np.sum(np.logical_and(y_true_liver == 1, y_true_liver == 1))
        #
        # # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        # TN = np.sum(np.logical_and(y_true_liver == 0, y_true_liver == 0))
        #
        # # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        # FP = np.sum(np.logical_and(y_true_liver == 1, y_true_liver == 0))
        #
        # # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        # FN = np.sum(np.logical_and(y_true_liver == 0, y_true_liver == 1))
        tn, fp, fn, tp = confusion_matrix(y_true_liver, y_pred_liver, labels=[0,1]).ravel()
        eps=0.00001
        precision.append(tp/(tp+fp+eps))
        recall.append(tp/(tp+fn+eps))
        dc.append(2*tp/(2*tp+fp+fn+eps))
        print('Dice :' + str(np.mean(dc)))
        print('precision :' + str(np.mean(precision)))
        print('recall: ' + str(np.mean(recall)))
    print('Dice :' + str(np.mean(dc))) #need to take care oof nan in mean
    print('precision :' + str(np.mean(precision)))
    print('recall: ' + str(np.mean(recall)))


# path2images = r'..\..\project\TrainData\ct\val'
# seg_dir=r'..\..\project\val_seg'
# weights_path =r'saved_models/Dec26_15-01_UNet_54_Adam_MultiClass_Dice_loss.pth.tar'
# # predict(path2images, weights_path, seg_dir)
# val_seg_dir = r'..\..\project\TrainData\seg\val'
# calc_accuracy(val_seg_dir,seg_dir, label=255)
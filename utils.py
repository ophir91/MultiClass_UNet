import datetime
import os
import numpy as np
#local imports
import losses
import UNet

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchsummary import summary


class model:
    def __init__(self, model_name, model_path=None, args=None):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.writer = None
        self.loss = ''
        self.optimizer_name = ''
        self.accuracy_name = ''
        self.checkpoint_name = ''
        self.optimizer = None
        self.accuracy = None
        self.criterion = None
        self.epoch = 0
        self.create_model(args=args)
        self.create_writer()

    def create_model(self, args):
        if self.model_path is not None:
            print("=> Loading checkpoint '{}'".format(self.model_path))
            self.load_checkpoint(self.model_path, args=args)
        else:
            print("=> Creating new model")
            self.model = getattr(UNet, self.model_name)(args)
        if torch.cuda.is_available():
            print("Using GPU")
            self.model = self.model.cuda()

    def create_writer(self):
        self.checkpoint_name = datetime.datetime.now().strftime('%b%d_%H-%M') + '_' + self.model_name
        writer_dir = os.path.join('runs',self.checkpoint_name)
        self.writer = SummaryWriter(log_dir=writer_dir)

    def fit(self, loss='MultiClass_Dice_loss', optimizer_name='Adam', lr=0.01, weight_decay=0, accuracy_name='MultiClass_Dice_acc'):
        self.loss = loss
        self.optimizer_name = optimizer_name
        self.accuracy_name = accuracy_name
        self.criterion = getattr(losses, loss)()
        self.optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if accuracy_name != '':
            self.accuracy = getattr(losses,accuracy_name)()

    def save_checkpoint(self, save_dir='', filename=None):
        if filename is None:
            filename = '{checkpoint_name}_{epoch_num}_{optimizer}_{loss_name}.pth.tar'\
                .format(checkpoint_name=self.checkpoint_name, epoch_num=self.epoch,
                        optimizer=self.optimizer_name, loss_name=self.loss)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'optimizer_name': self.optimizer_name,
            'accuracy_name': self.accuracy_name
        }, os.path.join(save_dir, filename))

        print("Saved checkpoint as: {}".format(os.path.join(save_dir, filename)))


    def load_checkpoint(self, filename, args):
        """
        loads checkpoint (that was save with save_checkpoint)
        No need to do .fit after
        :param filename: path to the checkpoint
        :return:
        """
        self.model = getattr(UNet, self.model_name)(args)
        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.optimizer_name = checkpoint['optimizer_name']
        self.accuracy_name = checkpoint['accuracy_name']

        self.fit(self.loss, self.optimizer_name, accuracy_name=self.accuracy_name)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("Loaded checkpoint as: {}".format(filename))

    def print_summary(self, input_size=(1, 32, 64, 64)):
        summ = summary(self.model, input_size=input_size)
        # self.writer.add_text('Summary', summ)

    def print_graph(self, dummy_input):
        # dummy_input = Variable(torch.rand(1, 1, 32, 64, 64))
        if self.writer is not None:
            self.writer.add_graph(model=self.model, input_to_model=(dummy_input,))

    def print_epoch_statistics(self, epoch, running_loss, running_accuracy, validation_accuracy=None):
        """
        :param epoch: number of epoch this results from
        :param running_loss: array of all the losses in this epoch
        :param running_accuracy: array of all the training accuracies in this epoch
        :param validation_accuracy: array of all the validation accuracies in this epoch
        :return: print on the stdout the results and log in tensorboard if defined
        """
        if validation_accuracy is None:
            print("End of epoch {:3d} | Training loss = {:5.4f} | Training acc = {:5.4f}"
                  .format(epoch, np.mean(running_loss), np.mean(running_accuracy)))
        else:
            print("End of epoch {:3d} | Training loss = {:5.4f} | Training acc = {:5.4f} | Valid acc =  {:5.4f}"
                  .format(epoch, np.mean(running_loss), np.mean(running_accuracy), np.mean(validation_accuracy)))
        if self.writer is not None:
            self.writer.add_scalar('Train/Loss', float(np.mean(running_loss)), epoch)
            self.writer.add_scalar('Train/accuracy', float(np.mean(running_accuracy)), epoch)
            if validation_accuracy is not None:
                self.writer.add_scalar('Validation/accuracy', float(np.mean(validation_accuracy)), epoch)

    def add_images_tensorboard(self, inputs, labels, outputs):  # TODO: check this function
        """
        :param inputs: the net input, a 5 dim tensor shape: [batch, channels, z, x, y]
        :param labels: the ground truth, a 5 dim tensor shape: [batch, channels, z, x, y]
        :param outputs: the net output, a 5 dim tensor shape: [batch, channels, z, x, y]
        :return: add images to tensorboard
        """
        if len(inputs.shape) == 4:
            self.writer.add_image('epoch ' + str(self.epoch) + '/Input',
                                  torch.round(inputs[0, 0,:, :] * 255))  # [batch, channels, x, y]
            self.writer.add_image('epoch ' + str(self.epoch) + '/GT_0',
                                  torch.round(labels[0, 0,:, :]))  # [batch, channels, x, y]
            self.writer.add_image('epoch ' + str(self.epoch) + '/GT_1',
                                  torch.round(labels[0, 1,:, :]))  # [batch, channels, x, y]
            self.writer.add_image('epoch ' + str(self.epoch) + '/GT_2',
                                  torch.round(labels[0, 2,:, :]))  # [batch, channels, x, y]
            self.writer.add_image('epoch ' + str(self.epoch) + '/output_0',
                                  torch.round(outputs[0, 0,:, :]))  # [batch, channels, x, y]
            self.writer.add_image('epoch ' + str(self.epoch) + '/output_1',
                                  torch.round(outputs[0, 1,:, :]))  # [batch, channels, x, y]
            self.writer.add_image('epoch ' + str(self.epoch) + '/output_2',
                                  torch.round(outputs[0, 2,:, :]))  # [batch, channels, x, y]

    def test_validation(self, validationloader=None):
        validation_accuracy = None
        if validationloader is not None:
            self.model.eval()  # changing to eval mode
            valid_running_accuracy = []
            with torch.no_grad():
                for k, sample in enumerate(validationloader, 0):
                    if sample is dict:
                        valid_inputs = sample['input']
                        valid_labels = sample['ground_truth']
                    else:
                        valid_inputs, valid_labels = sample

                    # wrap them in Variable
                    if torch.cuda.is_available():
                        valid_inputs, valid_labels = Variable(valid_inputs.cuda()), \
                                                     Variable(valid_labels.cuda())
                    else:
                        valid_inputs, valid_labels = Variable(valid_inputs), \
                                                     Variable(valid_labels)

                    valid_outputs = self.model(valid_inputs)
                    acc = self.accuracy(valid_outputs, valid_labels)
                    valid_running_accuracy.append(acc)
            validation_accuracy = np.mean(valid_running_accuracy)
            self.model.train()  # back to train mode
        return validation_accuracy

    def train(self, num_epochs, trainloader, valloader=None, epochs_per_save=1, save_dir='saved_models'):
        print("Start training")
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            self.epoch = epoch
            running_loss = []
            running_accuracy = []
            for i, sample in enumerate(trainloader, 0):
                if isinstance(sample, dict):
                    inputs = sample['input']
                    labels = sample['ground_truth']
                else:
                    inputs, labels = sample

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

                # for loss per epoch
                running_loss.append(loss.item())
                if self.accuracy is not None:
                    # for accuracy per epoch
                    running_accuracy.append(self.accuracy(outputs, labels))
            validation_accuracy = self.test_validation(valloader)
            self.print_epoch_statistics(epoch, running_loss, running_accuracy, validation_accuracy)
            if epoch % epochs_per_save == 0:
                self.save_checkpoint(save_dir=save_dir)
                self.add_images_tensorboard(inputs, labels, outputs)
        self.save_checkpoint(save_dir=save_dir)
        print('='*89)
        print("Finish Training")
        print('='*89)




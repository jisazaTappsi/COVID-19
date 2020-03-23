#!/usr/bin/python

RUN_WITH_CAFFE = True

# Deep Learning Libraries
if RUN_WITH_CAFFE:
    import onnx
    import caffe2.python.onnx.backend as onnx_caffe2_backend
else:
    import torchvision.models as m
    from torchvision import transforms  #datasets
    from torch.autograd import Variable
    import torch
    import torch.nn as nn
    #import torchvision.transforms.functional as F

# Utilities
import numpy as np
import json
import os

# Images
from skimage.transform import resize


"""
 Use model
"""


class CNNModel:

    def __init__(self, use_pretrained=False, num_classes=2):
        self.use_pretrained = use_pretrained
        self.num_classes = num_classes
        print('Initializing...')
        PATH_MODEL = f'{os.getcwd()}/app/checkpoints/chk_resnet_50_epoch_14.pt'

        #if not RUN_WITH_CAFFE:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = m.resnet50(pretrained=use_pretrained)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.num_classes)
        self.model.load_state_dict(torch.load(PATH_MODEL), strict=False)
        # Set Evaluation mode
        self.model.eval()

    def get_transformer(self):
        # Normalize images
        channel_stats = dict(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # Apply Transformations
        eval_transformation = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(**channel_stats)
        ])

        return eval_transformation

    def preprocess(self, image, transformer):
        x = transformer(image)
        image_tensor = transformer(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        return input

    def softmax(self, x):
        return np.exp(x)/sum(np.exp(x))

    def get_label(self, idx):
        with open("app/labels.json", encoding='utf-8', errors='ignore') as json_data:
            labels = json.load(json_data, strict=False)
            return labels[idx]

    @staticmethod
    def normalize(a, mean, std):
        return (a - mean) / std

    def preprocess_without_torch(self, image):
        """
        Does all necessary data wrangling. This is equivalent to self.preprocess(image, self.get_transformer())
        but without torch
        :param image:
        :return:
        """
        nd_image = np.asarray(image)

        # Equivalent to transforms.Resize(256),
        nd_image = resize(nd_image, (256, 256), preserve_range=True)

        # Equivalent to transforms.CenterCrop(224),
        delta = int((256 - 224) / 2)
        nd_image = nd_image[delta:256 - delta, delta:256 - delta]  # centercrop
        nd_image = np.rint(nd_image)

        """
        Equivalent to transforms.ToTensor():
        Converts a PIL Image or numpy.ndarray(H x W x C) in the range [0, 255] to a
        torch.FloatTensor of shape(C x H x W) in the range[0.0, 1.0]
        if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
        or if the numpy.ndarray has dtype = np.uint8
        """
        scaled_nd_image = nd_image / 255
        nd_image = np.transpose(scaled_nd_image, (2, 0, 1))

        # Equivalent to: transforms.Normalize(**channel_stats)
        nd_image[0, :, :] = CNNModel.normalize(nd_image[0, :, :], mean=0.485, std=0.229)
        nd_image[1, :, :] = CNNModel.normalize(nd_image[1, :, :], mean=0.456, std=0.224)
        nd_image[2, :, :] = CNNModel.normalize(nd_image[2, :, :], mean=0.406, std=0.225)
        nd_image = nd_image[np.newaxis, ...]  # the extra dimension
        return nd_image.astype('float32')

    def predict(self, image):
        """
            image: PIL Image
            output: idx, label, score
        """

        if RUN_WITH_CAFFE:

            caffe2_model = onnx.load("app/checkpoints/caffe2_model.proto")

            x = self.preprocess_without_torch(image)
            w = {caffe2_model.graph.input[0].name: x}

            #x = self.preprocess(image, self.get_transformer())
            #w = {caffe2_model.graph.input[0].name: x.data.numpy()}

            prepared_backend = onnx_caffe2_backend.prepare(caffe2_model)
            output = prepared_backend.run(w)[0]
        else:
            x = self.preprocess(image, self.get_transformer())
            output = self.model(x)  # get the output from the last hidden layer of the pretrained model
            if isinstance(output, torch.Tensor):
                output = output.detach().numpy()

        idx = np.argmax(output[0])
        label = self.get_label(idx)
        score = self.softmax(output[0])[idx]

        return idx, label, score

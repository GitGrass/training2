import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import json
import numpy as np
from PIL import Image

# https://aidiary.hatenablog.com/entry/20180212/1518404395
# cu110/torchvision-0.8.2%2Bcu110-cp38-cp38-win_amd64.whl

class Image2Text(object):
    def __init__(self):
        print('creating the estimating model...')
        self.model = models.vgg16(pretrained=True)
        self.model.eval()
        print('the model have been created!')

    def __load_image(self, filename):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        img = Image.open(filename).convert('RGB')
        img_tensor = preprocess(img)
        print(img_tensor.shape)
        img_tensor.unsqueeze_(0)
        print(img_tensor.shape)
        return img_tensor

    def predict(self, filename):
        input = self.__load_image(filename)
        out = self.model(Variable(input))

        class_index = json.load(open('./imagenet_class_index.json', 'r'))
        labels = {int(key):value for (key, value) in class_index.items()}
        print(labels[np.argmax(out.data.numpy())])
        return labels[np.argmax(out.data.numpy())][1]

if __name__ == "__main__":
    estimator = Image2Text()
    name = estimator.predict("./uploads/upload.png")
    print(torch.cuda.is_available())
    print(name)

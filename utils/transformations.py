import numpy as np
import torch

class transform():
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def initialize(self, image, isprocess):
        if isprocess:
            image = image.transpose(2, 0, 1)
            image = image / 255.
            return image
        else:
            image = torch.clamp(image, 0, 1)
            image = image * 255
            return image
        
    def normalize(self, image, isprocess):
        if isprocess:
            for channel, _ in enumerate(image):
                image[channel, :, :] = (image[channel, :, :] - self.mean[channel]) / self.std[channel]
            return image
        else:
            for channel, _ in enumerate(image):
                image[channel, :, :] = (image[channel, :, :] * self.std[channel]) + self.mean[channel]
            return image
        
class process(transform):
    def __init(self):
        super(process, self).__init__()
        
    def __call__(self, image):
        image = self.initialize(image, True)
        image = self.normalize(image, True)
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image).double()
        
        return torch.autograd.Variable(image, requires_grad=True)
    
class deprocess(transform):
    def __init_(self, image):
        super(deprocess, self).__init__()
        
    def __call__(self, image):
        image = copy.copy(image.detach())
        image = self.normalize(image, False)
        image = torch.squeeze(image)
        image = self.initialize(image, False)
        image = image.numpy().astype(np.uint8)
        image = image.transpose(1, 2, 0)
        return image

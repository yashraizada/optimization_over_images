import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import vgg16

from utils.transformations import process, deprocess

class visualize():
    def __init__(self, model, conv_layer, layer_filter):
        self.model = model
        self.conv_layer = conv_layer
        self.layer_filter = layer_filter
        
        self.hook_input = None
        self.hook_output = None
        
    def register_hook(self, forward=True):
        module = self.model[self.conv_layer]
        
        def hook_function(module, grad_in, grad_out):
            self.hook_output = grad_out[0, self.layer_filter]
        
        if forward:
            module.register_forward_hook(hook_function)
        else:
            module.register_backward_hook(hook_function)
    
    def execute(self, epochs, learning_rate):
        # define image to be optimized
        optimization_image = np.random.uniform(150, 180, (224, 224, 3))
        
        # instantiate and process image
        optimization_image = process()(optimization_image)
        
        # register hook
        self.register_hook()
        
        # define optimizer
        optimizer = Adam([optimization_image], lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            # clear grad cache
            optimizer.zero_grad()
            
            layer_input = optimization_image
            
            for index, layer in enumerate(self.model):
                layer_input = layer(layer_input)
                if index == self.conv_layer:
                    break
            
            loss = torch.mean(self.hook_output)
            loss.backward()
            optimizer.step()
            
            # print
            print('Epoch:', epoch, 'loss:', loss)
            
            # deprocess and save image
            output_image = deprocess()(optimization_image)
            plt.imshow(output_image)
            plt.imsave('Image-' + str(epoch) + '.png', output_image)

if __name__ == '__main__':
	# configure visualization
	conv_layer = 17
	layer_filter = 5

	epochs = 50
	learning_rate = 0.8

	# define model
	model = vgg16(pretrained=True).features
	model = model.double()

	viz = visualize(model=model, conv_layer=conv_layer, layer_filter=layer_filter)
	viz.execute(epochs, learning_rate)

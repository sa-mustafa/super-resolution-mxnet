# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:22:56 2019

@author:    https://github.com/sa-mustafa
@purpose:   Runs super resolution sample from MXNet documentation.
"""

import argparse
import mxnet as mx
import numpy as np
from mxnet.gluon import data
from PIL import Image
from collections import namedtuple

# CLI
parser = argparse.ArgumentParser(description='Super-resolution using an efficient sub-pixel convolution neural network.')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor. default is 2.")
parser.add_argument('--use-gpu', action='store_true', help='whether to use GPU.')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--img', type=str, help='input image to use')
opt = parser.parse_args()
print(opt)

mx.random.seed(opt.seed)
ctx = [mx.gpu(0)] if opt.use_gpu else [mx.cpu()]
upscale_factor=2

# Models are downloaded from https://github.com/WolframRhodium/Super-Resolution-Zoo
model_file = 'models/LapSRN_2x' # or 'models/SRGAN_2x'
image_file = 'D:/Downloads/Deep Learning/mxnet/Superresolution/super_res_input.jpg'

# Load the image
img = Image.open(image_file).resize((224, 224));
y, cb, cr = img.convert('YCbCr').split()
image = np.array(y)[np.newaxis, np.newaxis, :, :]

# Load the model
sym, arg, aux = mx.model.load_checkpoint(model_file, 0);
data_names = [graph_input for graph_input in sym.list_inputs()
              if graph_input not in arg and graph_input not in aux]
print(data_names)

mod = mx.mod.Module(symbol=sym, data_names=data_names, label_names=None);
mod.bind(for_training=False, data_shapes=[(data_names[0], image.shape)]);
mod.set_params(arg, aux)
batch = namedtuple('Batch', ['data'])

# Run the model
mod.forward(batch([mx.nd.array(image)]),is_train=False);

# Get the result
output = mod.get_outputs()[0][0][0].asnumpy()
img_out_y = Image.fromarray(np.uint8((output.clip(0, 255)), mode='L'))
out_img_cb = cb.resize(img_out_y.size, Image.BICUBIC)
out_img_cr = cr.resize(img_out_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [img_out_y, out_img_cb, out_img_cr]).convert('RGB')
out_img.save("super_res_output.jpg");

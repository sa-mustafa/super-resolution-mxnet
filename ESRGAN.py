# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:22:56 2019

@author:    https://github.com/sa-mustafa
@purpose:   Runs super resolution method by ESRGAN/SRGAN/DBPN/RCAN model on input images.
            There were virtually no sample code for the above mentioned models on mxnet.
            So I had to create my own!
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
model_file = 'models/EDSR_baseline_x2' # or SRGAN_2x or DBPN_2x or RCAN_BIX2 or ESRGAN_4x
image_file = opt.img;

# Load the image
img = Image.open(image_file).resize((224, 224));
y, cb, cr = img.convert('YCbCr').split()

# Convert the image to tensor
image = mx.nd.array(img).astype(np.float32);
transformer = data.vision.transforms.ToTensor();
image = transformer(image);
# Take the image as a batch
image = image.expand_dims(axis=0)

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
out_img_y = mod.get_outputs()[0][0][0].asnumpy();
out_img_y *= 255.0;
out_img_y = out_img_y.clip(0, 255);
out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

# Save the resulting image
result_img = Image.merge("YCbCr", [out_img_y,
              cb.resize(out_img_y.size, Image.BICUBIC),
              cr.resize(out_img_y.size, Image.BICUBIC)]).convert("RGB");
result_img.save("super_res_output.jpg");

# Created by Baole Fang at 7/23/23

import argparse

import yaml
import os
from PIL import Image

import io
import IPython.display
import numpy as np
import cv2
import PIL.Image

import torch

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator

from dash import Dash, dcc, html, Input, Output, ctx, State
import dash_mantine_components as dmc

import torchvision.transforms as T


def build_generator(model_name):
    """Builds the generator by model name."""
    gan_type = MODEL_POOL[model_name]['gan_type']
    if gan_type == 'pggan':
        generator = PGGANGenerator(model_name)
    elif gan_type == 'stylegan':
        generator = StyleGANGenerator(model_name)
    return generator


def sample_codes(generator, num, latent_space_type='Z', seed=0):
    """Samples latent codes randomly."""
    np.random.seed(seed)
    codes = generator.easy_sample(num)
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
        codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
        codes = generator.get_value(generator.model.mapping(codes))
    return codes


def imshow(images, col, viz_size=256):
    """Shows images in one figure."""
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col

    fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * viz_size
        x = j * viz_size
        if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
        fused_image[y:y + viz_size, x:x + viz_size] = image

    fused_image = np.asarray(fused_image, dtype=np.uint8)
    data = io.BytesIO()
    PIL.Image.fromarray(fused_image).save(data, 'jpeg')
    im_data = data.getvalue()
    disp = IPython.display.display(IPython.display.Image(im_data))
    return disp


def setup(model='stylegan_ffhq', latent_space_type='W', num_samples=5):
    generator = build_generator(model)
    boundary = np.load(os.path.join('boundaries', f'{model}_{latent_space_type.lower()}.npy'))
    latent_codes = sample_codes(generator, num_samples, latent_space_type)
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
        synthesis_kwargs = {'latent_space_type': 'W'}
    else:
        synthesis_kwargs = {}
    images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
    return generator, boundary, latent_codes, images, synthesis_kwargs


def get_app(imgs, slider, targets, H, W):
    app = Dash(__name__)
    images = []
    ids = []
    transform = T.ToPILImage()
    for i,img in enumerate(imgs):
        img=transform(img)
        div = html.Div(
            children=[html.Img(id=f'{i}_orig', src=img, height=H, width=W),
                      html.Img(id=f'{i}_gen', src=img, height=H, width=W)],
            style={'display': 'flex', 'flex-direction': 'column'}
        )
        images.append(div)
        ids.append(f'{i}_gen')
    img_layout = html.Div(children=images, style={'display': 'flex', 'flex-direction': 'row', 'flex-flow': 'row wrap'})
    control = html.Div(
        [
            html.Div([dmc.Slider(id="slider",value=0, min=slider[0], max=slider[1], step=slider[2], precision=2)],
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dmc.Select(data=targets, id="dropdown", value=targets[0], clearable=False, searchable=True)],
                     style={'width': '30%', 'display': 'inline-block'}),
            # html.Div([dcc.Loading(id="loading", children=html.Div(id=ids[0]), type="circle")],
            #          style={'width': '10%', 'display': 'inline-block'})
        ]
    )

    app.layout = html.Div([
        html.H1('Facial attribute slider', style={'textAlign': 'center'}),
        img_layout,
        control,
    ])
    return app, ids


def main(config, port):
    H, W = config['size']
    generator, boundary, latent_codes, images, synthesis_kwargs = setup(config['model'], config['type'],
                                                                        config['samples'])
    app, ids = get_app(images, config['alpha'], config['targets'], H, W)
    transform = T.ToPILImage()
    @app.callback(
        [Output(i, 'src') for i in ids],
        [Input('dropdown', 'value'), Input('slider', 'value')]
    )
    def handler(dropdown, slider):
        new_codes = latent_codes + slider * boundary[config['targets'].index(dropdown)]
        all_imgs = generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']
        imgs = []
        for img in all_imgs:
            img=transform(img)
            imgs.append(img.resize((H, W)))
        return imgs

    app.run_server(port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='slider parameters')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--port', type=int, default=8050, help='port')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(config_, args.port)

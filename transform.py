from pickle import load
import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path
from blend_models import load_pkl

def generate(Gs_blended, latent_file):
    latentf = np.load(latent_file)
    #import pdb; pdb.set_trace()
    latent = np.expand_dims(latentf['dlatents'], axis=0)
    synthesis_kwargs = dict(output_transform=dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
    images = Gs_blended.components.synthesis.run(
        latentf['dlatents'], randomize_noise=False, **synthesis_kwargs)
    Image.fromarray(images.transpose((0, 2, 3, 1))[0], 'RGB').save(
        latent_file.parent / (f"{latent_file.stem}-painted.jpg"))

if __name__ == '__main__':
    dnnlib.tflib.init_tf()
    latent_dir = Path("/home/jma/Projects/stylegan2-ada/SamplePhotos/projected-aligned")
    latents = latent_dir.glob("*.npz")
    _,_,Gs_blended = load_pkl('blended-res32.pkl')

    for latent_file in latents:
        generate(Gs_blended, latent_file)

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np


def getProcessor():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return processor

def getProcImage(imgs: np.ndarray):
    processor = getProcessor()
    size = {'height': 224, 'width': 224}
    proc_img = processor(images=imgs, size = size , return_tensors="pt")
    return proc_img

def prepMeta(dir : str = 'snaps'):
    files = os.listdir(dir)
    meta = []
    for file in files:
        meta.append(file.split('.png')[0].strip())
    return meta

def getClipModel():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch16")


def getEmbeddings(dirMode: bool = False, imageTensors: np.ndarray = None, model = None):
    model = getClipModel() if model is None else model
    if dirMode:
        meta = prepMeta(dir = 'snaps')
        imageNdarray = []
        imageNdarray.extend([Image.open(f'snaps/{m}.png') for m in meta])
        batch_tensor = getProcImage(imageNdarray)
        batch_tensor = batch_tensor.get('pixel_values')
          
    else:
        if imageTensors is None:
            raise Exception('No images found in array')
        batch_tensor = torch.stack(imageTensors)

    inputs = {'pixel_values': batch_tensor}
        
    with torch.no_grad():
        output = model.get_image_features(**inputs)

    documents = []
    for i, m in enumerate(meta):
        documents.append({'id': m, 'vector': output[i].tolist()})
    return documents

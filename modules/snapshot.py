from getEmbeddings import getProcImage, getEmbeddings, getClipModel
from insertVectors import insertVectors
from PIL import Image
import torch
from PIL import ImageGrab
import numpy as np
import time


def getScreen(init: bool = False):
   model = getClipModel()
   while init:
    s1 = ImageGrab.grab(); s1 = np.asarray(s1)
    s2 = ImageGrab.grab(); s2 = np.asarray(s2)

    imgs = np.array([s1, s2])
    procImages = getProcImage(imgs)
    s1p = procImages.get('pixel_values')[0]
    s2p = procImages.get('pixel_values')[1]

    #Rudamentary difference check
    # May need to implement a more robust method like PySceneDetect
    diff = torch.abs(s1p - s2p).sum()
    
    if int(diff) > 2000:
        s2 = Image.fromarray(s2)
        s2.save(f'snaps/{time.time()}.png')
        batch_tensor = s2p
        inputs = {'pixel_values': batch_tensor}
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            document = {'id': time.time(), 'vector': output[0].tolist()}
            insertVectors([document])
            print(f"Inserted {document['id']}")
        
        

    

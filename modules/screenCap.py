from modules.getEmbeddings import getProcImage, getEmbeddings, getClipModel
from modules.insertVectors import insertVectors
from PIL import Image
import torch
from PIL import ImageGrab
import numpy as np
import time
import threading
import os


stop_screenshot_thread = False
screenshot_thread = None

def getScreen():
    global stop_screenshot_thread

    model = getClipModel()
    while not stop_screenshot_thread:
        s1 = ImageGrab.grab(); s1 = np.asarray(s1)
        time.sleep(1)  # Add a delay between screenshots
        s2 = ImageGrab.grab(); s2 = np.asarray(s2)

        imgs = np.array([s1, s2])
        procImages = getProcImage(imgs)
        s1p = procImages.get('pixel_values')[0]
        s2p = procImages.get('pixel_values')[1]

        # Rudimentary difference check, may need to implement a more robust method like PySceneDetect
        diff = torch.abs(s1p - s2p).sum()

        if int(diff) > 2000:
            s2_img = Image.fromarray(s2)
            timestamp = int(time.time())
            if not os.path.exists('snaps'):
                os.makedirs('snaps')
            s2_img.save(f'snaps/{timestamp}.png')
            print(f'Screenshot saved: snaps/{timestamp}.png')

            batch_tensor = s2p
            batch_tensor = torch.stack([batch_tensor])
            inputs = {'pixel_values': batch_tensor}
            with torch.no_grad():
                output = model.get_image_features(**inputs)
                document = {'id': timestamp, 'vector': output[0].tolist()}
                insertVectors([document])
                print(f"Inserted {document['id']}")

def start_screenshot_thread():
    global stop_screenshot_thread
    global screenshot_thread

    stop_screenshot_thread = False
    screenshot_thread = threading.Thread(target=getScreen)
    screenshot_thread.start()
    print("Screenshot thread started.")

def stop_screenshot_thread_func():
    global stop_screenshot_thread
    global screenshot_thread

    stop_screenshot_thread = True
    if screenshot_thread:
        screenshot_thread.join()
    print("Screenshot thread stopped.")

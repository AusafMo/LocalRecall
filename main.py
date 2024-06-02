from modules.Query import dotQuery, getAllVectors, deleteAllVectors
from modules.getEmbeddings import getEmbeddings
from modules.insertVectors import insertVectors
from modules.screenCap import start_screenshot_thread, stop_screenshot_thread_func
import time

s = time()
documents = getEmbeddings(dirMode = True)
print(f"Total documents = {len(documents)}")
insertVectors(documents, pineCone = True)
print(f"time = {time()-s} sec")

query = "image of a white screen"
negative_text = "image of a black screen"
res = dotQuery(inputText = query, negative_text = negative_text,
                topk = 5, includeDistance = True,
                includeVector = False, pineCone = True)

print(f"query = {query}")
print(f"pgvec = {res['pgVector']}")
print(f"pineCone = {res['pineCone']['matches']}")
print(f"time = {time()-s} sec")

print("Deleting all vectors...")
deleteAllVectors()

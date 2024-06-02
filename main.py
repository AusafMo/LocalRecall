from modules.dotQuery import dotQuery
from time import time

s = time()
res = dotQuery(inputText = "snapshot of a white screen with blue and black text ", topk = 2, 
               includeDistance = True, includeVector = False, pineCone = True)

print(res, "\n",time()-s)

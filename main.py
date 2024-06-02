from modules.dotQuery import dotQuery
from time import time

s = time()
res = dotQuery(inputText = "image of a white background", topk = 5, 
               includeDistance = True, includeVector = False, pineCone = True)

print(res, "\n",time()-s)

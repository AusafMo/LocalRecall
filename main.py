from modules.dotQuery import dotQuery
from time import time

s = time()
query = "image of a white screen"
negative_text = "image of a black screen"
res = dotQuery(inputText = query, negative_text = negative_text,
                topk = 5, includeDistance = True,
                includeVector = False, pineCone = True)

print(f"query = {query}")
print(f"pgvec = {res['pgVector']}")
print(f"pineCone = {res['pineCone']['matches']}")
print(f"time = {time()-s} sec")
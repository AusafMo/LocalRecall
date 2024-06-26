from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.getConn import getConections
from modules.getConn import getIndex
from modules.getEmbeddings import getProcessor, getClipModel, prepMeta


def getProcImage(imgs: np.ndarray):
    processor = getProcessor()
    size = {'height': 224, 'width': 224}
    proc_img = processor(images=imgs, size = size , return_tensors="pt")
    return proc_img

def dotQuery(inputText:str = None, ids:list = None, topk = 1, includeDistance = False, 
             includeVector = False, metadataFilter:list = None, inId:list = None, pineCone = False,
             negative_text = None):  
    
    conn, cur = getConections()
    index = getIndex()
    
    model = getClipModel()

    if inputText is None:
        raise Exception('No input text found')
    
    processor = getProcessor()
    inputs = processor(text = inputText, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    if negative_text:
        print('Negative text found')
        inputs = processor(text = negative_text, return_tensors="pt", padding=True)
        neg_features = model.get_text_features(**inputs)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
        text_features = text_features - neg_features

    text_features = text_features.tolist()[0]
    if pineCone:

        pineres = index.query(vector = text_features, top_k = topk, 
                          include_values = includeVector, 
                          include_metadata= True if metadataFilter else False)

    base_query = "SELECT "
    
    # Select fields based on includeDistance and includeVector
    fields = []
    if includeVector:
        fields.append("id, vector")
    else:
        fields.append("id")
    
    if includeDistance:
        # Cosine similarity, higher is closer
        fields.append(f" 1 - (vector <=> '{text_features}'::vector) AS cosine_similarity")
    
    if metadataFilter:
        fields.extend(metadataFilter)
    
    base_query += ", ".join(fields) + " FROM vectors"

    if inId is not None:
        if len(inId) == 1:
            in_clause = f"WHERE id = '{inId[0]}'"
        else:
            in_clause = f"WHERE id IN {tuple(inId)}"
        base_query += " " + in_clause
    
    # Order by distance
    base_query += f" ORDER BY vector <=> '{text_features}'::vector LIMIT {topk}"

    query = base_query
    
    cur.execute(query)
    x = cur.fetchall()

    if includeDistance:
        x = sorted(x, key=lambda x: x[1], reverse=True)

    cur.close(), conn.close()
    
    if pineCone:
        res = {
            'pgVector': x,
            'pineCone': pineres
        }
        return res
    
    return {
        'pgVector': x
    }

def getAllVectors(ids = None):
    conn, cur = getConections()
    if ids is None:
        cur.execute("SELECT * FROM vectors")
        
    elif isinstance(id, list):
        cur.execute(f"SELECT * FROM vectors WHERE id IN {tuple(id)}")
    rows = cur.fetchall()
    
    cur.close(), conn.close()
    return rows

def deleteAllVectors():
    conn, cur = getConections()
    cur.execute("DELETE FROM vectors")
    conn.commit()
    cur.close(), conn.close()
    
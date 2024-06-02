from modules.getConn import getConections
from psycopg2.extras import execute_values
from pinecone import Pinecone
import os

def getIndex():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv('pineKey'))
    index = pc.Index("vectors")
    return index

def insertVectors(documents, insertOne = False, pineCone = False):
    conn, cur = getConections()
    
    if pineCone:
        index = getIndex()
        pineConeDocs = [{'id': doc.get('id'), 'values': doc.get('vector')} for doc in documents]
        index.upsert(pineConeDocs)

    if insertOne:
        cur.execute("INSERT INTO vectors (id, vector, ts) VALUES (%s, %s, %s)", (documents.get('id'), documents.get('vector'), documents.get('id')))
        conn.commit()

    data_list = [(doc.get('id'), doc.get('vector'), doc.get('id')) for doc in documents]
    execute_values(cur, "INSERT INTO vectors (id, vector, ts) VALUES %s", data_list)
    conn.commit()

    cur.close(), conn.close()

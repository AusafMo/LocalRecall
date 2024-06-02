import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
load_dotenv()

## I know its local, so we dont need to worry about the security of the key, but i will use the .env file to store the key anyway
def getConections():
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="postgres",
        user= os.getenv('user'),
        password= os.getenv('password')
    )

    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)
    return conn, cur

#FOR COMPARISION
def getIndex():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.getenv('pineKey'))
    index = pc.Index("vectors")
    return index
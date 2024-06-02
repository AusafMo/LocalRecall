from modules.getConn import getConections
from psycopg2.extras import execute_values

def insertVectors(documents, insertOne = False):
    conn, cur = getConections()
    if insertOne:
        cur.execute("INSERT INTO vectors (id, vector, ts) VALUES (%s, %s, %s)", (documents.get('id'), documents.get('vector'), documents.get('id')))
        conn.commit()

    data_list = [(doc.get('id'), doc.get('vector'), doc.get('id')) for doc in documents]
    execute_values(cur, "INSERT INTO vectors (id, vector, ts) VALUES %s", data_list)
    conn.commit()

    cur.close(), conn.close()

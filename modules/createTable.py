from modules.getConn import getConections


def createTable(name = 'vectors', curr = None, conn = None):
    if curr is None or conn is None:
        conn, cur = getConections()
    else:
        cur = curr
    cur.execute(f'''CREATE TABLE IF NOT EXISTS {name}(
        id VARCHAR PRIMARY KEY,
        ts float8,
        vector vector(512)
    )''')
    conn.commit()

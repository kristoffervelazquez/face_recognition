import psycopg2
import numpy as np
def get_connection():
    return psycopg2.connect(
        dbname="my_database",
        user="my_user",
        password="my_password",
        host="localhost",
        port=5432,
    )

def insert_face_vector(face_id, vector):
    conn = get_connection()
    cursor = conn.cursor()

    # Convierte el vector en una lista si es necesario
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    cursor.execute(
        "INSERT INTO face_data (face_id, face_vector) VALUES (%s, %s)",
        (face_id, vector)
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_all_face_vectors():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT face_id, face_vector FROM face_data")
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convierte los vectores almacenados en listas de Python a arrays de Numpy
    return [(row[0], np.array(row[1])) for row in data]
import cv2
import numpy as np
import dlib

# Ruta a los modelos Dlib
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Carga modelos de Dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

def extract_face_features(image):
    # Convierte la imagen a RGB (Dlib espera imágenes en color)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detecta rostros en la imagen
    faces = detector(rgb_image)

    if len(faces) == 0:
        return None, None

    # Obtiene el primer rostro detectado
    shape = shape_predictor(rgb_image, faces[0])
    face_descriptor = face_recognition_model.compute_face_descriptor(rgb_image, shape, num_jitters=10)

    # Dibuja un marco alrededor del rostro detectado
    for face in faces:
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Color azul para el marco

    return np.array(face_descriptor), image



def compare_faces(face_vector, known_vectors):
    best_match = None
    best_similarity = -1

    for face_id, known_vector in known_vectors:
        # Normaliza ambos vectores
        face_vector = face_vector / np.linalg.norm(face_vector)
        known_vector = known_vector / np.linalg.norm(known_vector)

        # Calcula la similitud coseno
        similarity = np.dot(face_vector, known_vector)

        print(f"Similitud con {face_id}: {similarity}")
        if similarity > best_similarity:  # Similitud más alta es mejor
            if similarity < 0.9:
                continue  # Ignora comparaciones irrelevantes
            best_match = face_id
            best_similarity = similarity

    return best_match, best_similarity


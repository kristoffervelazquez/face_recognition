import cv2
from db import get_all_face_vectors, insert_face_vector
from utils import extract_face_features, compare_faces


# Inicializa el clasificador de rostros
def capture_and_recognize():
    cap = cv2.VideoCapture(0)  # Captura de la cámara (0 para la cámara predeterminada)
    print("Presiona 'q' para salir, 'r' para registrar un rostro.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede acceder a la cámara")
            break

        # Modificación en la llamada a extract_face_features
        face_vector, frame_with_faces = extract_face_features(frame)

        if face_vector is None:
            cv2.imshow("Reconocimiento Facial", frame)  # Si no hay rostro, muestra el frame original
            continue

        # Consulta los rostros conocidos en la base de datos
        known_faces = get_all_face_vectors()
        match, similarity = compare_faces(face_vector, known_faces)



        if match and similarity > 0.9:  # Cambia el umbral
            cv2.putText(frame_with_faces, f"Reconocido: {match}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                        2)
        else:
            cv2.putText(frame_with_faces, "Desconocido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Muestra el video en tiempo real con los marcos y texto
        cv2.imshow("Reconocimiento Facial", frame_with_faces)

        # Captura teclas
        key = cv2.waitKey(1) & 0xFF

        # Presiona 'q' para salir
        if key == ord('q'):
            break

        # Presiona 'r' para registrar un nuevo rostro
        if key == ord('r'):
            face_id = input("Ingresa un ID único para este rostro: ")
            insert_face_vector(face_id, face_vector)
            print(f"Rostro registrado con ID: {face_id}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_recognize()

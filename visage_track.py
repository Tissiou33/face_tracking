import cv2
import mediapipe as mp

# Initialisation Mediapipe
mp_face_mesh = mp.solutions.face_mesh

# Accès webcam
cap = cv2.VideoCapture(0)

# Points d'intérêt (landmarks Mediapipe)
landmark_indices = {
    "left_eye": [33, 133],            # 2 points œil gauche
    "right_eye": [362, 263],          # 2 points œil droit
    "front": [10, 332, 103],          # 3 points front
    "mouth_center": [13],             # Centre bouche
    "mouth_edges": [78, 308],         # Extrémités gauche et droite de la bouche
    "nose": [1],                      # Nez (point central)
    "left_cheek": [234],              # Joue gauche
    "right_cheek": [454],             # Joue droite
    "milieu " : [168 ]
}

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture caméra.")
            break

        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = []
                nose_point = None

                # Récupération et dessin des points
                for key, indices in landmark_indices.items():
                    for idx in indices:
                        lm = face_landmarks.landmark[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        points.append((x, y))
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Vert

                        # Enregistrer le point du nez
                        if key == "nose":
                            nose_point = (x, y)

                # Relier tous les points au nez
                if nose_point:
                    for pt in points:
                        if pt != nose_point:
                            cv2.line(frame, pt, nose_point, (0, 0, 255), 1)

                # Dessiner un rectangle autour de l'ensemble des points
                if points:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 0), 2)

        # Affichage
        cv2.imshow("Tracking visage avec liaisons au nez", frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

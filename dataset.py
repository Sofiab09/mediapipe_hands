import cv2
import mediapipe as mp
import csv
import os

# CONFIGURACIÓN
LABEL = "B"
COUNT = 300
CSV_PATH = "gestures.csv"

# Si no existe el archivo, crea encabezado
if not os.path.exists(CSV_PATH):
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
collected = 0

while cap.isOpened() and collected < COUNT:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # modo espejo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)           
            # Extraer landmarks
            lm = hand_landmarks.landmark
            wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
            row = []
            for i in range(21):
                row.extend([lm[i].x - wrist_x, lm[i].y - wrist_y, lm[i].z - wrist_z])
            row.append(LABEL)

            # Guardar en CSV
            with open(CSV_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            collected += 1
            print(f"{collected}/{COUNT} capturado")

    cv2.putText(frame, f"Capturando: {LABEL} ({collected}/{COUNT})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Dataset", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Captura terminada")

import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse


def preprocess_landmarks(hand_landmarks):
    lm = hand_landmarks.landmark
    wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
    keypoints = []
    for i in range(21):
        keypoints.extend([lm[i].x - wrist_x, lm[i].y - wrist_y, lm[i].z - wrist_z])
    return np.array(keypoints).reshape(1, -1)


def main(args):
    modelo = joblib.load(args.model)
    le = joblib.load(args.labels)
    scaler = joblib.load(args.scaler)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        letra = ""
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = preprocess_landmarks(hand_landmarks)
            keypoints_scaled = scaler.transform(keypoints)
            pred = modelo.predict(keypoints_scaled)[0]
            letra = le.inverse_transform([pred])[0]

        cv2.putText(frame, f"Letra: {letra}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Reconocimiento de Letra", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="modelo.pkl")
    parser.add_argument("--labels", default="labels.pkl")
    parser.add_argument("--scaler", default="scaler.pkl")
    parser.add_argument("--flip", action="store_true", help="Si usaste flip al capturar dataset, usa --flip aquí también")
    args = parser.parse_args()
    main(args)

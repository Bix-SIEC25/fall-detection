import cv2
from ultralytics import YOLO
import torch
import time

# === CONFIG UTILISATEUR ===
MODEL_PATH = "/home/batiste/Documents/AI/runs/detect/train3/weights/best.pt"
CAM_INDEX = 0  # essaie 0 d'abord. Si ça ouvre pas, on testera 1.

# logiques anti faux positifs
MIN_AREA_RATIO = 0.02    # bbox doit faire au moins 2% de l'image
MAX_AREA_RATIO = 0.90    # bbox doit faire au moins 90% de l'image

CONF_THRESHOLD = 0.4     # confiance mini
FRAMES_REQUIRED = 10    # nb de frames d'affilee avant alerte

FALLEN_CLASS_ID = 0      # d'après data.yaml: names: ['Fall-Detected'] -> index 0

def main():
    print("[INFO] Chargement du modèle:", MODEL_PATH)
    try:
        model = YOLO(MODEL_PATH)
        print("[INFO] Modèle chargé OK.")
    except Exception as e:
        print("[ERREUR] Impossible de charger le modèle !")
        print(e)
        return

    print("[INFO] Ouverture caméra avec index", CAM_INDEX)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra avec CAM_INDEX =", CAM_INDEX)
        print("[ASTUCE] Essaie de changer CAM_INDEX = 1 dans le code, ou vérifie ta webcam (permissions).")
        return
    else:
        print("[INFO] Caméra ouverte OK.")

    fallen_counter = 0
    alert_active = False

    print("[INFO] Début boucle vidéo. Appuie sur 'q' pour quitter.")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERREUR] cap.read() a échoué, pas d'image depuis la caméra. J'arrête.")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"[DEBUG] Frame {frame_count} capturée.")

        # inférence YOLO
        results = model(frame)

        det = results[0].boxes
        h, w = frame.shape[:2]
        frame_area = float(w * h)

        found_valid_fallen_person = False

        if det is not None:
            for box in det:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = float(box_w * box_h)
                area_ratio = box_area / frame_area

                if (
                    cls_id == FALLEN_CLASS_ID and
                    conf >= CONF_THRESHOLD and
                    area_ratio >= MIN_AREA_RATIO
                    and area_ratio <= MAX_AREA_RATIO
                ):
                    found_valid_fallen_person = True

                    color = (0, 0, 255)  # rouge BGR
                    label = f"Fall-Detected {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # logique multi-frame
        if found_valid_fallen_person:
            fallen_counter += 1
        else:
            fallen_counter = 0
            alert_active = False

        if fallen_counter >= FRAMES_REQUIRED:
            alert_active = True

        if alert_active:
            text = "ALERTE: PERSONNE AU SOL"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.2
            thickness = 4
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            x, y = 20, 20 + text_h
            # fond rouge rempli derrière le texte pour meilleure lisibilité
            rect_tl = (x - 10, y - text_h - 10)
            rect_br = (x + text_w + 10, y + baseline + 10)
            cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 255), -1)
            # texte blanc sur fond rouge, anti-aliasé
            cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # essaie d'afficher la fenêtre
        cv2.imshow("Fall detection (filtered)", frame)

        # touche 'q' pour quitter
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Fermeture demandée par l'utilisateur ('q').")
            break

        # pour ne pas cramer 100% CPU sur laptop vieux
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Fin du script.")

if __name__ == "__main__":
    main()

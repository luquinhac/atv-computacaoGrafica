import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

CONF_THRESHOLD  = 0.4
IOU_THRESHOLD   = 0.5
SOURCE          = 'imagem.jpg'

is_image = isinstance(SOURCE, str) and SOURCE.lower().endswith(
    ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
)
is_live = isinstance(SOURCE, int)

INFER_WIDTH     = 480
YOLO_SKIP       = 1 if is_image else 2

model = YOLO("yolov8n.pt")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    raise RuntimeError("[Erro] haarcascade_frontalface_default.xml nao encontrado.")

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"[Erro] Nao foi possivel abrir a fonte: {SOURCE}")

cap.set(cv2.CAP_PROP_FPS, 60)

fps_counter  = 0
fps_display  = 0
start_time   = time.time()
frame_count  = 0
last_boxes   = None
last_xyxy    = np.empty((0, 4), int)
last_names   = {}

print("[Sistema] Iniciando... Pressione 'q' na janela para sair.")

def add_header(img, titulo):
    header = np.full((36, img.shape[1], 3), (25, 25, 40), dtype=np.uint8)
    cv2.putText(header, titulo, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)
    return np.vstack([header, img])

def pad_to_height(img, target_h):
    dh = target_h - img.shape[0]
    if dh > 0:
        img = cv2.copyMakeBorder(img, 0, dh, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return img

def resize_panel(img, w, h):
    return cv2.resize(img, (w, h))

def get_face_box(frame_gray, x1, y1, x2, y2):
    roi = frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    faces = face_cascade.detectMultiScale(
        roi,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    return x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh

def classificar_imagem(media_intensidade):
    if media_intensidade > 180:
        return "Clara / Alta luminosidade"
    elif media_intensidade > 100:
        return "Media luminosidade"
    return "Escura / Baixa luminosidade"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            if is_image:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        
        media_blur = float(blur.mean())
        canny_lo   = max(int(media_blur * 0.4), 10)
        canny_hi   = max(int(media_blur * 1.2), 30)
        edges      = cv2.Canny(blur, canny_lo, canny_hi)

        hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_mod = hsv.copy()
        hsv_mod[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.int32) + 80, 0, 255
        ).astype(np.uint8)
        frame_saturado = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)
        canal_v        = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)

        intensidade_media = float(gray.mean())
        tipo_imagem       = classificar_imagem(intensidade_media)

        _, thresh  = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        frame_count += 1
        h_orig, w_orig = frame.shape[:2]

        if frame_count % YOLO_SKIP == 0:
            scale       = INFER_WIDTH / w_orig
            infer_h     = int(h_orig * scale)
            frame_small = cv2.resize(frame, (INFER_WIDTH, infer_h))

            results = model.track(
                frame_small, persist=True,
                conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                verbose=False
            )
            raw_boxes = results[0].boxes

            if len(raw_boxes):
                coords = raw_boxes.xyxy.cpu().numpy().copy()
                coords[:, [0, 2]] /= scale
                coords[:, [1, 3]] /= scale
            else:
                coords = raw_boxes.xyxy.cpu().numpy()

            last_boxes = raw_boxes
            last_xyxy  = coords.astype(int)
            last_names = results[0].names

        boxes = last_boxes
        xyxy  = last_xyxy
        names = last_names

        if boxes is None:
            continue

        fps_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            start_time  = time.time()

        dashboard          = frame.copy()
        count_frame        = len(boxes)
        rostos_encontrados = 0

        for i in range(count_frame):
            cls_id      = int(boxes.cls[i])
            conf        = float(boxes.conf[i])
            x1, y1, x2, y2 = xyxy[i]

            if cls_id == 0:
                face = get_face_box(gray, x1, y1, x2, y2)
                if face is None:
                    continue
                fx1, fy1, fx2, fy2 = face
                cv2.rectangle(dashboard, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                label = f"rosto {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(dashboard,
                              (fx1, max(fy1 - lh - 8, 0)),
                              (fx1 + lw + 4, fy1), (0, 255, 255), -1)
                cv2.putText(dashboard, label, (fx1 + 2, max(fy1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                rostos_encontrados += 1
            else:
                cv2.rectangle(dashboard, (x1, y1), (x2, y2), (255, 165, 0), 2)
                label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(dashboard,
                              (x1, max(y1 - lh - 8, 0)),
                              (x1 + lw + 4, y1), (255, 165, 0), -1)
                cv2.putText(dashboard, label, (x1 + 2, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        overlay = dashboard.copy()
        cv2.rectangle(overlay, (0, 0), (260, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, dashboard, 0.45, 0, dashboard)
        cv2.putText(dashboard, f"FPS: {fps_display}",                    (10, 22),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0),   2)
        cv2.putText(dashboard, f"Objetos: {count_frame}",                (10, 44),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0),   2)
        cv2.putText(dashboard, f"Rostos detectados: {rostos_encontrados}",(10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        cv2.putText(dashboard, f"Int: {intensidade_media:.1f}/255",      (10, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 2)
        cv2.putText(dashboard, tipo_imagem,                              (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        tracking = frame.copy()
        overlay_trk = tracking.copy()
        cv2.rectangle(overlay_trk, (0, 0),
                      (tracking.shape[1], tracking.shape[0]), (15, 15, 30), -1)
        cv2.addWeighted(overlay_trk, 0.25, tracking, 0.75, 0, tracking)

        rostos_rastreados = 0
        has_ids           = boxes.id is not None

        for i in range(count_frame):
            if int(boxes.cls[i]) != 0:
                continue

            x1, y1, x2, y2 = xyxy[i]
            conf            = float(boxes.conf[i])
            obj_id          = int(boxes.id[i]) if has_ids else -1

            face = get_face_box(gray, x1, y1, x2, y2)
            if face is None:
                continue
            fx1, fy1, fx2, fy2 = face
            cv2.rectangle(tracking, (fx1, fy1), (fx2, fy2), (255, 0, 255), 2)

            label = f"ID:{obj_id}  {conf:.2f}" if obj_id >= 0 else f"rosto {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(tracking,
                          (fx1, max(fy1 - lh - 10, 0)),
                          (fx1 + lw + 4, fy1), (255, 0, 255), -1)
            cv2.putText(tracking, label, (fx1 + 2, max(fy1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cx = (fx1 + fx2) // 2
            cy = (fy1 + fy2) // 2
            cv2.circle(tracking, (cx, cy), 5, (255, 0, 255), -1)
            rostos_rastreados += 1

        if rostos_rastreados == 0:
            cv2.putText(tracking, "Nenhuma pessoa detectada",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(tracking, f"Rostos rastreados: {rostos_rastreados}",
                    (10, tracking.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 255), 2)

        dashboard_h = add_header(dashboard, "Dashboard IA  |  Rosto + Objetos")
        tracking_h  = add_header(tracking,  "Tracking IA   |  Somente Rosto")

        h_top       = max(dashboard_h.shape[0], tracking_h.shape[0])
        dashboard_h = pad_to_height(dashboard_h, h_top)
        tracking_h  = pad_to_height(tracking_h,  h_top)

        divider_v = np.full((h_top, 4, 3), (40, 40, 40), dtype=np.uint8)
        top_row   = np.hstack([dashboard_h, divider_v, tracking_h])

        target_w = top_row.shape[1]
        w4       = target_w // 4
        h_bot    = top_row.shape[0] // 2

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        p_edges   = add_header(resize_panel(edges_bgr,      w4, h_bot - 36), "Bordas (Canny)")
        p_thresh  = add_header(resize_panel(thresh_bgr,     w4, h_bot - 36), "Binarizacao")
        p_sat     = add_header(resize_panel(frame_saturado, w4, h_bot - 36), "HSV Saturacao+")
        p_hsv_v   = add_header(resize_panel(canal_v,        w4, h_bot - 36), "HSV Canal V")

        bot_row = np.hstack([p_edges, p_thresh, p_sat, p_hsv_v])
        bot_row = pad_to_height(bot_row, h_bot)
        if bot_row.shape[1] != top_row.shape[1]:
            bot_row = cv2.resize(bot_row, (top_row.shape[1], h_bot))

        divider_h = np.full((4, top_row.shape[1], 3), (40, 40, 40), dtype=np.uint8)
        canvas = np.vstack([top_row, divider_h, bot_row])
        canvas = cv2.copyMakeBorder(canvas, 10, 10, 10, 10,
                                    cv2.BORDER_CONSTANT, value=(15, 15, 30))

        canvas_show = cv2.resize(canvas, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("Sistema Inteligente de Analise de Imagens com IA", canvas_show)

        if is_image:
            print("\n[Info] Imagem processada com sucesso. A janela do OpenCV está aberta.")
            print("[Info] Pressione QUALQUER TECLA (com a janela em foco) para fechá-la e abrir o Histograma.")
            cv2.waitKey(0)
            break
        else:
            key = cv2.waitKey(1 if is_live else 30) & 0xFF
            if key == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

if is_image:
    frame_final = cv2.imread(SOURCE)
    if frame_final is None:
        print(f"[Aviso] Nao foi possivel recarregar '{SOURCE}' para o histograma.")
    else:
        gray_final = cv2.cvtColor(frame_final, cv2.COLOR_BGR2GRAY)
        media      = float(gray_final.mean())
        tipo       = classificar_imagem(media)

        plt.figure(figsize=(8, 4))
        plt.hist(gray_final.ravel(), bins=256, range=(0, 256),
                 color='steelblue', alpha=0.85)
        plt.axvline(media, color='red', linestyle='--',
                    label=f"Media: {media:.1f} | {tipo}")
        plt.title("Histograma de Intensidade")
        plt.xlabel("Intensidade (0 = preto  ·  255 = branco)")
        plt.ylabel("Pixels")
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

        print(f"\nIntensidade media : {media:.1f}/255")
        print(f"Tipo de imagem    : {tipo}")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

# =====================
# CONFIGURAÇÕES
# =====================
CONF_THRESHOLD = 0.4
IOU_THRESHOLD  = 0.5
SOURCE = 'imagem.jpg'

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(SOURCE)
is_video = isinstance(SOURCE, int)

fps_counter = 0
fps_display = 0
total_objetos = 0
start_time = time.time()

print("[Sistema] Iniciando... Pressione 'q' para sair.")

def add_header(img, titulo):
    header = np.full((36, img.shape[1], 3), (25, 25, 40), dtype=np.uint8)
    cv2.putText(header, titulo, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)
    return np.vstack([header, img])

def pad_to_height(img, target):
    dh = target - img.shape[0]
    if dh > 0:
        img = cv2.copyMakeBorder(img, 0, dh, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return img

def get_face_box(x1, y1, x2, y2):
    """Retorna coordenadas do rosto (35% superior da bounding box da pessoa)"""
    rosto_y2 = y1 + int((y2 - y1) * 0.35)
    return x1, y1, x2, rosto_y2

def filtrar_melhor_por_classe(boxes_result):
    """Mantém apenas a detecção com maior confiança por classe"""
    melhores = {}
    names = boxes_result.names if hasattr(boxes_result, 'names') else {}
    for box in boxes_result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id not in melhores or conf > melhores[cls_id][4]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            nome = names.get(cls_id, str(cls_id)) if names else str(cls_id)
            melhores[cls_id] = (x1, y1, x2, y2, conf, cls_id, nome)
    return list(melhores.values())

def classificar_imagem(media_intensidade):
    if media_intensidade > 180:
        return "Clara / Alta luminosidade"
    elif media_intensidade > 100:
        return "Media luminosidade"
    else:
        return "Escura / Baixa luminosidade"

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    # =====================
    # ETAPA 2 — Processamento
    # =====================
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    # =====================
    # ETAPA 3 — HSV
    # =====================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_modificado = hsv.copy()
    hsv_modificado[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int32) + 80, 0, 255).astype(np.uint8)
    frame_saturado = cv2.cvtColor(hsv_modificado, cv2.COLOR_HSV2BGR)
    canal_v = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)

    # =====================
    # ETAPA 4 — Intensidade média
    # =====================
    intensidade_media = gray.mean()
    tipo_imagem = classificar_imagem(intensidade_media)

    # =====================
    # ETAPA 5 — Binarização
    # =====================
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # =====================
    # ETAPA 6a — DASHBOARD IA
    # Pessoa: mostra SOMENTE caixa do rosto (amarelo)
    # Outros objetos: mostra caixa normal com label
    # =====================
    results_det = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    dashboard   = frame.copy()

    deteccoes      = filtrar_melhor_por_classe(results_det[0])
    count_frame    = len(deteccoes)
    total_objetos += count_frame

    fps_counter += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        start_time  = time.time()

    rostos_encontrados = 0

    for (x1, y1, x2, y2, conf, cls_id, nome) in deteccoes:
        if cls_id == 0:
            # Pessoa: desenha SOMENTE o rosto, sem a caixa do corpo
            fx1, fy1, fx2, fy2 = get_face_box(x1, y1, x2, y2)
            cv2.rectangle(dashboard, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            label_rosto = f"rosto {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label_rosto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(dashboard, (fx1, max(fy1 - lh - 8, 0)), (fx1 + lw + 4, fy1), (0, 255, 255), -1)
            cv2.putText(dashboard, label_rosto, (fx1 + 2, max(fy1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            rostos_encontrados += 1
        else:
            # Outros objetos: caixa normal colorida
            cv2.rectangle(dashboard, (x1, y1), (x2, y2), (255, 165, 0), 2)
            label_obj = f"{nome} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label_obj, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(dashboard, (x1, max(y1 - lh - 8, 0)), (x1 + lw + 4, y1), (255, 165, 0), -1)
            cv2.putText(dashboard, label_obj, (x1 + 2, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Overlay métricas
    overlay = dashboard.copy()
    cv2.rectangle(overlay, (0, 0), (260, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, dashboard, 0.45, 0, dashboard)
    cv2.putText(dashboard, f"FPS: {fps_display}",          (10, 22),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0),    2)
    cv2.putText(dashboard, f"Objetos: {count_frame}",      (10, 44),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0),    2)
    cv2.putText(dashboard, f"Total: {total_objetos}",      (10, 66),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200),  2)
    cv2.putText(dashboard, f"Rostos: {rostos_encontrados}",(10, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255),  2)
    cv2.putText(dashboard, f"Int: {intensidade_media:.1f}/255", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 2)
    cv2.putText(dashboard, tipo_imagem,                    (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0),  1)

    # =====================
    # ETAPA 6b — TRACKING IA (somente rosto com ID)
    # =====================
    results_trk = model.track(frame, persist=True, conf=CONF_THRESHOLD,
                               iou=IOU_THRESHOLD, classes=[0], verbose=False)
    tracking = frame.copy()

    overlay_trk = tracking.copy()
    cv2.rectangle(overlay_trk, (0, 0), (tracking.shape[1], tracking.shape[0]), (15, 15, 30), -1)
    cv2.addWeighted(overlay_trk, 0.25, tracking, 0.75, 0, tracking)

    rostos_rastreados = 0

    if results_trk[0].boxes.id is not None:
        ids   = results_trk[0].boxes.id.cpu().numpy()
        xyxy  = results_trk[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results_trk[0].boxes.conf.cpu().numpy()

        melhor_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[melhor_idx]
        obj_id = int(ids[melhor_idx])
        fx1, fy1, fx2, fy2 = get_face_box(x1, y1, x2, y2)

        cv2.rectangle(tracking, (fx1, fy1), (fx2, fy2), (255, 0, 255), 2)
        label = f"ID:{obj_id} rosto"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(tracking, (fx1, max(fy1 - lh - 10, 0)), (fx1 + lw + 4, fy1), (255, 0, 255), -1)
        cv2.putText(tracking, label, (fx1 + 2, max(fy1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2
        cv2.circle(tracking, (cx, cy), 5, (255, 0, 255), -1)
        rostos_rastreados = 1
    else:
        cv2.putText(tracking, "Nenhuma pessoa detectada",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(tracking, f"Rostos rastreados: {rostos_rastreados}",
                (10, tracking.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 255), 2)

    # =====================
    # ETAPA 7 — Canvas final (painel superior + inferior)
    # =====================
    dashboard_h = add_header(dashboard, "Dashboard IA  |  Rosto + Objetos")
    tracking_h  = add_header(tracking,  "Tracking IA   |  Somente Rosto")

    h_top = max(dashboard_h.shape[0], tracking_h.shape[0])
    dashboard_h = pad_to_height(dashboard_h, h_top)
    tracking_h  = pad_to_height(tracking_h,  h_top)

    divider_v = np.full((h_top, 4, 3), (40, 40, 40), dtype=np.uint8)
    top_row   = np.hstack([dashboard_h, divider_v, tracking_h])

    # Painel inferior: 4 sub-painéis
    target_w = top_row.shape[1]
    w4       = target_w // 4
    h_bot    = top_row.shape[0] // 2

    def resize_panel(img, w, h):
        return cv2.resize(img, (w, h))

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

    key = cv2.waitKey(1 if is_video else 30) & 0xFF
    if key == ord('q'):
        break
    if not is_video:
        cv2.waitKey(6000)
        break

# =====================
# HISTOGRAMA FINAL (Etapa 4)
# =====================
cap.release()
frame_final = cv2.imread(SOURCE)
gray_final  = cv2.cvtColor(frame_final, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 4))
plt.hist(gray_final.ravel(), bins=256, range=(0, 256), color='steelblue', alpha=0.85)
plt.title("Histograma de Intensidade — frame final")
plt.xlabel("Intensidade (0=preto · 255=branco)")
plt.ylabel("Pixels")
media = gray_final.mean()
tipo  = classificar_imagem(media)
plt.axvline(media, color='red', linestyle='--', label=f"Media: {media:.1f} | {tipo}")
plt.legend()
plt.tight_layout()
plt.show(block=True)

print(f"\nIntensidade media: {media:.1f}/255")
print(f"Tipo de imagem:    {tipo}")
print(f"Total de objetos detectados: {total_objetos}")

cv2.destroyAllWindows()
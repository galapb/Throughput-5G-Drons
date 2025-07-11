import math
from geopy.distance import geodesic
import os
from ultralytics import YOLO
import cv2
import numpy as np
import os
import Analisis_Distancias

# Parámetros de la cámara (ajustar según tu configuración)
lat_c = 40.7128  # Latitud de la cámara
lon_c = -3.765  # Longitud de la cámara
h_c = 15  # Altura de la cámara (en metros)


# Inicializa el modelo YOLO
model = YOLO("yolo11n.pt")

# Parámetros de la cámara (ajustar según tu configuración)
distancia_focal = 1000  # en píxeles (ajustar según la cámara)
tamaño_real_persona = 1.7  # altura promedio de una persona en metros

def obtener_coordenadas_geograficas(lat_c, lon_c, h_c, distancia, angulo_x, angulo_y):
    """
    Devuelve latitud y longitud estimadas a partir del centro (lat_c, lon_c),
    distancia al objeto, y ángulos horizontales y verticales desde la cámara.
    """

    # Radio de la Tierra (en metros)
    R = 6378137  

    # Cálculo de desplazamientos en metros
    dx = distancia * math.sin(angulo_x)
    dy = distancia * math.sin(angulo_y)

    # Convertir desplazamientos a coordenadas geográficas
    dlat = (dy / R) * (180 / math.pi)
    dlon = (dx / (R * math.cos(math.radians(lat_c)))) * (180 / math.pi)

    # Coordenadas finales
    lat_obj = lat_c + dlat
    lon_obj = lon_c + dlon

    return lat_obj, lon_obj

# Nombre del archivo de video
video_filename = 'video_2.mp4'
output_filename = f"{os.path.splitext(video_filename)[0]}_mapa.txt"

# Abre el archivo para guardar las distancias
with open(output_filename, "w") as distancias_file:
    distancias_file.write("Frame,ID,Distancia(m),Latitud,Longitud\n")  # Encabezado

    # Cargar el video
    cap = cv2.VideoCapture(video_filename)

    if not cap.isOpened():
        print(f"Error al abrir el video {video_filename}.")
        exit()

    # Define el codec y crea el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(f"{os.path.splitext(video_filename)[0]}_processed.mp4", fourcc, fps, (width, height))

    frame_count = 0
    fov_x = math.radians(50)  # FOV horizontal en radianes (ajustar según cámara)
    fov_y = math.radians(70)
    # Procesa cada fotograma del video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Realiza la inferencia sobre el fotograma con tracking
        results = model.track(frame, persist=True)
        
        # Si hay detecciones en el fotograma
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Obtén las cajas delimitadoras
            boxes = results[0].boxes
            
            # Procesa cada caja (objeto detectado)
            for i, box in enumerate(boxes):
                # Verifica si es una persona (clase 0 en YOLO)
                if box.cls.cpu().numpy()[0] == 0:  # 0 es el índice de la clase 'person'
                    # Obtén las coordenadas de la caja
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calcula la altura de la caja en píxeles (para una persona)
                    altura_px = y2 - y1
                    
                    # Calcula la distancia usando la fórmula de la lente
                    distancia = (tamaño_real_persona * distancia_focal) / altura_px
                    
                    ####################################################################################################
                    center_x = frame.shape[1] / 2
                    center_y = frame.shape[0] / 2

                    px = (x1 + x2) / 2
                    py = (y1 + y2) / 2

                    angulo_x = (px - center_x) / center_x * (fov_x / 2)
                    angulo_y = (py - center_y) / center_y * (fov_y / 2)
                    # Calcula las coordenadas geográficas
                    lat_ind, lon_ind = obtener_coordenadas_geograficas(lat_c, lon_c, h_c, distancia, angulo_x, angulo_y)

                    ####################################################################################################


                    # Obtén el ID de tracking si está disponible
                    track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else i
                    
                    # Guarda la distancia en el archivo
                    distancias_file.write(f"{frame_count},{i},{distancia:.2f},{lat_ind:.6f},{lon_ind:.6f}\n")
                    
                    # Dibuja la caja y la distancia en el fotograma
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}, Dist: {distancia:.2f}m", 
                                (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Muestra el número de frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Escribe el fotograma procesado en el archivo de salida
        out.write(frame)
        
        # Muestra el fotograma en una ventana
        cv2.imshow("Video", frame)
        
        # Espera una tecla para continuar o salir (1 ms)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Si presionas 'q' se cerrará el video
            break

    # Libera los recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Procesamiento completo. Distancias guardadas en '{output_filename}'")


result_file =  Analisis_Distancias.Analisis_Distancias_Dron(output_filename)

print(f"Resultado Final. Distancias guardadas en '{result_file}'")
import os
from collections import defaultdict

def Analisis_Distancias_Dron(input_file):

    #Abrimos el archivo para leer las distancias
    with open(input_file, 'r') as f:
            lines = f.readlines()

    # Extraer cabecera y datos -- En el caso de Collision13 - Frame, ID, Distancia
    header = lines[0].strip() # .strip elimina los espacios en blanco al principio y al final
    data_lines = [line.strip() for line in lines[1:] if line.strip()] #cogemos los datos ignorando la cabecera





    ###################################################################################################################
    # PROMEDIO DE DISTANCIAS
    ###################################################################################################################
    # Agrupar distancias por ID
    distancias_por_id = defaultdict(list) # Crea una lista con las distancias de cada id, es como si hicieramos distancias_por_id[1].append(8.5), distancias_por_id[1].append(9.1)...
    latitudes_por_id = defaultdict(list)
    longitudes_por_id = defaultdict(list)

    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) == 5:
            frame, id_, distancia, lat, lon = parts
            if not id_.isdigit():
                print(f"Línea con id no válido: {line.strip()}")
                continue
            try:
                id_ = int(id_)
                distancia = float(distancia)
                lat = float(lat)
                lon = float(lon)

                distancias_por_id[id_].append(distancia)
                latitudes_por_id[id_].append(lat)
                longitudes_por_id[id_].append(lon)

            except ValueError:
                print(f"Línea con valores inválidos: {line.strip()}")
                continue

    # Calcular promedio por ID
    resultados = []
    for id_ in sorted(distancias_por_id.keys()):
        distancias = distancias_por_id[id_]
        lats = latitudes_por_id[id_]
        lons = longitudes_por_id[id_]

        promedio_dist = sum(distancias) / len(distancias)
        promedio_lat = sum(lats) / len(lats)
        promedio_lon = sum(lons) / len(lons)
        
        resultados.append((id_, promedio_dist, promedio_lat, promedio_lon))

    # Crear nombre del archivo de salida
    base_filename = os.path.splitext(input_file)[0].replace('_distances', '')
    result_file = f"{base_filename}_result.txt"

    # Escribir resultados
    # Guardar resultados
    with open(result_file, 'w') as f:
        #f.write("ID,Distancia promedio (m),Latitud media,Longitud media\n")
        for id_, dist, lat, lon in resultados:
            f.write(f"{id_},{dist:.2f},{lat:.6f},{lon:.6f}\n")

    return result_file
        
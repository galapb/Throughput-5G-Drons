import math
import pandas as pd
import numpy as np
import Throughput_SNR_real_modelos_final as Throughput_SNR_real_final
from sklearn.cluster import KMeans
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj
import time

celdas_circulares_radio = 700 #metro
area = (celdas_circulares_radio ^2) * math.pi
throughput_final = 0
throughput_result = 0
tiempo_dron_a_centroide_izquierda_final = 0
tiempo_centroide_izquierda_a_derecha_final = 0
media_tiempo_tx_final = 0
media_tiempo_tx_result = 0
total_usuarios = 0
usuarios_cluster_0 = 0
usuarios_cluster_1 = 0

def plot_clusters(df):
    """
    Plotea un scatterplot de latitud vs longitud, coloreado por el cluster,
    con longitud en el eje Y.
    """
    plt.figure(figsize=(10, 8))
    clusters = df['cluster'].unique()

    for cluster in clusters:
        subset = df[df['cluster'] == cluster]
        # Aquí intercambiamos latitud (X) y longitud (Y)
        plt.scatter(subset['latitud'], subset['longitud'], label=f'Cluster {int(cluster)}', s=100)

    plt.title('Distribución de usuarios por clusters')
    plt.xlabel('Latitud')   # Latitud en el eje X
    plt.ylabel('Longitud')  # Longitud en el eje Y
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()

#Archivo calculadora 5G
print("Ejercicio: Posicion del dron INTERMEDIA.")
v_dron = 15 #km/h
v_dron_mps = v_dron * 1000 / 3600  # 1 km/h = 1000 m / 3600 s
Ganancia_antena = 1

#DATOS -- Va a depender de los vídeos y gente que se encuentre.
#Densidad habitantes. El área será el producto del área que forman los clústeres con su radio.

celdas_circulares_radio = 700 #metro
area = (celdas_circulares_radio ^2) * math.pi
throughput_final = 0
throughput_result = 0
tiempo_dron_a_centroide_izquierda_final = 0
tiempo_centroide_izquierda_a_derecha_final = 0
media_tiempo_tx_final = 0
media_tiempo_tx_result = 0
tiempo_tx_maximo = 0.0 
suma_nuevos_maximos=0
max_iteracion=0 

for i in range(1):
    
    ##########################################################################################################################
    #Generamos datos aleatorios de 15 puntos y hacemos clusterización
    ###########################################################################################################################

    ids = []
    distancias = []
    latitud = []
    longitud = []

    #Abrimos el archivo para leer las distancias
    with open("video_2_mapa_result.txt", 'r') as f:
        for line in f:  # ← iteramos directamente sobre f, sin usar readlines()
            partes = line.strip().split(",")
            if len(partes) == 4:
                id_str, distancia_str, lat_str, lon_str = partes
                try:
                    ids.append(int(id_str))
                    distancias.append(float(distancia_str))
                    latitud.append(float(lat_str))
                    longitud.append(float(lon_str))
                except ValueError:
                    continue
    
    # Tráfico que envían: más variabilidad y posibilidad de 0 con más peso
    valores = np.random.uniform(10, 2048, size=len(ids))
    valores = np.round(valores, 2)
 
    data = {
        'ID': ids,
        'distancia (m)': distancias,
        'latitud' : latitud,
        'longitud' : longitud, 
        'trafico_envian [MB]' : valores
    }
    
    df = pd.DataFrame(data)
     # === 2. Convertir a coordenadas UTM (metros)
    proj = Proj(proj='utm', zone=30, ellps='WGS84')  # Para Madrid
    df['x'], df['y'] = proj(df['longitud'].values, df['latitud'].values)

    # 5. Mostrar el resultado
    print(df)


    #DATOS DRON
    # 1. Generar posición aleatoria del dron
    dron_lat = 40.327
    dron_lon = -3.765
    dron_alt = 15 # Altitud de 15 para poder detectar a los usuarios.

    densidad_usuarios = len(df) / area 
    #print(f"Área total: {area:.2f} metros cuadrados")
    #print(f"Número de usuarios: {len(df)}")
    #print(f"Densidad de usuarios: {densidad_usuarios:.8f} usuarios por metro cuadrado")



    #POTENCIA
    #pire -- para ver cual es la potencia que recibirá.
    #perdidas de propagación.




    # CLUSTERIZACIÓN
    num_clusters = 2

    # Preparar los datos para el clustering
    X = df[['x', 'y']]
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=None).fit(X)
    df['cluster'] = kmeans.labels_

    #MAL -- Calidad de canal se calcula por los BS y UE mediante las señales de referencia
    # Damos una calidad de canal aleatoria.
    #df['calidad_canal'] = np.random.uniform(0, 1, len(df))

    #Centroíde de cada clúster
    centroides = df.groupby('cluster')[['latitud', 'longitud']].mean()

    # Mostrar los centroides
    #print("\nLos centroides de nuestra clusterización son los siguientes:\n",centroides)
    #print ("\n")

    # Mostrar los centroides
    #print("\nLos centroides de nuestra clusterización son los siguientes:\n",centroides)
    #print ("\n")
    usuarios_cluster_0 += len(df[df['cluster'] == 0])
    usuarios_cluster_1 += len(df[df['cluster'] == 1])

    #DATOS DRON
    #Posición del dron central a los dos centroídes
    dron_lat = centroides['latitud'].mean()
    dron_lon = centroides['longitud'].mean()
    densidad_usuarios = len(df) / area 


    ###########################################################################################################################
    #Calculamos distancias de los usuarios a sus centroídes
    ###########################################################################################################################
    def distancia(lat1, lon1, lat2, lon2):
        # Aproximación: 1 grado de latitud/longitud ≈ 111 km
        distancia_metros = np.sqrt(((lat1 - lat2) * 111000)**2 + ((lon1 - lon2) * 111000)**2)
        return round(distancia_metros, 2)  # Distancia en metros, redondeada a 2 decimales
    #Primera distancia que tiene que recorrer: Distancia del dron al centroíde de la izquierda. Calculamos esa distancia:
    distancia_dron_a_centroide_izquierda = distancia(
        dron_lat, dron_lon, centroides.loc[0, 'latitud'], centroides.loc[0, 'longitud']
    )
    tiempo_dron_a_centroide_izquierda = distancia_dron_a_centroide_izquierda / v_dron_mps
    #print(f"Tiempo del dron al centroide de la izquierda: {tiempo_dron_a_centroide_izquierda:.2f} segundos")
    tiempo_dron_a_centroide_izquierda_final +=distancia_dron_a_centroide_izquierda


    #Segunda distancia que tiene que recorrer: Distancia del centroíde izquierda al centróde derecha. Calculamos esa distancia:
    distancia_centroide_izquierda_a_derecha = distancia(
        centroides.loc[0, 'latitud'], centroides.loc[0, 'longitud'],
        centroides.loc[1, 'latitud'], centroides.loc[1, 'longitud']
    )
    tiempo_centroide_izquierda_a_derecha = distancia_centroide_izquierda_a_derecha / v_dron_mps
    #print(f"Tiempo del centroide izquierda al centroide derecha: {tiempo_centroide_izquierda_a_derecha:.2f} segundos")
    tiempo_centroide_izquierda_a_derecha_final += distancia_centroide_izquierda_a_derecha



     # Añadir la columna 'distancia_a_centroide' al DataFrame
    df['distancia_a_centroide (m)'] = distancias
    #print(df)
    #print("")





    ###########################################################################################################################
    #Cálculo Throughput
    ###########################################################################################################################
    #print("\n --> Datos Cálculo Throughput:")
    distancias = df['distancia_a_centroide (m)']
    DD_us="Uplink"
    BW_us = 20 # Mhz define la antena
    SC_us = 15

    # Parámetros del sistema
    Carrier_frequency = 5e9      # Carrier frequency (Hz)
    Height_antenna_BS = 100                 # Height of the BS antenna (m)
    Power_delivered_to_all_antennas_of_BS_dBm = 40                  # Power delivered to all antennas of the BS on a fully allocated grid (dBm)
    Height_antenna_UE = 1.5                # Height of UE antenna (m)
    Noise_figure_UE = 6             # Noise figure of the UE (dB)
    Temperatura_antenna_UE = 290        # Antenna temperature of the UE (K)


    df_throughput = Throughput_SNR_real_final.Cal_throughput(
        distancias,
        DD_us,
        BW_us,
        SC_us,
        Carrier_frequency,
        Height_antenna_BS,
        Power_delivered_to_all_antennas_of_BS_dBm,
        Height_antenna_UE,
        Noise_figure_UE,
        Temperatura_antenna_UE 

    )

    df = pd.concat([df, df_throughput], axis=1)
    #print("\n\n\n --> Resultados Obtenidos Finales")
    #print(df)

    # Elimina unidades como ' Mbps' y convierte a float
    df['Throughput'] = df['Throughput'].str.replace(' Mbps', '', regex=False).astype(float)

    # Calculamos media
    throughput_medio = round(df['Throughput'].mean(),2)
    print("--> Throughput medio", throughput_medio, " Mbps")
    throughput_final += throughput_medio
    #print( throughput_final)
    #Calculamos tiempo de tx = Throughput x Tráfico_envían
    df['Tiempo_tx (s)'] = (df['trafico_envian [MB]']*8)/ df['Throughput']
    print(df)
    # Calcular la media del tiempo de transmisión
    media_tiempo_tx = df['Tiempo_tx (s)'].mean()
    #print(f"Media del Tiempo_tx: {media_tiempo_tx:.2f}")

    max_iteracion = df['Tiempo_tx (s)'].max()

    if max_iteracion > tiempo_tx_maximo:
        tiempo_tx_maximo = max_iteracion
        print(f"tiempo_tx_maximo: {tiempo_tx_maximo:.2f}")
        suma_nuevos_maximos += max_iteracion  # sumar nuevos máximos
        print(f"suma_nuevos_maximos: {suma_nuevos_maximos:.2f}")
        max_iteracion +=1
    ###########################################################################################################################
    #Cálculo medio del throughput, en 100 iteraciones.
    ###########################################################################################################################
throughput_result =  round(throughput_final/1, 3)
print( "\nConclusión Throughput:", throughput_result, " Mbps")

media_tiempo_tx_result = round(media_tiempo_tx_final/1, 2)
tiempo_total = round(media_tiempo_tx_result /60, 2)
print("Tiempo Tardaríamos en tx:", tiempo_total, "min")

# Imprimir el máximo después de todas las iteraciones
print("El valor máximo de 'Tiempo_tx (s)' fue:", tiempo_tx_maximo)
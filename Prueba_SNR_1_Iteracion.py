""
# `Prueba 3_1 Con SNR Real`, 1 Iteración
""

import math
import pandas as pd
import numpy as np
import Throughput_SNR_real_final
from sklearn.cluster import KMeans




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
distancia_total_dron_final = 0
for i in range(1):
    
    ##########################################################################################################################
    #Generamos datos aleatorios de 15 puntos y hacemos clusterización
    ###########################################################################################################################


    # Generar datos de ejemplo (15 puntos en Leganés)
    #np.random.seed(42) No queremos una semilla, esta que sea aleatoria
    latitudes = 40.327 + np.random.rand(15) * 0.01
    longitudes = -3.765 + np.random.rand(15) * 0.01
    valores = np.arange(0, 2.01, 0.01)  # 2.01 para incluir 2.00 en el rango
    trafico_envian = np.random.choice(valores, 15)

    data = {
        'latitud': latitudes,
        'longitud': longitudes,
        'trafico_envian': trafico_envian
    }

    df = pd.DataFrame(data)

    # Número de clústeres
    num_clusters = 3

    # Preparar los datos para el clustering
    X = df[['latitud', 'longitud', 'trafico_envian']]
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    df['cluster'] = kmeans.labels_

    #MAL -- Calidad de canal se calcula por los BS y UE mediante las señales de referencia
    # Damos una calidad de canal aleatoria.
    #df['calidad_canal'] = np.random.uniform(0, 1, len(df))

    #Centroíde de cada clúster
    centroides = df.groupby('cluster')[['latitud', 'longitud']].mean()

    # Mostrar los centroides
    #print("\nLos centroides de nuestra clusterización son los siguientes:\n",centroides)
    #print ("\n")



    ###########################################################################################################################
    #Calculamos distancias de los usuarios a sus centroídes
    ###########################################################################################################################
    # Función para calcular la distancia
    def distancia(lat1, lon1, lat2, lon2):
        # Aproximación: 1 grado de latitud/longitud ≈ 111 km
        distancia_metros = np.sqrt(((lat1 - lat2) * 111000)**2 + ((lon1 - lon2) * 111000)**2)
        return round(distancia_metros, 2)  # Distancia en metros, redondeada a 2 decimales
    # Crear una lista vacía para almacenar las distancias, y luego añadimos esta columna a nuestro dataframe
    distancias = []
    for index, row in df.iterrows():
        cluster_id = row['cluster']  # Obtener el ID del clúster del usuario
        centroid_lat = centroides.loc[cluster_id, 'latitud']
        centroid_lon = centroides.loc[cluster_id, 'longitud']

        distancia_al_centroide = distancia(row['latitud'], row['longitud'], centroid_lat, centroid_lon)
        distancias.append(distancia_al_centroide)


    # Añadir la columna 'distancia_a_centroide' al DataFrame
    df['distancia_a_centroide (m)'] = distancias
    #print(df)
    #print("")


    
    ###########################################################################################################################
    #Recopilamos Datos
    ###########################################################################################################################

    #DATOS DRON
    #Posición del dron central a los centroídes
    dron_lat = centroides['latitud'].mean()
    dron_lon = centroides['longitud'].mean()
    densidad_usuarios = len(df) / area 
    #print(f"Área total: {area:.2f} metros cuadrados")
    #print(f"Número de usuarios: {len(df)}")
    #print(f"Densidad de usuarios: {densidad_usuarios:.8f} usuarios por metro cuadrado")




    # Inicializar variables
    tiempo_total_dron = 0
    distancia_total_dron = 0

    # Lista con el orden de visita
    orden_visita = [0, 1, 2]

    # Ir del dron al primer centroide
    primero = orden_visita[0]
    distancia_inicial = distancia(dron_lat, dron_lon, centroides.loc[primero, 'latitud'], centroides.loc[primero, 'longitud'])
    tiempo_total_dron += distancia_inicial / v_dron_mps
    distancia_total_dron += distancia_inicial

    # Recorrer los centroídes uno tras otro
    for i in range(len(orden_visita) - 1):
        origen = orden_visita[i]
        destino = orden_visita[i + 1]
        
        d = distancia(
            centroides.loc[origen, 'latitud'], centroides.loc[origen, 'longitud'],
            centroides.loc[destino, 'latitud'], centroides.loc[destino, 'longitud']
        )
        tiempo_total_dron += d / v_dron_mps
        distancia_total_dron += d

    distancia_total_dron_final += tiempo_total_dron




    ###########################################################################################################################
    #Cálculo Throughput
    ###########################################################################################################################
    #print("\n --> Datos Cálculo Throughput:")
    distancias = df['distancia_a_centroide (m)']
    DD_us="Uplink"
    BW_us = 20 # Mhz define la antena
    SC_us = 15

    # Parámetros del sistema
    Carrier_frequency = 3.5e9      # Carrier frequency (Hz)
    Height_antenna_BS = 25                 # Height of the BS antenna (m)
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



    ################################################################################################################
    #Calculamos tiempo de tx = Throughput x Tráfico_envían




    # Calculamos media
    throughput_medio = round(df['Throughput'].mean(),2)
    #print("--> Throughput medio", throughput_medio, " Mbps")
    throughput_final += throughput_medio
    #print( throughput_final)


    
    ################################################################################################################
    #Calculamos tiempo de tx = Throughput x Tráfico_envían
    df['Tiempo_tx (s)'] = df['trafico_envian'] * df['Throughput']
    
    
    print(df)
    # Calcular la media del tiempo de transmisión
    media_tiempo_tx = df['Tiempo_tx (s)'].mean()
    #print(f"Media del Tiempo_tx en segundos: {media_tiempo_tx:.2f}")


    media_tiempo_tx_final += media_tiempo_tx 
    ###########################################################################################################################
    #Cálculo medio del throughput, en 100 iteraciones.
    ###########################################################################################################################
throughput_result =  round(throughput_final/1, 3)
print( "\nConclusión Throughput:", throughput_result, " Mbps")


distancia_total_dron_result = distancia_total_dron_final/100
media_tiempo_tx_result = round(media_tiempo_tx_final/100, 2)

tiempo_total_tx = round(media_tiempo_tx_result/100, 2)
#print("Tiempo Tardaríamos en tx:", tiempo_total_tx/60, "min")

#print("Conclusión Tiempo Añadido de desplazamiento -Gasto Recuros-:", media_tiempo_tx_result/60, "min")

#tiempo_dron_a_centroide_izquierda_result = round (tiempo_dron_a_centroide_izquierda_final/100, 2)
#tiempo_centroide_izquierda_a_derecha_result= round (tiempo_centroide_izquierda_a_derecha_final/100, 2)
#tiempo_total = round ( (tiempo_dron_a_centroide_izquierda_result + tiempo_centroide_izquierda_a_derecha_result)/60 ,2)
#print("Conclusión Tiempo Añadido de desplazamiento -Gasto Recuros-:", tiempo_total, "min")


#Tiempo medio
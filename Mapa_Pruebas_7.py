import math
import pandas as pd
import numpy as np
import Throughput_SNR_real_modelos_final as Throughput_SNR_real_final
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pyproj import Proj
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import time
start = time.perf_counter()
def generar_df(seed=None, n_usuarios=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Si no se especifica el número de usuarios, lo generamos aleatorio
    if n_usuarios is None:
        n_usuarios = np.random.randint(10, 21)  # Valor aleatorio si no se especifica
    
    # Generar datos pseudoaleatorios en función de la semilla, NO del número de usuarios
    latitudes = 40.3327973 + np.random.rand(n_usuarios) * 0.005
    longitudes = -3.7653449 + np.random.rand(n_usuarios) * 0.005
    valores = np.random.uniform(10, 2048, size=n_usuarios)
    valores = np.round(valores, 2)

    df = pd.DataFrame({
        'latitud': latitudes,
        'longitud': longitudes,
        'trafico_envian': valores
    })
    # === 2. Convertir a coordenadas UTM (metros)
    proj = Proj(proj='utm', zone=30, ellps='WGS84')  # Para Madrid
    df['x'], df['y'] = proj(df['longitud'].values, df['latitud'].values)

    return df

celdas_circulares_radio = 300 #metro
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

for i in range(1):

 
    df = generar_df(seed=i, n_usuarios=None) 
    # Número de clústeres
    num_clusters = 2

    # Preparar los datos para el clustering
    X = df[['x', 'y']]

    #', 'longitud']]
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    df['cluster'] = kmeans.labels_

    #MAL -- Calidad de canal se calcula por los BS y UE mediante las señales de referencia
    # Damos una calidad de canal aleatoria.
    #df['calidad_canal'] = np.random.uniform(0, 1, len(df))

    #Centroíde de cada clúster
    centroides = df.groupby('cluster')[['latitud', 'longitud']].mean()

    # Mostrar los centroides
    print("\nLos centroides de nuestra clusterización son los siguientes:\n",centroides)
    print ("\n")
 
    #Centroíde de cada clúster
    centroides = df.groupby('cluster')[['latitud', 'longitud']].mean()

    # Mostrar los centroides
    print("\nLos centroides de nuestra clusterización son los siguientes:\n",centroides)
    print ("\n")



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
        if row['cluster'] == 0:
            centroid_lat = centroides.loc[0, 'latitud']
            centroid_lon = centroides.loc[0, 'longitud']
        else:  # row['cluster'] == 1
            centroid_lat = centroides.loc[1, 'latitud']
            centroid_lon = centroides.loc[1, 'longitud']

        distancia_al_centroide = distancia(row['latitud'], row['longitud'], centroid_lat, centroid_lon)
        distancias.append(distancia_al_centroide)


    # Añadir la columna 'distancia_a_centroide' al DataFrame
    df['distancia_a_centroide (m)'] = distancias
    print(df)
    print("")


    ###################### Cluster 0 #####################
    print("\n --> Datos generales:")
    # Obtener los datos del cluster 0
    cluster_0_data = df[df['cluster'] == 0]
    # Encontrar la distancia máxima
    max_distancia = cluster_0_data['distancia_a_centroide (m)'].max()
    # El tamaño del área a cubrir es la distancia máxima
    print(f"El usuario más lejano en el cluster 0 se encuentra a una distancia: {max_distancia:.2f} metros")

    ###################### Cluster 1 #####################

    cluster_1_data = df[df['cluster'] == 1]
    # Encontrar la distancia máxima
    max_distancia_1 = cluster_1_data['distancia_a_centroide (m)'].max()
    # El tamaño del área a cubrir es la distancia máxima
    print(f"El usuario más lejano en el cluster 1 se encuentra a una distancia: {max_distancia_1:.2f} metros")


    #Asumimos que la cobertura que puede dar el dron es de 10 metros.
    #Calculamos el throughput a cada uno de los usarios.


    ###########################################################################################################################
    #Recopilamos Datos
    ###########################################################################################################################

    #DATOS DRON
    #Posición del dron central a los dos centroídes
    dron_lat = centroides['latitud'].mean()
    dron_lon = centroides['longitud'].mean()
    v_dron = 15 #km/h
    v_dron_mps = v_dron * 1000 / 3600  # 1 km/h = 1000 m / 3600 s
    Ganancia_antena = 1

    #DATOS -- Va a depender de los vídeos y gente que se encuentre.
    #Densidad habitantes. El área será el producto del área que forman los clústeres con su radio.

    celdas_circulares_radio = 300 #metro
    area = (celdas_circulares_radio **2) * math.pi
    densidad_usuarios = len(df) / area 
    print(f"Área total: {area:.2f} metros cuadrados")
    print(f"Número de usuarios: {len(df)}")
    print(f"Densidad de usuarios: {densidad_usuarios:.8f} usuarios por metro cuadrado")



    #POTENCIA
    #pire -- para ver cual es la potencia que recibirá.
    #perdidas de propagación.



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



    ###########################################################################################################################
    #Cálculo Throughput
    ###########################################################################################################################
    #print("\n --> Datos Cálculo Throughput:")
    distancias = df['distancia_a_centroide (m)']
    DD_us="Downlink"
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
    df['Tiempo_tx (s)'] = (df['trafico_envian']*8)/ df['Throughput']
    #print(df)
    # Calcular la media del tiempo de transmisión
    media_tiempo_tx = df['Tiempo_tx (s)'].mean()
    #print(f"Media del Tiempo_tx: {media_tiempo_tx:.2f}")


    media_tiempo_tx_final += media_tiempo_tx 
    #Graficamos en mapa
    # Crear un mapa centrado en el promedio de los centroides
    mapa = folium.Map(location=[df['latitud'].mean(), df['longitud'].mean()], zoom_start=15)

    # Obtener una lista de colores para los clusters
    num_unique_clusters = df['cluster'].nunique() # Use nunique() for robustness
    colormap = cm.get_cmap('tab10', num_unique_clusters) # Use tab10 or another suitable cmap
    colores_mapa = [mcolors.rgb2hex(colormap(i)[:3]) for i in range(num_unique_clusters)]

    # Graficar los puntos de cada clúster
    for _, row_user in df.iterrows(): # Renamed row to row_user to avoid conflict
        folium.CircleMarker(
            location=(row_user['latitud'], row_user['longitud']),
            radius=5,
            color=colores_mapa[int(row_user['cluster'])],
            fill=True,
            fill_color=colores_mapa[int(row_user['cluster'])],
            fill_opacity=0.6,
            popup=f"Cluster {row_user['cluster']}, Dist: {row_user['distancia_a_centroide (m)']:.2f}m"
        ).add_to(mapa)

    # Graficar los centroides
    for cluster_idx, row_centroid in centroides.iterrows(): # Renamed row to row_centroid
        folium.Marker(
            location=(row_centroid['latitud'], row_centroid['longitud']),
            popup=f"Centroide {cluster_idx}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(mapa)

    # Mostrar el mapa (guardar en un archivo HTML y abrirlo en el navegador)
    mapa.save("mapa_clusters_folium.html")
    print("\nMapa de Folium guardado como 'mapa_clusters_folium.html'. Ábrelo en tu navegador.")
    ###########################################################################################################################
    #Cálculo medio del throughput, en 100 iteraciones.
    ###########################################################################################################################

    # Plot de los usuarios con sus clústeres y centroides
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colores para cada cluster
    colores = ['red', 'blue']

    # Graficamos cada clúster por separado
    for i in range(num_clusters):
        cluster_data = df[df['cluster'] == i]
        ax.scatter(cluster_data['longitud'], cluster_data['latitud'], 
                c=colores[i], label=f'Cluster {i}', alpha=0.6)

        # Graficar centroides
        ax.scatter(centroides.loc[i, 'longitud'], centroides.loc[i, 'latitud'], 
                c='black', marker='X', s=200, label=f'Centroide {i}')

        # Dibujar un círculo con radio = distancia máxima para mostrar el alcance del clúster
        radio_circulo = 300

        circulo = plt.Circle(
            (centroides.loc[i, 'longitud'], centroides.loc[i, 'latitud']),
            radio_circulo / 111000,  # Convertir metros a grados aproximados (1 grado ≈ 111km)
            color=colores[i], fill=False, linestyle='--', linewidth=1.5
        )
        ax.add_patch(circulo)

    # Ajustes de gráfico
    ax.set_title('Distribución Geográfica de Usuarios y Centroides')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.legend()
    ax.grid(True)
    plt.show()
throughput_result =  round(throughput_final/1, 3)
print( "\nConclusión Throughput:", throughput_result, " Mbps")

media_tiempo_tx_result = round(media_tiempo_tx_final/1, 2)
tiempo_total = round(media_tiempo_tx_result /60, 2)
print("Tiempo Tardaríamos en tx:", tiempo_total, "min")


tiempo_dron_a_centroide_izquierda_result = round(tiempo_dron_a_centroide_izquierda_final/1, 2)
tiempo_centroide_izquierda_a_derecha_result= round(tiempo_centroide_izquierda_a_derecha_final/1, 2)
tiempo_total = round((tiempo_dron_a_centroide_izquierda_result + tiempo_centroide_izquierda_a_derecha_result)/60 ,2)
print("Conclusión Tiempo Añadido de desplazamiento -Gasto Recuros-:", tiempo_total, "min")

end = time.perf_counter()
print(f"Tiempo de ejecución: {end - start} segundos")

# Al terminar todas las simulaciones
promedio_total_usuarios = total_usuarios / 15
promedio_cluster_0 = usuarios_cluster_0 / 15
promedio_cluster_1 = usuarios_cluster_1 / 15

print(f"Promedio total de usuarios por simulación: {promedio_total_usuarios:.2f}")
print(f"Promedio de usuarios en el clúster 0 por simulación: {promedio_cluster_0:.2f}")
print(f"Promedio de usuarios en el clúster 1 por simulación: {promedio_cluster_1:.2f}")

#Tiempo medio 
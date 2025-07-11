import pandas as pd
import numpy as np
import sys
def Obtener_MCS(SINR, distancias):
    SINR = np.array(SINR)  # Este es el SINR objetivo
    distancias = np.array(distancias)
    datos = []
    for i in range(len(SINR)):  # Usamos range para obtener los índices
        #print(f"Índice {i}: {SINR[i]}")
        SINR_calc = SINR[i]
        if SINR_calc < -9.478:
            cqi = 1
            modulation = "QPSK"
            Q= 2
            Rmx = 78/1024
        elif SINR_calc < -6.658:
            cqi = 2
            modulation = "QPSK"
            Q= 2
            Rmx = 120/1024
            
        elif SINR_calc < -4.098:
            cqi = 3
            modulation = "QPSK"
            Q= 2
            Rmx = 193/1024
        elif SINR_calc < -1.798:
            cqi = 4
            modulation = "QPSK"
            Q= 2
            Rmx = 308/1024
        elif SINR_calc < 0.399:
            cqi = 5
            modulation = "QPSK"
            Q= 2
            Rmx = 449/1024
        elif SINR_calc < 2.424:
            cqi = 6
            modulation = "QPSK"
            Q= 2
            Rmx = 602/1024
        elif SINR_calc < 4.489:
            cqi = 7
            modulation = "QPSK"
            Q= 4
            Rmx = 378/1024
        elif SINR_calc < 6.367:
            cqi = 8
            modulation = "16QAM"
            Q= 4
            Rmx = 490/1024
        elif SINR_calc < 8.456:
            cqi = 9
            modulation = "16QAM"
            Q= 4
            Rmx = 616/1024
        elif SINR_calc < 10.266:
            cqi = 10
            modulation = "64QAM"
            Q= 6
            Rmx = 466/1024
        elif SINR_calc < 12.218:
            cqi = 11
            modulation = "64QAM"
            Q= 6
            Rmx = 567/1024
        elif SINR_calc < 14.122:
            cqi = 12
            modulation = "64QAM"
            Q= 6
            Rmx = 666/1024
        elif SINR_calc < 15.849:
            cqi = 13
            modulation = "64QAM"
            Q= 6
            Rmx = 772/1024
        elif SINR_calc < 17.786:
            cqi = 14
            modulation = "64QAM"
            Q= 6
            Rmx = 873/1024
        elif SINR_calc < 19.809:
            cqi = 15
            modulation = "64QAM"
            Q= 6
            Rmx = 948/1024
        else: #Asumimos que será mayor de 19.809
            cqi = 15
            modulation = "256QAM"
            Q= 8
            Rmx = 948/1024
        # Imprimir resultados
        datos.append([SINR_calc, Q, modulation, Rmx, cqi, distancias[i]])

        #print(f"SINR objetivo: {SINR_calc} dB")
        #print(f"CQI seleccionado: {cqi}")
        #print(f"Modulación: {modulation}")
        #print(f"Q = {Q}")
        #print(f"Rmx = {Rmx}")
        #print("")

    # Crear DataFrame
    df = pd.DataFrame(datos, columns=["SNR", "Q", "Modulación", "Rmx", "CQI", "Distancia"])

    #print(df)
    return df
    



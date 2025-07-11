import time
import pandas as pd
import sys  # Añade esto al principio del archivo
import SNR_con_canal_Modelo
import numpy as np
from math import sqrt, log10
import MCS_SNR_final
import numpy as np
from scipy import constants
import math
def Cal_throughput(
    distancias, 
    DD_us,  
    BW_us, 
    SC_us,
    Carrier_frequency, 
    Height_antenna_BS,
    Power_delivered_to_all_antennas_of_BS_dBm,
    Height_antenna_UE,
    Noise_figure_UE,
    Temperatura_antenna_UE ):
    ##########################################
    # COMPROBACIÓN DE DISTANCIAS
    ##########################################

    # Suponiendo que distancias es un array NumPy o lista
    distancias = np.array(distancias)  # por si no lo es aún
    # Reemplazar valores cero por 0.1
    distancias[distancias == 0] = 0.1

    ###########################################################################################################################
    #SNR
    ###########################################################################################################################
    # Creamos un diccionario para almacenar todos los parámetros
    simParameters = {}

    # Parámetros del sistema
    simParameters['CarrierFrequency'] = Carrier_frequency     # Carrier frequency (Hz)



    ###########################################################################################################################
    #Determinar el RB, en función de BW y el SCS
    ###########################################################################################################################
    BW = BW_us # Mhz define la antena
    SC = SC_us # Definimos nosotros

    ######## DEFINIMOS FR #####################################################################################################
    if 450e6 <= simParameters['CarrierFrequency'] <= 7125e6:
        FR = 1
    elif 24.25e9 <= simParameters['CarrierFrequency'] <= 52.6e9:
        FR = 2
    else:
        FR = None  # O maneja el error según convenga, ya que la frecuencia no está en los rangos definidos


    FR = 1 # Esyo depende de la frecuencia portadora que aparezca arriba.


    ######## Cálculo RB ##############

    column = str(BW) + 'MHz'
        
    #Cálculo de número de prbs
    try:
        if FR == 1: 
            T = pd.read_table('table.txt', delimiter=',')
            if SC ==60: 
                row = 2
            else: 
                row = int((SC/15) - 1)
            RB = T.iloc[row][column]
            mu = row
        else:
            T = pd.read_table('table2.txt', delimiter=',')
            row = int((SC/60)-1)
            RB = T.iloc[row][column]
            mu = row + 3
    except Exception as e:
        print(f"Error accediendo a la tabla: {e}")
        sys.exit(0)

    time.sleep(1)



    ######## RESTO PARÁMETROS #####################################################################################################


    # ----------------------------------------
    # SNR + CONFIGURACIÓN INCIAL
    # ----------------------------------------

    # Creamos un diccionario para almacenar todos los parámetros
    simParameters = {}

    # CONFIGURACIÓN de la portadora
    simParameters['Carrier'] = SNR_con_canal_Modelo.NRCarrierConfig()
    simParameters['Carrier'].NSizeGrid = RB           # Número de RB por BW (51 RBs at 30 kHz SCS for 20 MHz BW)
    simParameters['Carrier'].SubcarrierSpacing = SC    # Espacio entre subportadoras 15, 30, 60, 120, 240 (kHz)
    simParameters['Carrier'].CyclicPrefix = 'Normal'   # 'Normal' or 'Extended' (Extended CP is relevant for 60 kHz SCS only)

    # Parámetros del sistema
    simParameters['TxHeight'] = Height_antenna_UE               
    simParameters['TxPower'] = Power_delivered_to_all_antennas_of_BS_dBm                 # Power delivered to all antennas of the BS on a fully allocated grid (dBm)
    simParameters['RxHeight'] = Height_antenna_BS               # Height of UE antenna (m)
    simParameters['RxNoiseFigure'] = Noise_figure_UE             # Noise figure of the UE (dB)
    simParameters['RxAntTemperature'] = Temperatura_antenna_UE        # Antenna temperature of the UE (K)
    simParameters['TxRxDistance'] = distancias  # Distance between the BS and UE (m)

    # Modelo de pérdida de propagación 5G NR con el escenario UMa (Urban Macro)
    simParameters['PathLossModel'] = '5G-NR'       # '5G-NR' or 'fspl'
    
    # RETARDO - Se usa un perfil TDL-A (canal multipath NLOS)
    simParameters['DelayProfile'] = 'TDL-A'  # A, B, C son canales NLOS. D y E son canales LOS.

    # Creamos el canal según el perfil elegido
    if 'CDL' in simParameters['DelayProfile'].upper():
        channel = SNR_con_canal_Modelo.NRCDLChannel()
        channel.DelayProfile = simParameters['DelayProfile']
        chInfo = channel.info()
        kFactor = chInfo.KFactorFirstCluster  # dB
    else:  # TDL
        channel = SNR_con_canal_Modelo.NRTDLChannel()
        channel.DelayProfile = simParameters['DelayProfile']
        chInfo = channel.info()
        kFactor = chInfo.KFactorFirstTap  # dB

    # Determine the sample rate and FFT size that are required for this carrier.
    waveformInfo = SNR_con_canal_Modelo.nrOFDMInfo(simParameters['Carrier'])
    maxChDelay = chInfo.MaximumChannelDelay

    # Calculate the path loss
    if '5G' in simParameters['PathLossModel'].upper():
        # Creamos los vectores de posición para TX y RX
        # txPosition = np.array([[0], [0], [simParameters['TxHeight']]])  # Posición del transmisor [x, y, z]
        
        # Para las posiciones de receptor, creamos una matriz de 3xN donde cada columna es una posición [x, y, z]
        dtr = simParameters['TxRxDistance']  # Distancias entre TX y RX
        num_positions = len(dtr)
        rxPosition = np.zeros((3, num_positions))
        
        # Asignamos valores para cada posición del receptor
        rxPosition[0, :] = dtr  # Posición x (distancia horizontal)
        rxPosition[1, :] = 0    # Posición y (0 para todas las posiciones)
        rxPosition[2, :] = simParameters['RxHeight']  # Altura del receptor igual para todas las posiciones
        
        # ------------------- MODELO A2G PERSONALIZADO NLOS URBANO -------------------

        # Parámetros del modelo L3(θ)
        gamma_0 = 20.43
        gamma_1 = 14.6948
        nu = 10.50

        # Calculamos la longitud de onda
        lambda_val = constants.c /Carrier_frequency

        # Extraemos posiciones
        tx_h = simParameters['TxHeight']
        rx_h = simParameters['RxHeight']
        d_2d = simParameters['TxRxDistance']  # Horizontal distances (array)

        # Cálculo del ángulo de elevación θ (en grados) para cada receptor
        # θ = arctan((h_tx - h_rx) / d_horizontal)
        theta_rad = np.arctan2(rx_h - tx_h, d_2d)
        theta_deg = np.degrees(theta_rad)

        # Calculamos L3(θ) para cada θ
        L3_theta = gamma_0 - gamma_1 * np.exp(-(90 - theta_deg) / nu)

        # FSPL en dB
        fspl_dB = SNR_con_canal_Modelo.fspl(d_2d, lambda_val)

        # Pérdida total
        pathLoss = fspl_dB + L3_theta  # Resultado en dB

        # --- SHADOWING en el peor caso (θ ≈ 0°) ---
        rho = 2.7940
        gamma = 0.2259
        sigma_s = rho * (90 ** gamma)  # σ_s en dB para θ=0°

        # Añadir shadowing log-normal (en dB)
        #shadowing_dB = np.random.normal(0, sigma_s, size=pathLoss.shape)  # media 0, std dev sigma_s
        #pathLoss += shadowing_dB
    
    else:  # Free-space path loss
        lambda_val = constants.c /Carrier_frequency
        pathLoss = SNR_con_canal_Modelo.fspl(simParameters['TxRxDistance'], lambda_val)
    # Constantes físicas
    kBoltzmann = constants.Boltzmann  # Constante de Boltzmann en J/K
        
    # Cálculo de la densidad espectral de ruido (N0) en W/Hz
    T = simParameters['RxAntTemperature']
    NF = 10**(simParameters['RxNoiseFigure']/10)
    N0 = kBoltzmann * T * NF

    # Cálculo del SNR
    fftOccupancy = 12 * simParameters['Carrier'].NSizeGrid / waveformInfo.Nfft

    # Initialize parameters
    NFrames = 20
    NSlots = NFrames * simParameters['Carrier'].SlotsPerFrame
    nSNRPoints = len(pathLoss)  # Number of SNR points

    # Initialize measurements and create auxiliary variables
    nTxAnt = chInfo.NumTransmitAntennas
    nRxAnt = chInfo.NumReceiveAntennas

    # Initialize arrays
    powSignalRE = np.zeros((nSNRPoints, nRxAnt, NSlots))
    powSignal = np.zeros((nSNRPoints, nRxAnt, NSlots))
    powNoiseRE = np.zeros((nSNRPoints, nRxAnt, NSlots))
    powNoise = np.zeros((nSNRPoints, nRxAnt, NSlots))
    pgains = np.zeros((len(chInfo.PathDelays), nTxAnt, nRxAnt, nSNRPoints, NSlots))

    scs = simParameters['Carrier'].SubcarrierSpacing
    nSizeGrid = simParameters['Carrier'].NSizeGrid
    nfft = waveformInfo.Nfft
    cp_type = simParameters['Carrier'].CyclicPrefix
    # Reset the random generator for reproducibility
    np.random.seed(0)


    P_tx_W    = 10**((simParameters['TxPower'] - 30)/10)  # 40 dBm → 10 W
    A_tx      = np.sqrt(P_tx_W)                          # Amplitud lineal
    scs_hz    = simParameters['Carrier'].SubcarrierSpacing * 1e3
    nSubc     = simParameters['Carrier'].NSizeGrid * 12
    B         = scs_hz * nSubc                            # ≈ 18.36 MHz
    noise_pow = N0 * B                                    # Potencia total de ruido (W)
    noise_std = np.sqrt(noise_pow / 2)                    # σ por componente

    signal_power_per_distance = []
    noise_power_per_distance = []
    snr_dB_per_distance = []

    # Transmit a CP-OFDM waveform through the channel and measure the SNR for
    # each distance between Tx and Rx (path loss values)
    for pl in range(nSNRPoints):
        carrier = simParameters['Carrier'].copy()  # Create a copy to avoid modifying the original
        signal_powers = []
        noise_powers = []
        # Este bucle simula la transmisión y recepción de señales en cada slot de una portadora 5G. 
        # En 5G NR, un slot es una unidad de tiempo/frecuencia donde se transmite un conjunto de símbolos OFDM
        for slot in range(NSlots):
            # 1) Genera y modula txgrid → txWaveform (norm IFFT SIN /√nfft)
            # 1) Creamos el grid vacío
            txgrid = SNR_con_canal_Modelo.nrResourceGrid(carrier, nTxAnt)

            # 2) Generamos bits y QPSK
            bits       = np.random.randint(0,2, size=(np.prod(txgrid.shape)*2,))
            symbolsQPSK = SNR_con_canal_Modelo.nrSymbolModulate(bits, 'QPSK')

            # 3) Mapear esos símbolos al grid (dejando a cero donde no ponemos datos)
            txgrid = SNR_con_canal_Modelo.mapDatosEnGrid(txgrid, symbolsQPSK)

            # 4) Modular OFDM
            txWaveform = SNR_con_canal_Modelo.nrOFDMModulate(txgrid, scs, nfft, cp_type)
            
            # 2) Escala a la potencia correcta
            txWaveform *= A_tx
            
            # 3) Padding si quieres…
            #    txWaveform = np.vstack([txWaveform, zeros_padding])
            
            # 4) Canal y path‐loss
            rxWaveform, _, _ = channel(txWaveform)
            rxWaveform     *= 10**(-pathLoss[pl]/10)
            
            # 5) Mide potencia recibida (después de path‐loss, antes del ruido)
                        # 1) Calcula potencia medida de rxWaveform
            P_meas = np.mean(np.abs(rxWaveform)**2)          # W

                        # 2) Calcula potencia esperada según dBm-pathloss
            P_exp_dBm = simParameters['TxPower'] - pathLoss[pl]
            P_exp_W   = 10**((P_exp_dBm - 30)/10)

                        # 3) Factor de corrección para “forzar” P_meas → P_exp
            alpha     = np.sqrt(P_exp_W / P_meas)
            rxWaveform *= alpha

                        # 4) Verifica que ahora coincide
            P_rx_W    = np.mean(np.abs(rxWaveform)**2)
            P_rx_dBm  = 10*np.log10(P_rx_W) + 30
            #print(f">> Rx power FORZADA: {P_rx_dBm:.2f} dBm  (esperado {P_exp_dBm:.2f} dBm)")

            
            # 6) Genera AWGN con σ = sqrt(noise_pow/2)
            noise   = (np.random.randn(*rxWaveform.shape)
                    + 1j*np.random.randn(*rxWaveform.shape)) * noise_std
            
            ## Forzar la señal recibida con ruido
            rx_noisy= rxWaveform + noise
            
            # 7) Demodula rx_noisy → rxgrid, mide potencias RE
            rxgrid       = SNR_con_canal_Modelo.nrOFDMDemodulate(carrier, rx_noisy)
            powSignalRE  =float(np.mean(np.abs(rxgrid)**2 ))
            ngrid        = SNR_con_canal_Modelo.nrOFDMDemodulate(carrier, noise)
            powNoiseRE   = float(np.mean(np.abs(ngrid)**2))

            signal_powers.append(powSignalRE)
            noise_powers.append(powNoiseRE)
            
        # Calcular la media después de 400 slots
        avg_signal_power = np.mean(signal_powers)
        avg_noise_power = np.mean(noise_powers)
        snr = avg_signal_power / avg_noise_power
        snr_dB = 10 * np.log10(snr)

        signal_power_per_distance.append(avg_signal_power)
        noise_power_per_distance.append(avg_noise_power)
        snr_dB_per_distance.append(snr_dB)
        # Print resource grid usage info
        #print(f'The resource grid uses {fftOccupancy*100:.1f} % of the FFT size, introducing a {-10*log10(fftOccupancy):.1f} dB SNR gain.')


    df = pd.DataFrame({
        'Distance (m)': simParameters['TxRxDistance'],  # o la lista de distancias que usas
        'Path Loss (dB)': pathLoss,
        'SNR (dB)': snr_dB_per_distance
    })

    ######################################################################### MODULAMOS OFDM #################################################################################
    ######################################################################### MODULAMOS OFDM #################################################################################

    #print("\nSNR Results:")
    print(df)

    #print(f"\nAverage SNR across all distances: {np.mean(snr_dB_per_distance):.4f} dB")

    SINR_dB = df['SNR (dB)'].values
    #print("Valor SNR que pasamos para calcular el MCS:", SINR_dB)
    ###########################################################################################################################
    #SNR --> MCS
    ###########################################################################################################################

    df_MCS = MCS_SNR_final.Obtener_MCS(SINR_dB, simParameters['TxRxDistance'])

    #print("\nObtenemos el CQI y definimos Q y Rmx:")
    #print(df_MCS)

    Q_array = df_MCS['Q'].to_numpy()
    Rmx_array = df_MCS['Rmx'].to_numpy()
    Distancias = simParameters['TxRxDistance']


    ###########################################################################################################################
    #Cálculo  5G: Definir enlace, portadoras, factor escaladp
    ###########################################################################################################################

    DD = DD_us
    J = 8 #  número de portadoras. 
    f = 0.4
    v = 1

    #Comprobaciones, añadir más
    if DD == "Uplink" and v==8: 
        print("Warning: Número máximo de capas MIMO para uplink es 4x4.\n")
    elif FR==2 and BW<50:
        print("Warning: El valor de ancho de banda no corresponde con FR2.\n")
    elif FR==1 and BW>100:
        print("Warnig: El valor de ancho de banda no corresponde con FR1.\n")
    elif BW>50 and SC==15: 
        print("Warning: Nrb fue establecido como N/A.\n")
        


    #Cáculo de OH - Depende de enlace
    if FR == 1 and DD=='Downlink':   OH = 0.14
    elif FR == 2 and DD=='Downlink': OH = 0.18
    elif FR == 1 and DD=='Uplink': OH = 0.08
    elif FR == 2 and DD=='Uplink': OH = 0.1



    # Lista para guardar los datos
    datos_throughput = []

    # Cálculo por usuario
    for idx in range(len(Distancias)):
        Q = Q_array[idx]
        Rmx = Rmx_array[idx]
        distancia = Distancias[idx]
        path_loss_val = pathLoss[idx]

        Th = 0
        for _ in range(J):
            Th += v * Q * f * Rmx * ((RB * 12) * (14 * (2 ** mu) / (10 ** -3))) * (1 - OH)
        Th = round(Th * 10**-6, 2)  # Mbps

        if Th > 1024:
            Th_str = f"{round(Th / 1024, 2)} Gbps"
        else:
            Th_str = f"{Th} Mbps"

        # Guardamos los datos por usuario
        datos_throughput.append({
            #'Distancia (m)': distancia,
            'Q': Q,
            'Rmx': Rmx,
            'Throughput': Th_str,
            'Path Loss (dB)': path_loss_val,
            'SNR (dB)': SINR_dB[idx]

        })

    # Creamos el DataFrame final
    df_resultados = pd.DataFrame(datos_throughput)

    # Mostramos el resultado
    #print("\nResultados Throughput:")
    print(df_resultados)
    return(df_resultados)
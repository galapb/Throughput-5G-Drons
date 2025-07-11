import numpy as np
import pandas as pd
from scipy import constants
from math import sqrt, log10
import matplotlib.pyplot as plt

# Clase para emular la configuración de portadora de NR
class NRCarrierConfig:
    def __init__(self):
        self.NSizeGrid = None
        self.SubcarrierSpacing = None
        self.CyclicPrefix = None
        self.SlotsPerFrame = 10  # Valor estándar para 5G NR (depende del SCS)
        self.NSlot = 0  # Para controlar el índice de slot actual
    
    def copy(self):
        # Método para crear una copia del objeto
        new_carrier = NRCarrierConfig()
        new_carrier.NSizeGrid = self.NSizeGrid
        new_carrier.SubcarrierSpacing = self.SubcarrierSpacing
        new_carrier.CyclicPrefix = self.CyclicPrefix
        new_carrier.SlotsPerFrame = self.SlotsPerFrame
        new_carrier.NSlot = self.NSlot
        return new_carrier

# Clase para emular la configuración de pérdida de trayectoria NR
class NRPathLossConfig:
    def __init__(self):
        self.Scenario = None
        self.EnvironmentHeight = None

# Clase para emular el canal TDL
class NRTDLChannel:
    def __init__(self):
        self.DelayProfile = None
        self.Seed = 0
        # Agregamos los atributos que faltaban para el canal
        self.NumTransmitAntennas = 1
        self.NumReceiveAntennas = 1
        self.PathDelays = np.array([0, 100e-9, 200e-9, 300e-9, 400e-9])  # Ejemplo de delays
    
    def info(self):
        # Simulamos la información del canal
        class ChannelInfo:
            def __init__(self, delay_profile):
                self.MaximumChannelDelay = 1e-6  # 1 μs por defecto
                # Asignamos KFactorFirstTap según el perfil de retardo
                if delay_profile in ["TDL-D", "TDL-E"]:  # Canales LOS
                    self.KFactorFirstTap = 10  # Valor positivo para LOS
                else:  # Canales NLOS (TDL-A, TDL-B, TDL-C)
                    self.KFactorFirstTap = -np.inf  # -infinito para NLOS
                self.NumTransmitAntennas = 1
                self.NumReceiveAntennas = 1
                self.PathDelays = np.array([0, 100e-9, 200e-9, 300e-9, 400e-9])  # Ejemplo de delays
        
        return ChannelInfo(self.DelayProfile)
    
    def reset(self):
        # Método para restablecer el canal
        self.Seed = 0
    
    def __call__(self, txWaveform):
        # Simular el comportamiento del canal
        # Esto es una simulación muy simplificada
        rxWaveform = txWaveform.copy()  # En un caso real, aplicaríamos efectos del canal
        pathGains = np.ones((1, len(self.PathDelays), self.NumTransmitAntennas, self.NumReceiveAntennas))
        sampleTimes = np.arange(rxWaveform.shape[0])
        return rxWaveform, pathGains, sampleTimes

# Clase para emular el canal CDL
class NRCDLChannel:
    def __init__(self):
        self.DelayProfile = None
        self.Seed = 0
        # Agregamos los atributos que faltaban para el canal
        self.NumTransmitAntennas = 1
        self.NumReceiveAntennas = 1
        self.PathDelays = np.array([0, 100e-9, 200e-9, 300e-9, 400e-9])  # Ejemplo de delays
    
    def info(self):
        # Simulamos la información del canal
        class ChannelInfo:
            def __init__(self, delay_profile):
                self.MaximumChannelDelay = 1e-6  # 1 μs por defecto
                self.KFactorFirstCluster = -np.inf  # Por defecto NLOS
                # Asignamos KFactorFirstCluster según el perfil
                if "LOS" in delay_profile:
                    self.KFactorFirstCluster = 10  # Valor positivo para LOS
                self.NumTransmitAntennas = 2
                self.NumReceiveAntennas = 1
                self.PathDelays = np.array([0, 100e-9, 200e-9, 300e-9, 400e-9])  # Ejemplo de delays
        
        return ChannelInfo(self.DelayProfile)
    
    def reset(self):
        # Método para restablecer el canal
        self.Seed = 0
    
    def __call__(self, txWaveform):
        # Simular el comportamiento del canal
        # Esto es una simulación muy simplificada
        rxWaveform = txWaveform.copy()  # En un caso real, aplicaríamos efectos del canal
        pathGains = np.ones((1, len(self.PathDelays), self.NumTransmitAntennas, self.NumReceiveAntennas))
        sampleTimes = np.arange(rxWaveform.shape[0])
        return rxWaveform, pathGains, sampleTimes

# Función para calcular información OFDM (equivalente a nrOFDMInfo)
def nrOFDMInfo(carrier):
    # Cálculo simplificado del tamaño de FFT basado en NSizeGrid y SCS
    # En 5G NR, el tamaño de FFT depende del ancho de banda y el espaciado de subportadoras
    # Estos son valores aproximados basados en especificaciones 3GPP
    scs_to_fft = {
        15: 2048,   # 15 kHz SCS -> FFT size típico
        30: 2048,   # 30 kHz SCS -> FFT size típico
        60: 4096,   # 60 kHz SCS -> FFT size típico
        120: 4096,  # 120 kHz SCS -> FFT size típico
        240: 8192   # 240 kHz SCS -> FFT size típico
    }
    
    class OFDMInfo:
        def __init__(self):
            self.Nfft = scs_to_fft.get(carrier.SubcarrierSpacing, 2048)
    
    return OFDMInfo()


# Función para calcular la pérdida de trayectoria en espacio libre
def fspl(distances, wavelength):
    """
    Free Space Path Loss (FSPL)

    """
    # Free Space Path Loss (FSPL) = 20*log10(4*pi*d/lambda)
    path_loss = 20 * np.log10(4 * np.pi * distances / wavelength)
    return path_loss

def db2mag(db_value):
    """Convert dB value to magnitude"""
    return 10 ** (db_value / 20)
def nrResourceGrid(carrier, nTxAnt):
    Nsc = carrier.NSizeGrid * 12
    Nsym = 14
    return np.zeros((Nsc, Nsym, nTxAnt), dtype=complex)

def mapDatosEnGrid(txgrid, data_symbols):
    """
    Rellena txgrid con data_symbols (un vector plano de length = Nsc*Nsym*nTxAnt)
    recorriendo primero subcarrier, luego símbolo, luego antena.
    """
    # data_symbols debe tener longitud Nsc*Nsym*nTxAnt
    return data_symbols.reshape(txgrid.shape)

def nrSymbolModulate(bits, modulation_type):
    """5G NR Symbol Modulation - QPSK implementation"""
    if modulation_type == 'QPSK':
        # Group bits in pairs
        bits_reshaped = bits.reshape(-1, 2)
        symbols = np.zeros(len(bits_reshaped), dtype=complex)
        
        # QPSK mapping
        symbols[np.all(bits_reshaped == [0, 0], axis=1)] = complex(1/sqrt(2), 1/sqrt(2))
        symbols[np.all(bits_reshaped == [0, 1], axis=1)] = complex(1/sqrt(2), -1/sqrt(2))
        symbols[np.all(bits_reshaped == [1, 0], axis=1)] = complex(-1/sqrt(2), 1/sqrt(2))
        symbols[np.all(bits_reshaped == [1, 1], axis=1)] = complex(-1/sqrt(2), -1/sqrt(2))
        
        return symbols
    else:
        raise ValueError(f"Modulation type {modulation_type} not implemented")

def nrOFDMModulate(grid, scs, nfft, cp_type):
    """
    Simplified 5G NR OFDM Modulation with normal cyclic prefix
    """
    nSubcarriers, nSymbols, nTxAnt = grid.shape
    cp_length = int(nfft * 0.07)  # Aproximación CP normal ≈ 7%

    offset = (nfft - nSubcarriers) // 2
    waveform = []

    for sym in range(nSymbols):
        # Para cada símbolo construimos symbol_freq nuevo
        # Con forma (nfft, nTxAnt), zeros excepto tus subportadoras
        symbol_freq = np.zeros((nfft, nTxAnt), dtype=complex)
        # Mapeo centrado de las subportadoras
        symbol_freq[offset:offset+nSubcarriers, :] = grid[:, sym, :]

        # IFFT normalizada para conservar energía
        symbol_time = np.fft.ifft(
            np.fft.ifftshift(symbol_freq, axes=0),
            n=nfft, axis=0
        ) / np.sqrt(nfft)

        # Añadir prefijo cíclico
        cyclic_prefix = symbol_time[-cp_length:, :]
        ofdm_symbol = np.vstack((cyclic_prefix, symbol_time))

        waveform.append(ofdm_symbol)

    # Concatenar todos los símbolos OFDM por tiempo
    return np.vstack(waveform)

def nrOFDMDemodulate(carrier, waveform):
    """
    Simplified 5G NR OFDM Demodulation with normal cyclic prefix
    """
    scs = carrier.SubcarrierSpacing
    nSizeGrid = carrier.NSizeGrid
    nfft = 2 ** int(np.ceil(np.log2(nSizeGrid * 12)))  # 12 subcarriers por resource block
    cp_length = int(nfft * 0.07)  # Aproximación CP normal

    nSubcarriers = nSizeGrid * 12
    symbol_length = cp_length + nfft
    nSymbols = waveform.shape[0] // symbol_length
    nRxAnt = waveform.shape[1]

    grid = np.zeros((nSubcarriers, nSymbols, nRxAnt), dtype=complex)

    for sym in range(nSymbols):
        start = sym * symbol_length + cp_length
        end = start + nfft
        symbol_freq = np.fft.fftshift(np.fft.fft(waveform[start:end, :], n=nfft, axis=0), axes=0)
        grid[:, sym, :] = symbol_freq[:nSubcarriers, :]

    return grid

def getPathFilters(channel):
    """Get path filters - simplificado"""
    # Retorna filtros ficticios
    return np.zeros((10, 10))

def nrPerfectTimingEstimate(pathGains, pathFilters):
    """Perfect timing estimation - simplificado"""
    # Retorna un offset fijo para simulación
    return 10


""" 
# ----------------------------------------
# CONFIGURACIÓN PRINCIPAL 
# ----------------------------------------

# Creamos un diccionario para almacenar todos los parámetros
simParameters = {}

# CONFIGURACIÓN de la portadora
simParameters['Carrier'] = NRCarrierConfig()
simParameters['Carrier'].NSizeGrid = 51            # Número de RB por BW (51 RBs at 30 kHz SCS for 20 MHz BW)
simParameters['Carrier'].SubcarrierSpacing = 30    # Espacio entre subportadoras 15, 30, 60, 120, 240 (kHz)
simParameters['Carrier'].CyclicPrefix = 'Normal'   # 'Normal' or 'Extended' (Extended CP is relevant for 60 kHz SCS only)
simParameters['Carrier'].SlotsPerFrame = 10        # 10 slots per frame for 30 kHz SCS

# Parámetros del sistema
simParameters['CarrierFrequency'] = 5e9      # Carrier frequency (Hz)
simParameters['TxHeight'] = 1.5                 # Height of the BS antenna (m)
simParameters['TxPower'] = 40                  # Power delivered to all antennas of the BS on a fully allocated grid (dBm)
simParameters['RxHeight'] = 100                # Height of UE antenna (m)
simParameters['RxNoiseFigure'] = 6             # Noise figure of the UE (dB)
simParameters['RxAntTemperature'] = 290        # Antenna temperature of the UE (K)
simParameters['TxRxDistance'] = np.array([123, 250, 679, 800])  # Distance between the BS and UE (m)

# Modelo de pérdida de propagación 5G NR con el escenario UMa (Urban Macro)
simParameters['PathLossModel'] = '5G-NR'       # '5G-NR' or 'fspl'

# RETARDO - Se usa un perfil TDL-A (canal multipath NLOS)
simParameters['DelayProfile'] = 'TDL-A'  # A, B, C son canales NLOS. D y E son canales LOS.

# Creamos el canal según el perfil elegido
if 'CDL' in simParameters['DelayProfile'].upper():
    channel = NRCDLChannel()
    channel.DelayProfile = simParameters['DelayProfile']
    chInfo = channel.info()
    kFactor = chInfo.KFactorFirstCluster  # dB
else:  # TDL
    channel = NRTDLChannel()
    channel.DelayProfile = simParameters['DelayProfile']
    chInfo = channel.info()
    kFactor = chInfo.KFactorFirstTap  # dB

# Determine the sample rate and FFT size that are required for this carrier.
waveformInfo = nrOFDMInfo(simParameters['Carrier'])

# Get the maximum delay of the fading channel.
maxChDelay = chInfo.MaximumChannelDelay

# Calculate the path loss
if '5G' in simParameters['PathLossModel'].upper():
    # Creamos los vectores de posición para TX y RX
    txPosition = np.array([[0], [0], [simParameters['TxHeight']]])  # Posición del transmisor [x, y, z]
    
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
    lambda_val = constants.c / simParameters['CarrierFrequency']

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
    fspl_dB = fspl(d_2d, lambda_val)

    # Pérdida total
    pathLoss = fspl_dB + L3_theta  # Resultado en dB

     # --- SHADOWING en el peor caso (θ ≈ 0°) ---
    rho = 2.7940
    gamma = 0.2259
    sigma_s = rho * (90 ** gamma)  # σ_s en dB para θ=0°

    # Añadir shadowing log-normal (en dB)
    #shadowing_dB = np.random.normal(0, sigma_s, size=pathLoss.shape)  # media 0, std dev sigma_s
    #pathLoss += shadowing_dB
    print("Distancias:", d_2d)
    print("Ángulo elevación (°):", theta_deg)
    
    print("L3(θ):", L3_theta)
    print("FSPL (dB):", fspl_dB)
    #print("Shadowing (°):",shadowing_dB)
    print("\nValores de Path Loss (dB):", pathLoss) 
else:  # Free-space path loss
    lambda_val = constants.c / simParameters['CarrierFrequency']
    pathLoss = fspl(simParameters['TxRxDistance'], lambda_val)

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
        txgrid = nrResourceGrid(carrier, nTxAnt)

        # 2) Generamos bits y QPSK
        bits       = np.random.randint(0,2, size=(np.prod(txgrid.shape)*2,))
        symbolsQPSK = nrSymbolModulate(bits, 'QPSK')

        # 3) Mapear esos símbolos al grid (dejando a cero donde no ponemos datos)
        txgrid = mapDatosEnGrid(txgrid, symbolsQPSK)

        # 4) Modular OFDM
        txWaveform = nrOFDMModulate(txgrid, scs, nfft, cp_type)
        
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
        rxgrid       = nrOFDMDemodulate(carrier, rx_noisy)
        powSignalRE  =float(np.mean(np.abs(rxgrid)**2 ))
        ngrid        = nrOFDMDemodulate(carrier, noise)
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

print("Largo de cada lista:")
print("Distances:", len(simParameters['TxRxDistance']))
print("Path Loss:", len(pathLoss))
print("Signal Power:", len(signal_power_per_distance))
print("Noise Power:", len(noise_power_per_distance))
print("SNR (dB):", len(snr_dB_per_distance))

df = pd.DataFrame({
    'Distance (m)': simParameters['TxRxDistance'],  # o la lista de distancias que usas
    'Path Loss (dB)': pathLoss,
    'SNR (dB)': snr_dB_per_distance
})

######################################################################### MODULAMOS OFDM #################################################################################
######################################################################### MODULAMOS OFDM #################################################################################

# Use the actual number of receive antennas from the channel info instead of hardcoding
nRxAnt = chInfo.NumReceiveAntennas  # This should be 2 based on your channel configuration

# Print some debug info
print(f"Number of receive antennas: {nRxAnt}")

print("\nSNR Results:")
print(df)

print(f"\nAverage SNR across all distances: {np.mean(snr_dB_per_distance):.4f} dB")



# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(df['Distance (m)'], df['SNR (dB)'], marker='o', linestyle='-')
plt.grid(True)
plt.xlabel('Distance Tx-Rx (m)')
plt.ylabel('SNR (dB)')
plt.title('SNR vs Distance in 5G Transmission')
plt.tight_layout()
plt.show() """
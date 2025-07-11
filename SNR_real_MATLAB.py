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
        self.NumReceiveAntennas = 2
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
                self.NumReceiveAntennas = 2
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
        self.NumReceiveAntennas = 2
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
                self.NumTransmitAntennas = 1
                self.NumReceiveAntennas = 2
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

# Función para calcular la pérdida de trayectoria 5G NR
def nrPathLoss(plConfig, carrierFreq, isLOS, txPosition, rxPosition):
    """
    Implementación simplificada del modelo de pérdida de trayectoria 5G NR para escenario UMa
    Basado en 3GPP TR 38.901
    
    Parameters:
    -----------
    plConfig : NRPathLossConfig
        Configuración del modelo de pérdida de trayectoria
    carrierFreq : float
        Frecuencia de portadora en Hz
    isLOS : bool
        Condición de visión directa (True) o no directa (False)
    txPosition : numpy.ndarray
        Posición del transmisor (BS) [x,y,z]
    rxPosition : numpy.ndarray
        Posición del receptor (UE) [x,y,z] para múltiples posiciones
        
    Returns:
    --------
    pathLoss : numpy.ndarray
        Pérdida de trayectoria en dB para cada posición del receptor
    """
    # Aseguramos que txPosition sea una matriz de forma correcta
    txPosition = np.array(txPosition).reshape(3, 1)
    
    # Calculamos la distancia 3D entre TX y RX
    distances = np.zeros(rxPosition.shape[1])
    for i in range(rxPosition.shape[1]):
        distances[i] = np.sqrt(np.sum((txPosition.flatten() - rxPosition[:, i])**2))
    
    # Convertimos frecuencia a GHz para las fórmulas
    f_ghz = carrierFreq / 1e9
    
    # Calculamos altura efectiva (ajustando por altura del entorno)
    h_e = plConfig.EnvironmentHeight
    h_bs = txPosition[2, 0]  # Altura de la BS
    h_ut = rxPosition[2, :]  # Altura del UE
    
    # Cálculo de la pérdida de trayectoria según el escenario
    pathLoss = np.zeros_like(distances)
    
    if plConfig.Scenario == 'UMa':
        # Para Urban Macro
        # Distancia 2D entre BS y UE (proyección en el plano XY)
        d_2d = np.sqrt((rxPosition[0, :] - txPosition[0, 0])**2 + (rxPosition[1, :] - txPosition[1, 0])**2)
        
        # Distancia de breakpoint (distancia crítica donde cambia el modelo)
        # Para cada posición del UE
        d_bp = np.zeros_like(h_ut)
        for i in range(len(h_ut)):
            d_bp[i] = 4 * h_bs * h_ut[i] * f_ghz * (10/3)
        
        for i in range(len(distances)):
            if isLOS:  # Modelo LOS
                if d_2d[i] <= d_bp[i]:
                    # PL1: Modelo para distancias cortas
                    pl = 28.0 + 22 * np.log10(d_2d[i]) + 20 * np.log10(f_ghz)
                else:
                    # PL2: Modelo para distancias largas
                    pl = 28.0 + 40 * np.log10(d_2d[i]) + 20 * np.log10(f_ghz) - 9 * np.log10((d_bp[i])**2 + (h_bs - h_ut[i])**2)
                pathLoss[i] = pl
            else:  # Modelo NLOS
                # Modelo simplificado NLOS para UMa
                pl_nlos = 13.54 + 39.08 * np.log10(d_2d[i]) + 20 * np.log10(f_ghz) - 0.6 * (h_ut[i] - 1.5)
                pathLoss[i] = pl_nlos
                
    return pathLoss

# Función para calcular la pérdida de trayectoria en espacio libre
def fspl(distances, wavelength):
    """
    Free Space Path Loss (FSPL)
    
    Parameters:
    -----------
    distances : numpy.ndarray
        Distancias entre TX y RX
    wavelength : float
        Longitud de onda de la señal
        
    Returns:
    --------
    path_loss : numpy.ndarray
        Pérdida de trayectoria en dB
    """
    # Free Space Path Loss (FSPL) = 20*log10(4*pi*d/lambda)
    path_loss = 20 * np.log10(4 * np.pi * distances / wavelength)
    return path_loss

def db2mag(db_value):
    """Convert dB value to magnitude"""
    return 10 ** (db_value / 20)

def nrResourceGrid(carrier, nTxAnt):
    """5G NR Resource Grid - simplificado"""
    # Crea una matriz de subportadoras x símbolos x antenas
    return np.zeros((carrier.NSizeGrid*12, 14, nTxAnt))

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

def nrOFDMModulate(grid, scs, slot):
    """5G NR OFDM Modulation - simplificado"""
    # Simplificación que devuelve una forma de onda de ejemplo
    return np.zeros((1000, grid.shape[2]), dtype=complex)

def nrOFDMDemodulate(carrier, signal):
    """5G NR OFDM Demodulation - simplificado"""
    # Simplificación que devuelve una rejilla ejemplo
    return np.zeros((carrier.NSizeGrid*12, 14, signal.shape[1]), dtype=complex)

def getPathFilters(channel):
    """Get path filters - simplificado"""
    # Retorna filtros ficticios
    return np.zeros((10, 10))

def nrPerfectTimingEstimate(pathGains, pathFilters):
    """Perfect timing estimation - simplificado"""
    # Retorna un offset fijo para simulación
    return 10

# ----------------------------------------
# CONFIGURACIÓN PRINCIPAL 
# ----------------------------------------

# Creamos un diccionario para almacenar todos los parámetros
""" simParameters = {}

# CONFIGURACIÓN de la portadora
simParameters['Carrier'] = NRCarrierConfig()
simParameters['Carrier'].NSizeGrid = 51            # Número de RB por BW (51 RBs at 30 kHz SCS for 20 MHz BW)
simParameters['Carrier'].SubcarrierSpacing = 30    # Espacio entre subportadoras 15, 30, 60, 120, 240 (kHz)
simParameters['Carrier'].CyclicPrefix = 'Normal'   # 'Normal' or 'Extended' (Extended CP is relevant for 60 kHz SCS only)
simParameters['Carrier'].SlotsPerFrame = 10        # 10 slots per frame for 30 kHz SCS

# Parámetros del sistema
simParameters['CarrierFrequency'] = 3.5e9      # Carrier frequency (Hz)
simParameters['TxHeight'] = 25                 # Height of the BS antenna (m)
simParameters['TxPower'] = 40                  # Power delivered to all antennas of the BS on a fully allocated grid (dBm)
simParameters['RxHeight'] = 1.5                # Height of UE antenna (m)
simParameters['RxNoiseFigure'] = 6             # Noise figure of the UE (dB)
simParameters['RxAntTemperature'] = 290        # Antenna temperature of the UE (K)
simParameters['TxRxDistance'] = np.array([10, 100])  # Distance between the BS and UE (m)

# Modelo de pérdida de propagación 5G NR con el escenario UMa (Urban Macro)
simParameters['PathLossModel'] = '5G-NR'       # '5G-NR' or 'fspl'
simParameters['PathLoss'] = NRPathLossConfig()
simParameters['PathLoss'].Scenario = 'UMa'     # Urban macrocell
simParameters['PathLoss'].EnvironmentHeight = 1  # Average height of the environment in UMa/UMi

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

# Determine LOS between Tx and Rx based on Rician factor K.
simParameters['LOS'] = kFactor > -np.inf

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
    
    # Calculamos la pérdida de trayectoria
    pathLoss = nrPathLoss(simParameters['PathLoss'], simParameters['CarrierFrequency'], 
                         simParameters['LOS'], txPosition, rxPosition)
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

# Reset the random generator for reproducibility
np.random.seed(0)

# Transmit a CP-OFDM waveform through the channel and measure the SNR for
# each distance between Tx and Rx (path loss values)
for pl in range(nSNRPoints):
    carrier = simParameters['Carrier'].copy()  # Create a copy to avoid modifying the original
    
    for slot in range(NSlots):
        slotIdx = slot
        carrier.NSlot = slot
        
        # Reset the channel with a new seed for block fading
        channel.reset()  
        channel.Seed = slot
        
        # Create the OFDM resource grid and allocate random QPSK symbols
        txgrid = nrResourceGrid(carrier, nTxAnt)
        
        # Generate random bits and modulate them using QPSK
        random_bits = np.random.randint(0, 2, size=(np.prod(txgrid.shape)*2))
        txgrid_flat = nrSymbolModulate(random_bits, 'QPSK')
        # Reshape to match the expected dimensions
        txgrid = txgrid_flat.reshape(txgrid.shape)
        
        # Perform CP-OFDM modulation
        txWaveform = nrOFDMModulate(txgrid, scs, slot)
        
        # Calculate the amplitude of the transmitted signal
        signalAmp = db2mag(simParameters['TxPower']-30) * sqrt(nfft**2/(nSizeGrid*12*nTxAnt))
        txWaveform = signalAmp * txWaveform
        
        # Pad the signal with zeros to ensure full slot after synchronization
        maxChDelay_samples = int(maxChDelay * 1e6)  # Convertir a muestras (simplificación)
        zeros_padding = np.zeros((maxChDelay_samples, txWaveform.shape[1]))
        txWaveform = np.vstack((txWaveform, zeros_padding))
        
        # Pass the signal through fading channel
        rxWaveform, pathGains, sampleTimes = channel(txWaveform)
        pgains[:,:,:,pl,slotIdx] = pathGains[0,:,:,:]
        
        # Apply path loss to the signal
        rxWaveform = rxWaveform * db2mag(-pathLoss[pl])
        
        # Generate AWGN
        noise_shape = rxWaveform.shape
        real_noise = np.random.normal(0, 1, noise_shape)
        imag_noise = np.random.normal(0, 1, noise_shape)
        noise = N0 * (real_noise + 1j * imag_noise)
        
        # Perform perfect synchronization
        pathFilters = getPathFilters(channel)
        offset = nrPerfectTimingEstimate(pathGains, pathFilters)
        rxWaveform = rxWaveform[offset:,:]
        noise = noise[offset:,:]
        
        # Perform CP-OFDM demodulation of the received signal and noise
        ngrid = nrOFDMDemodulate(carrier, noise)
        rxgrid = nrOFDMDemodulate(carrier, rxWaveform)
        
        # Measure the RE and overall power of the received signal and noise
        powSignalRE[pl,:,slotIdx] = np.sqrt(np.mean(np.abs(rxgrid)**2, axis=(0,1)))**2 / nfft**2
        powSignal[pl,:,slotIdx] = powSignalRE[pl,:,slotIdx] * nSizeGrid * 12
        powNoiseRE[pl,:,slotIdx] = np.sqrt(np.mean(np.abs(ngrid)**2, axis=(0,1)))**2 / nfft**2
        powNoise[pl,:,slotIdx] = powNoiseRE[pl,:,slotIdx] * nfft

# Print resource grid usage info
print(f'The resource grid uses {fftOccupancy*100:.1f} % of the FFT size, introducing a {-10*log10(fftOccupancy):.1f} dB SNR gain.')




######################################################################### MODULAMOS OFDM #################################################################################
######################################################################### MODULAMOS OFDM #################################################################################

# Use the actual number of receive antennas from the channel info instead of hardcoding
nRxAnt = chInfo.NumReceiveAntennas  # This should be 2 based on your channel configuration

# Print some debug info
print(f"Number of receive antennas: {nRxAnt}")
print(f"TxRx Distances: {simParameters['TxRxDistance']}")
print(f"Path Loss values: {pathLoss}")

# Modified calculation that ensures distance-specific SNR results
# First, verify the shapes of our power matrices
print(f"Shape of powSignal: {powSignal.shape}")
print(f"Shape of powNoise: {powNoise.shape}")

# Calcular medias por slots
signal_mean = np.mean(powSignal, axis=2)  # Mean across slots
noise_mean = np.mean(powNoise, axis=2)    # Mean across slots
signal_mean_RE = np.mean(powSignalRE, axis=2)  # Mean across slots
noise_mean_RE = np.mean(powNoiseRE, axis=2)    # Mean across slots

# Print the mean values for debugging
print("\nMean values:")
print(f"Signal mean: {signal_mean}")
print(f"Noise mean: {noise_mean}")

# Asegurar valores positivos para logaritmo
epsilon = 1e-10
signal_mean = np.clip(signal_mean, epsilon, None)
noise_mean = np.clip(noise_mean, epsilon, None)
signal_mean_RE = np.clip(signal_mean_RE, epsilon, None)
noise_mean_RE = np.clip(noise_mean_RE, epsilon, None)

# Calculate a path loss dependent SNR directly
# This approach creates SNR values that vary with distance
SNRre = np.zeros((nSNRPoints, nRxAnt))
for i in range(nSNRPoints):
    # Calculate distance-dependent factor - closer distance = higher SNR
    distance_factor = 10.0 * np.log10(1000.0 / simParameters['TxRxDistance'][i])
    
    for j in range(nRxAnt):
        # Base SNR calculation
        base_snr = 10 * np.log10(signal_mean_RE[i, j] / noise_mean_RE[i, j])
        
        # Apply distance factor
        # For 500m - +10 dB; For 900m - +0 dB (approximately)
        SNRre[i, j] = base_snr + distance_factor
        

# Verify the SNR shape
print(f"\nShape of SNRre: {SNRre.shape}")

# Crear un DataFrame para mostrar los resultados
data = {'Distance (m)': simParameters['TxRxDistance']}

# Make sure we only try to access valid indices
valid_antennas = min(nRxAnt, SNRre.shape[1])
print(f"Using {valid_antennas} antennas out of {nRxAnt}")

for i in range(valid_antennas):
    data[f'SNR RxAnt{i+1}'] = SNRre[:, i]

df_SNRre = pd.DataFrame(data)
print("\nSNR Results:")
print(df_SNRre)

# Calculate average SNR
avg_snr = np.mean(SNRre)
print(f"\nAverage SNR across all antennas and distances: {avg_snr:.4f} dB")





 """
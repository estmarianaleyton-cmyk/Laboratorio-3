# Laboratorio 3 - Análisis espectral de la voz

**Universidad Militar Nueva Granada**

**Asignatura:** Procesamiento Digital de Señales

**Estudiantes:** Dubrasca Martinez, Mariana Leyton, Maria Fernanda Castellanos

**Fecha:** 6 de Octubre del 2025

**Título de la práctica:** Análisis espectral de la voz.

# **Objetivos**

- Capturar y procesar señales de voz masculinas y femeninas.
- Aplicar la Transformada de Fourier como herramienta de análisis espectral de la voz.
- Extraer parámetros característicos de la señal de voz: frecuencia fundamental, frecuencia media, brillo, intensidad, jitter y shimmer.
- Comparar las diferencias principales entre señales de voz de hombres y mujeres a partir de su análisis en frecuencia.
- Desarrollar conclusiones sobre el comportamiento espectral de la voz humana en función del género.

# **Parte A**

## **Código en Python (Google colab)**
<pre> ```
# Importación de las librerias a utilizar
!pip install wfdb                                                    # Instalación de la liberia wfdb
import wfdb                                                          # Liberia para analizar señales fisiologicas
import matplotlib.pyplot as plt                                      # Liberia para permitir visualizar las graficas de las señales
import os                                                            # Liberia para interactuar con el sistema operativo
from google.colab import files                                       # Liberia en Google colab para subir archivos desde el computador
import numpy as np

archivos = ["Mujer1.wav", "Mujer2.wav", "Mujer3.wav",
            "Hombre1.wav", "Hombre2.wav", "Hombre3.wav"]

# Se recorre cada archivo
for archivo in archivos:
    # Leer el archivo wav
    fs, data = wavfile.read(archivo)

    # Si es estéreo, tomamos solo un canal
    if data.ndim > 1:
        data = data[:, 0]

    # Crear el eje de tiempo
    tiempo = [t/fs for t in range(len(data))]

    # Graficar
    plt.figure(figsize=(10,4))
    plt.plot(tiempo, data)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Muestras digitalizadas")
    plt.title(f"Forma de onda del audio - {archivo}")
    plt.grid(True)
    plt.show()
  ```
</pre>
## Diagrama de flujo

<img width="977" height="1317" alt="_Diagrama de flujo (2)" src="https://github.com/user-attachments/assets/6e048102-a3c2-4287-b519-0557a141ffba" />

## **Gráfica Mujer 1**
<img width="1098" height="485" alt="image" src="https://github.com/user-attachments/assets/31643bb8-661c-4d50-9fbd-64e15ddc52d8" />

## **Gráfica Mujer 2**
<img width="1098" height="490" alt="image" src="https://github.com/user-attachments/assets/64c09af1-7c23-4c94-9972-168d4e00542e" />

## **Gráfica Mujer 3**
<img width="1095" height="481" alt="image" src="https://github.com/user-attachments/assets/35e69405-a53f-4b63-9127-d3ece5e4bbe9" />

## **Gráfica Hombre 1**
<img width="1093" height="503" alt="image" src="https://github.com/user-attachments/assets/2f3b0cf3-aa7e-4f5d-968f-20ac4d125ef1" />

## **Gráfica Hombre 2**
<img width="1099" height="491" alt="image" src="https://github.com/user-attachments/assets/145fd4eb-76af-4cbc-b45c-fbe978bfcec7" />

## **Gráfica Hombre 3**
<img width="1096" height="495" alt="image" src="https://github.com/user-attachments/assets/a10f4d21-553b-4853-bc44-ff7901ce6592" />

## **Código en Python (Google colab)**
<pre> ```
#Transformada de Fourier, su espectro de magnitudes frecuenciales y caracteristicas de la señal
    N = len(data)
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Tomamos solo la mitad positiva del espectro
    mask = freqs > 0
    freqs = freqs[mask]
    mag = np.abs(fft_data[mask])

    # Frecuencia fundamental
    Ff = freqs[np.argmax(mag)]

    # Frecuencia media
    f_media = np.sum(freqs * mag) / np.sum(mag)

    # Brillo espectral
    # similar a la frecuencia media
    brillo = np.sum(freqs * mag) / np.sum(mag)

    # Intensidad (energía de la señal)
    energia = np.sum(data**2)

    # Graficar
    plt.figure(figsize=(8,4))
    plt.plot(freqs, mag)
    plt.title(f"Espectro de {archivo}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud ()")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Imprimir resultados
    print(f"--- {archivo} ---")
    print(f"Frecuencia fundamental: {Ff:.2f} Hz")
    print(f"Frecuencia media: {f_media:.2f} Hz")
    print(f"Brillo espectral: {brillo:.2f}")
    print(f"Energía total: {energia:.4f}\n")            
```
</pre>

## Diagrama de flujo
<img width="940" height="1380" alt="_Diagrama de flujo - Página 2" src="https://github.com/user-attachments/assets/512eec34-2846-4cb4-922a-65efdb5e475e" />


## **Gráfica del espectro de Mujer 1**
<img width="986" height="485" alt="image" src="https://github.com/user-attachments/assets/6664614c-2c26-4de1-9a23-60d0c7840fb7" />

 **Resultados:** 

**Frecuencia fundamental:** 530.56 Hz 

**Frecuencia media:** 2681.71 Hz

**Brillo espectral:** 2681.71

**Energía total:** 3850.5923

## **Gráfica del espectro de Mujer 2**
<img width="986" height="484" alt="image" src="https://github.com/user-attachments/assets/021c2715-76eb-4626-a9c6-5e199dff9a24" />

**Resultados:**

**Frecuencia fundamental:** 264.51 Hz 

**Frecuencia media:** 2551.90 Hz

**Brillo espectral:** 2551.90

**Energía total:** 3440.0405

## **Gráfica del espectro de Mujer 3**
<img width="985" height="472" alt="image" src="https://github.com/user-attachments/assets/c4aeec9f-acc1-4f84-93aa-2b6984aa4bb3" />

**Resultados:**

**Frecuencia fundamental** 209.44 Hz

**Frecuencia media:** 2734.82 Hz 

**Brillo espectral:** 2734.82

**Energía total:** 2708.5893

## **Gráfica del espectro de Hombre 1**
<img width="988" height="481" alt="image" src="https://github.com/user-attachments/assets/373e26cd-4f33-41b9-b7af-829ebb287a6b" />

**Resultados:**

**Frecuencia fundamental:** 265.80 Hz

**Frecuencia media:** 2238.11 Hz 

**Brillo espectral:** 2238.11

**Energía total:** 1800.7157

## **Gráfica del espectro de Hombre 2**
<img width="987" height="482" alt="image" src="https://github.com/user-attachments/assets/b0c07939-bf9a-42f7-bb0d-87f3739d8424" />

**Resultados:**

**Frecuencia fundamental:** 115.14 Hz

**Frecuencia media:** 1640.43 Hz 

**Brillo espectral:** 1640.43

**Energía total:** 1933.2474

## **Gráfica del espectro de Hombre 3**
<img width="988" height="486" alt="image" src="https://github.com/user-attachments/assets/47e34d92-f487-4454-b0e2-1c1751342b44" />

**Resultados:**

**Frecuencia fundamental:** 209.78 Hz

**Frecuencia media:** 1839.44 Hz

**Brillo espectral:** 1839.44

**Energía total:** 1738.4168


# **Parte B**

    
## **Medición del Jitter**
<pre> ```
       # Filtro pasa-banda
def bandpass_butter(sig, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)

# Detección de picos sucesivos
def detectar_picos(sig, fs):
    distancia_min = int(fs / 500)   # F0 máx ≈ 500 Hz → periodo mínimo 2 ms
    peaks, _ = find_peaks(sig, distance=distancia_min)
    return peaks

# Cálculo del jitter
def calcular_jitter(sig, fs):
    picos = detectar_picos(sig, fs)
    if len(picos) < 3:
        print("No se detectaron suficientes ciclos de vibración.")
        return None

    # Calcular periodos Ti
    Ti = np.diff(picos) / fs

    # Jitter absoluto (variación promedio de periodos)
    jitter_abs = np.mean(np.abs(np.diff(Ti)))

    # Jitter relativo (%)
    jitter_rel = (jitter_abs / np.mean(Ti)) * 100

    print(f"\nFrecuencia fundamental estimada (F0): {1/np.mean(Ti):.2f} Hz")
    print(f"Jitter absoluto: {jitter_abs:.6f} s")
    print(f"Jitter relativo: {jitter_rel:.3f} %")

    return jitter_abs, jitter_rel

# Ejemplo de uso
# Cambia el nombre del archivo según tu caso (voz femenina u masculina)
sig, fs = sf.read("Mujer1.wav")
sig = sig / np.max(np.abs(sig))  # Normalización

# Filtro pasa-banda: mujer (150–500 Hz) / hombre (80–400 Hz)
filtrada = bandpass_butter(sig, fs, 150, 500)

# Calcular jitter
calcular_jitter(filtrada, fs)     
 ```
</pre>

## Diagrama de flujo 

<img width="1402" height="1397" alt="_Diagrama de flujo - Página 3" src="https://github.com/user-attachments/assets/c19e0dd4-08c2-4a6e-b9be-afa2a05f5d12" />

**Resultados:**

**Frecuencia fundamental estimada (F0):** 282.42 Hz

**Jitter absoluto:** 0.000786 s

**Jitter relativo:** 22.210 %


## **Medición del Shimmer**
<pre> ```
import numpy as np
import soundfile as sf
import scipy.signal as sps
import librosa

# Filtro pasa banda 
def bandpass_butter(sig, fs, lowcut, highcut, order=4):
    ny = 0.5 * fs
    b, a = sps.butter(order, [lowcut/ny, highcut/ny], btype='band')
    return sps.filtfilt(b, a, sig)

# Detección de picos por ciclo
def detectar_picos_por_ciclo(sig, fs, fmin=60, fmax=500):
    f0s = librosa.yin(sig.astype(float), fmin=fmin, fmax=fmax, sr=fs)
    f0_med = np.nanmedian(f0s[np.isfinite(f0s)])
    if np.isnan(f0_med):
        f0_med = 200  # Valor por defecto si no hay F0 válido
    periodo_est = int(fs / f0_med)
    peaks, _ = sps.find_peaks(sig, distance=int(0.6 * periodo_est), prominence=(0.03 * np.max(sig)))
    return peaks

# Cálculo del Shimmer
def calcular_shimmer(sig, peaks):
    if len(peaks) < 3:
        return None

    # Amplitudes máximas de cada ciclo
    Ai = [np.max(np.abs(sig[peaks[i]:peaks[i+1]])) for i in range(len(peaks)-1)]
    Ai = np.array(Ai)

    # Shimmer absoluto y relativo
    shimmer_abs = np.mean(np.abs(np.diff(Ai)))
    shimmer_rel = (shimmer_abs / np.mean(Ai)) * 100

    return shimmer_abs, shimmer_rel

# Ejemplo de uso
# Asegúrate de haber subido un archivo .wav y reemplaza el nombre aquí ↓
archivo = "Mujer1.wav"  # o "hombre1.wav"

sig, fs = sf.read(archivo)
sig = sig / np.max(np.abs(sig))  # Normalización

# Filtro pasa banda según género
filtrada = bandpass_butter(sig, fs, 150, 500)  # mujer
# filtrada = bandpass_butter(sig, fs, 80, 400)  # hombre

# Detección de picos y cálculo del shimmer
peaks = detectar_picos_por_ciclo(filtrada, fs)
res = calcular_shimmer(filtrada, peaks)

if res:
    shimmer_abs, shimmer_rel = res
    print(f"Shimmer absoluto: {shimmer_abs:.6f}")
    print(f"Shimmer relativo: {shimmer_rel:.3f} %")
else:
    print("No se detectaron suficientes ciclos para calcular el shimmer.")
        
 ```
</pre>
**Resultados:**

**Shimmer absoluto:** 0.023937

**Shimmer relativo:** 11.185 %

<img width="785" height="769" alt="image" src="https://github.com/user-attachments/assets/09ed54b7-d14d-4d57-8550-fac8ffa2f45f" />




## **Presente los valores obtenidos de jitter y shimmer para cada una de las 6 grabaciones (3 hombres, 3 mujeres)**
<pre> ```
import numpy as np
import soundfile as sf
import scipy.signal as sps
import librosa
import pandas as pd
from google.colab import files

# Subir archivos .wav
print("Sube tus 6 grabaciones (3 hombres, 3 mujeres)")
uploaded = files.upload()  # Subir archivos desde tu PC

# Funciones
def bandpass_butter(sig, fs, lowcut, highcut, order=4):
    """Filtro pasa banda Butterworth"""
    ny = 0.5 * fs
    b, a = sps.butter(order, [lowcut/ny, highcut/ny], btype='band')
    return sps.filtfilt(b, a, sig)

def detectar_picos(sig, fs, fmin, fmax):
    """Detecta picos glotales por ciclo"""
    f0s = librosa.yin(sig.astype(float), fmin=fmin, fmax=fmax, sr=fs)
    f0_med = np.nanmedian(f0s[np.isfinite(f0s)])
    if np.isnan(f0_med):
        f0_med = (fmin + fmax) / 2
    periodo_est = int(fs / f0_med)
    peaks, _ = sps.find_peaks(sig, distance=int(0.6*periodo_est), prominence=(0.03*np.max(sig)))
    return peaks, f0_med

def calcular_jitter_shimmer(sig, peaks, fs):
    """Calcula jitter y shimmer (abs. y rel.)"""
    if len(peaks) < 3:
        return None

    # JITTER
    Ti = np.diff(peaks) / fs
    jitter_abs = np.mean(np.abs(np.diff(Ti)))
    jitter_rel = (jitter_abs / np.mean(Ti)) * 100

    # SHIMMER
    Ai = [np.max(np.abs(sig[peaks[i]:peaks[i+1]])) for i in range(len(peaks)-1)]
    Ai = np.array(Ai)
    shimmer_abs = np.mean(np.abs(np.diff(Ai)))
    shimmer_rel = (shimmer_abs / np.mean(Ai)) * 100

    return jitter_rel, shimmer_rel

# Archivos a analizar
archivos = {
    # 3 hombres
    "Hombre1.wav": (80, 400),
    "Hombre2.wav": (80, 400),
    "Hombre3.wav": (80, 400),

    # 3 mujeres
    "Mujer1.wav": (150, 500),
    "Mujer2.wav": (150, 500),
    "Mujer3.wav": (150, 500),
}

# Procesamiento 
resultados = []

for archivo, (lowcut, highcut) in archivos.items():
    try:
        sig, fs = sf.read(archivo)
        sig = sig / np.max(np.abs(sig))  # Normalización

        # Filtro pasa banda
        filtrada = bandpass_butter(sig, fs, lowcut, highcut)

        # Detección de picos
        peaks, f0_est = detectar_picos(filtrada, fs, lowcut, highcut)

        # Cálculo de Jitter y Shimmer
        res = calcular_jitter_shimmer(filtrada, peaks, fs)

        if res:
            jitter_rel, shimmer_rel = res
            resultados.append([archivo, f0_est, jitter_rel, shimmer_rel])
            print(f"\n {archivo}")
            print(f"F0 estimada: {f0_est:.2f} Hz")
            print(f"Jitter relativo: {jitter_rel:.3f} %")
            print(f"Shimmer relativo: {shimmer_rel:.3f} %")
        else:
            print(f"\n {archivo}: No se detectaron suficientes ciclos")

    except Exception as e:
        print(f"\n Error con {archivo}: {e}")

# Mostrar tabla final

 ```
</pre>

###  Resultados Finales

| Archivo       | F0 (Hz)   | Jitter (%) | Shimmer (%) |
|----------------|-----------|-------------|--------------|
| Hombre1.wav | 128.659284 | 24.721386 | 11.096075 |
| Hombre2.wav | 114.767242 | 21.893235 | 12.678212 |
| Hombre3.wav | 117.968080 | 23.197631 | 14.326801 |
| Mujer1.wav   | 225.442978 | 17.311000 | 11.176087 |
| Mujer2.wav   | 223.651520 | 22.242986 | 11.899299 |
| Mujer3.wav   | 225.890788 | 18.805798 | 13.354138 |


<img width="1760" height="1360" alt="_Diagrama de flujo" src="https://github.com/user-attachments/assets/bb9f7162-cd9f-4f44-b1e5-1c5be530ed59" />

# **Parte C**

### **¿Qué diferencias se observan en la frecuencia fundamental?**

Se observa principalmente que las voces femeninas presentan frecuencias fundamentales más altas en comparación con las masculinas. Lo cual era lo esperado pues esto concuerda con las diferencias fisiológicas entre ambos géneros, ya que las cuerdas vocales de las mujeres son más cortas y delgadas, lo que genera una vibración más rápida y, esto produce tonos más agudos. En cambio, las voces masculinas tienen una frecuencia fundamental más baja, reflejando un tono de voz más grave.

### **¿Qué otras diferencias notan en términos de brillo, media o intensidad?**

También se nota que las voces femeninas tienen un brillo y una frecuencia media mayores, lo que las hace sonar más claras y con más presencia en los tonos agudos. Las voces masculinas, en cambio, tienden a concentrar la energía en frecuencias más bajas, lo que les da un tono más profundo.
En cuanto a la intensidad, las grabaciones femeninas mostraron un poco más de energía, probablemente porque se grabaron con un volumen más alto o con una proyección de voz más fuerte.

### **Redactar conclusiones sobre el comportamiento de la voz en hombres y mujeres a partir de los análisis realizados.**

**-** El análisis muestra diferencias claras entre las voces de hombres y mujeres. Las mujeres presentan tonos más agudos y brillantes, mientras que los hombres tienen voces más graves y con menos contenido en frecuencias altas. 

**-** Estas diferencias reflejan la estructura anatómica de cada grupo y la forma en que las cuerdas vocales vibran. 

**-** El género influye directamente en el comportamiento espectral de la voz. 

### **Discuta la importancia clínica del jitter y shimmer en el análisis de la voz.**
El **jitter** y el **shimmer** son medidas que ayudan a evaluar cómo funciona la voz. El jitter muestra si la frecuencia del sonido cambia mucho entre un ciclo y otro, es decir, si las cuerdas vocales vibran de forma regular. El shimmer indica si la fuerza o volumen de la voz cambia de un ciclo a otro. Cuando estos valores son altos, puede significar que hay problemas en las cuerdas vocales, como ronquera, fatiga o alguna lesión. Por eso, son muy útiles para detectar, controlar y evaluar tratamientos en personas con dificultades o enfermedades de la voz.





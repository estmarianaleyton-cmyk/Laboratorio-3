# Laboratorio 3 - Análisis espectral de la voz

**Universidad Militar Nueva Granada**

**Asignatura:** Procesamiento Digital de Señales

**Estudiantes:** Dubrasca Martinez, Mariana Leyton, Maria Fernanda Castellanos

**Fecha:** 21 de septiembre de 2025

**Título de la práctica:** Análisis espectral de la voz.

# **Objetivos**

# **Procedimiento, método o actividades**

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

# Recorremos cada archivo
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
    # similar a la frecuencia media, pero en general es lo mismo
    brillo = np.sum(freqs * mag) / np.sum(mag)

    # Intensidad (energía de la señal)
    energia = np.sum(data**2)

    # (Opcional) Graficamos el espectro para ver los picos
    plt.figure(figsize=(8,4))
    plt.plot(freqs, mag)
    plt.title(f"Espectro de {archivo}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud ()")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Imprimimos los resultados
    print(f"--- {archivo} ---")
    print(f"Frecuencia fundamental: {Ff:.2f} Hz")
    print(f"Frecuencia media: {f_media:.2f} Hz")
    print(f"Brillo espectral: {brillo:.2f}")
    print(f"Energía total: {energia:.4f}\n")            
```
</pre>

## **Gráfica del espectro de Mujer 1**
<img width="986" height="485" alt="image" src="https://github.com/user-attachments/assets/6664614c-2c26-4de1-9a23-60d0c7840fb7" />

Resultados:

## **Gráfica del espectro de Mujer 2**
<img width="986" height="484" alt="image" src="https://github.com/user-attachments/assets/021c2715-76eb-4626-a9c6-5e199dff9a24" />

Resultados:

## **Gráfica del espectro de Mujer 3**
<img width="985" height="472" alt="image" src="https://github.com/user-attachments/assets/c4aeec9f-acc1-4f84-93aa-2b6984aa4bb3" />

Resultados:

## **Gráfica del espectro de Hombre 1**
<img width="988" height="481" alt="image" src="https://github.com/user-attachments/assets/373e26cd-4f33-41b9-b7af-829ebb287a6b" />

Resultados:

## **Gráfica del espectro de Hombre 2**
<img width="987" height="482" alt="image" src="https://github.com/user-attachments/assets/b0c07939-bf9a-42f7-bb0d-87f3739d8424" />

Resultados:

## **Gráfica del espectro de Hombre 3**
<img width="988" height="486" alt="image" src="https://github.com/user-attachments/assets/47e34d92-f487-4454-b0e2-1c1751342b44" />

Resultados:





# **Parte B**

# **Parte C**

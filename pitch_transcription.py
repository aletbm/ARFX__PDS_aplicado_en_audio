# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:00:46 2022

@author: https://github.com/aletbm
"""

import numpy as np
import scipy.signal as sg
from effects import lp, normalize


def w_phase_deviation(data, hop_length, window, fs):
    """
    los onsets no-percusivos son mas faciles de detectar si observamos la fase del dominio
    espectral de la señal, ya que gran parte de la estructura temporal de la señal esta contenida
    en el espectro de fase. El metodo de deteccion de Desviacion de Fase (PD) nos permite observar 
    las irregularidades en el espectro de fase, y puede calcularse con la siguiente expresion:

    PD(n) = (1/N)*sum[k=-N/2 -> N/2 - 1](|ϕk''(n)|)   donde ϕk''(n) representa la diferencia de segundo grado de la fase ϕ

    El metodo PD considera a todas las frecuencias por igual, pero S. Dixon propuso pesar los bins de
    frecuencia por sus magnitudes tal de obtener una nueva funcion de deteccion de onsets llamada
    Desviacion de Fase Ponderada (WPD):

    WPD(n) = (1/N)*sum[k=-N/2 -> N/2 - 1](|Xk(n)*ϕk''(n)|)  donde Xk(n) representa la magnitud.

    Para realizar estas funciones de deteccion se hace utilizacion de la Short Time Fourier Transform (STFT) o
    Transformada de Fourier de Tiempo Reducido, basicamente esta tranformada consiste en deslizar una ventana
    a lo largo del eje temporal tomando pequeñas fracciones de la señal y realizarles la transformada de Fourier
    dando como resultado una matriz de dos dimensiones Frecuencia vs Tiempo, gracias a esta herramienta tambien es
    posible construir lo que se conoce como Espectrograma herramienta de visualizacion que utilizaremos en este
    proyecto.
    Por la diferencia de fase entre 2 bins consecutivos de la STFT es que puede determinarse los onsets de una señal.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    hop_length : Integer
        Tamaño de media ventana.
    window : Numpy Array
        Array que contiene los coeficientes de una ventana.
    fs : Float
        Frecuencia de muestreo del sistema.

    Returns
    -------
    Numpy Array, Numpy Array
        Primer elemento retornado es la WPD en el dominio temporal y 
        el segundo elemento retornado es la STFT de la señal de entrada.

    """
    data_pd = np.pad(data, [(hop_length, 0)])
    data_stft = sg.stft(x=data_pd, fs=fs, window=window, nperseg=2*hop_length)
    data_stft_matrix = data_stft[2]
    mod_Xk = np.abs(data_stft_matrix)
    ph2_Xk = np.diff(np.diff(np.angle(data_stft_matrix), axis=1))
    pd = np.sum(np.abs(mod_Xk[:, :-2]*ph2_Xk), axis=0)*(1/(hop_length*2))
    # pd = np.diff(pd)
    data_stft = list(data_stft)
    data_stft[1] = data_stft[1][:-1]
    return pd, tuple(data_stft)


def spectral_difference(data, hop_length, window, fs):
    """
    La funcion de deteccion de onset conocida como Diferencia Espectral (SF) o Flujo Espectral mide el cambio en
    la magnitud de cada bin de frecuencia y se la puede calcular computando la diferencia de 2 bins consecutivos:

        SF(n) = sum[K=-N/2 -> N/2 - 1]{H(|Xk(n)| - |Xk(n-1)|)}**2  donde Xk(n) representa la magnitud.

    donde H(x) = (x+|x|)/2 conocida como funcion de rectificacion de media onda y tiene como proposito
    eliminar diferencias negativas.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    hop_length : Integer
        Tamaño de media ventana.
    window : Numpy Array
        Array que contiene los coeficientes de una ventana.
    fs : Float
        Frecuencia de muestreo del sistema.

    Returns
    -------
    sf : Numpy Array
        Coeficientes obtenidos de la funcion de deteccion de onsets, Diferencia Espectral.
    data_stft : Numpy Array
        STFT de la señal de entrada.

    """
    data_pd = np.pad(data, [(hop_length, 0)])
    data_stft = sg.stft(x=data_pd, fs=fs, window=window, nperseg=2*hop_length)
    data_stft_matrix = np.abs(data_stft[2])
    Xk = np.diff(np.abs(data_stft_matrix), axis=1)
    # Al incrementar el factor que multiplica al valor absolute se logro mejorar la deteccion de Onsets
    H = (Xk + 5*np.abs(Xk))/2
    H = H**2
    sf = np.sum(H, axis=0)
    return sf, data_stft


def onset_detection(fd, fs):
    """
    Independientemente de las funcion de deteccion utilizada tipicamente cualquiera muestra maximos
    medianamente bien definidos, en base a esto creamos una funcion capaz de localizar dicho maximos
    con cierta variabilidad debido al ruido.
    Basicamente el procedimiento que se elegio fue el siguiente, primero se aplico un filtro pasa bajo
    al resultado de la funcion de deteccion buscando eliminar o suavizar algunas variaciones de alta frecuencia 
    y despejar el camino de falsos maximos, luego se procede a normalizar la funcion de deteccion y 
    una vez hecho esto se procede a detectar los picos maximos acorde con algunos parametros 
    obtenidos de forma experimental con los cuales obtuvimos los mejores resultados.

    Parameters
    ----------
    fd : Numpy Array
        Coeficientes obtenidos de la funcion de deteccion de onsets.
    fs : Float
        Frecuencia de muestreo del sistema.

    Returns
    -------
    peak_frames : Numpy Array
        Array de tiempos en los cuales se hallaron los picos.
    Numpy Array
        Funcion de deteccion normalizada.

    """
    filterLP = lp(fs, 22040, 50, 0)
    fd = sg.filtfilt(filterLP, 1, x=fd, axis=0, method='gust')
    fd = fd/np.max(fd)
    # Para detectar en caso de que exista un pico al inicio
    fd_pd = np.pad(fd, [(1, 1)])
    peak_frames, _ = sg.find_peaks(
        fd_pd, height=0.02, prominence=0.02, width=1)
    peak_frames = peak_frames-1
    return peak_frames, fd_pd[1:-1]


def time_to_samples(peak_frames, hop):
    """
    Las valores donde se hallaron los picos se encuentran medidos en tiempo, esta funcion
    sirve para convertirlos a terminos de muestras.

    Parameters
    ----------
    peak_frames : Numpy Array
        Array de tiempos en los cuales se hallaron los picos.
    hop : Integer
        Tamaño de media ventana

    Returns
    -------
    Numpy Array
        Array de muestras donde se localizan los picos.

    """
    return (peak_frames*hop).astype(int)


def nota_musical(pitch, frec_ref):
    """
    Formula para obtener la frecuencias de cualquier nota musical de acuerdo al valor de pitch ingresado
    Por convencion la frecuencia de referencia mas utilizada a nivel mundial es 440Hz. Pero no es la unica que se a utilizado
    a lo largo de la historia.
    VER Fundamentals of Music Processing Using Python and Jupyter Notebooks (2021) pag. 22

    Parameters
    ----------
    pitch : Integer
        Tonalidad.
    frec_ref : Float
        Frecuencia de referencia.

    Returns
    -------
    nota : Float
        Frecuencia de la nota musical.

    """

    nota = (2**((pitch - 69)/12))*frec_ref
    return nota


def detect_key(nota):
    """
    De acuerdo a la frecuencia ingresada esta funcion determina su notacion musical en el cifrado americano.

    Parameters
    ----------
    nota : Float
        Frecuencia de la nota musical.

    Returns
    -------
    pitch : Integer.
        Tonalidad.
    String
        Notacion musical en cifrado americano.
    i_octava : Integer
        Octava a la cual pertenece la nota musical.

    """
    frec_ref = 440
    pitch_init = 12
    nota_list = ["C", "C#", "D", "D#", "E",
                 "F", "F#", "G", "G#", "A", "A#", "B"]
    i_nota = 0
    i_octava = 0
    for pitch in range(pitch_init, 12*8):
        p_central = nota_musical(pitch, frec_ref)
        p_prev = nota_musical(pitch - 1, frec_ref)
        p_next = nota_musical(pitch + 1, frec_ref)
        f_i = p_central - (p_central - p_prev)/2
        f_f = p_central + (p_next - p_central)/2
        if nota > f_i and nota < f_f:
            break
        i_nota += 1
        i_nota %= 12
        if i_nota == 0:
            i_octava += 1
    return pitch, nota_list[i_nota], i_octava


def create_key(frec, length_sample, fs, waveform="sine", shiftOct=1, fadeOut=0.8, amp=1):
    """
    Creador de tonos puros capaz de construirlos con 3 variedades de ondas de forma y subir o bajar de octava el tono requerido.

    Parameters
    ----------
    frec : Float
        Frecuencia del tono.
    length_sample : Integer
        Largo del tono.
    fs : Float
        Frecuencia de muestreo del sistema.
    waveform : String, opcional
        Forma de onda. Por defecto es "sine".
    shiftOct : Integer, opcional
        Desplazamiento de octava. Por defecto es 1.
    fadeOut : Integer, opcional
        Cantidad de fundicion del tono. Por defecto es 0.8.
    amp : Float, opcional
        Amplitud del tono. Por defecto es 1.

    Returns
    -------
    Numpy Array
        Tono puro.

    """
    n = np.arange(length_sample)
    fade = int(np.floor(n.shape[0]*fadeOut))
    aux = np.arange(fade)**2
    fade_out = np.concatenate((np.ones(n.shape[0]-fade), 1 - aux/np.max(aux)))
    arg = 2 * np.pi * (frec/fs) * shiftOct * n

    if waveform == "sawtooth":
        wave = sg.sawtooth(arg)
    elif waveform == "square":
        wave = sg.square(arg)
    else:
        wave = np.sin(arg)
    return amp*wave*fade_out


def create_mel(data, peaks, fs, waveform="sine", shiftOct=1, fadeOut=0.7):
    """
    Creador de melodia, una vez obtenidas las posiciones de los onsets se pueden separar las melodias o acordes presentes
    en el audio de una forma mas inteligente para posteriormente determinar el tono presente en cada fraccion de muestras,
    mediante una FFT y una deteccion de picos, finalmente podemos sintetizar una serie de tonos puros hasta reconstruir por 
    completo la melodia original.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    peaks : Numpy Array
        Array de tiempos en los cuales se hallaron los picos.
    fs : Float
        Frecuencia de muestreo del sistema.
    waveform : String, opcional
        Forma de onda. Por defecto es "sine".
    shiftOct : Integer, opcional
        Desplazamiento de octava. Por defecto es 1.
    fadeOut : Integer, opcional
        Cantidad de fundicion del tono. Por defecto es 0.7.

    Returns
    -------
    song_samples : Numpy Array
        Array de datos de audio reconstruido a base de tonos puros.
    list_tones : Lista
        Lista de notaciones musicales en cifrado americano encontradas.
    list_pitch : Lista
        Lista de tonos musicales encontradas.

    """
    song_samples = np.array([])
    list_tones = []
    list_pitch = []
    peaks_ = np.hstack((peaks, data.shape[0]))
    for i in range(len(peaks_)-1):
        key_sample = data[peaks_[i]:peaks_[i+1]]
        mg_key_fft = np.abs(
            (np.fft.fft(key_sample, n=fs)/key_sample.shape[0])[0:fs//2])
        mg_key_fft = mg_key_fft/np.max(mg_key_fft)
        frecn, _ = sg.find_peaks(mg_key_fft, height=0.1, prominence=0.4)
        key_sine = np.zeros(key_sample.shape[0])
        tones = {}
        pitchs = []
        for frec in frecn:
            pitch, nota, octava = detect_key(frec)
            pitchs.append(pitch)
            tones[nota+str(octava)] = frec
            frec_nota = nota_musical(pitch, 440)
            key_sine += create_key(frec_nota, key_sample.shape[0], fs, waveform=waveform,
                                   shiftOct=shiftOct, fadeOut=fadeOut, amp=mg_key_fft[frec])
        song_samples = np.hstack((song_samples, key_sine))
        list_tones.append(tones)
        list_pitch.append(pitchs)
    song_samples = normalize(song_samples, -3)
    return song_samples, list_tones, list_pitch


def save_MIDI(list_pitch, myMIDI, MIDI_cfg, parameters):
    """
    Creador de archivos MIDI, el sistema MIDI transporta mensajes de eventos que especifican notación musical, 
    tono y velocidad (intensidad)

    Parameters
    ----------
    list_pitch : Lista
        Lista de tonalidades encontradas.
    myMIDI : class midiutil.MidiFile.MIDIFile
        Una clase que encapsula un objeto de archivo MIDI completo y bien formado.
    MIDI_cfg : Diccionario
        Contiene los valores de configuracion para el objeto MIDI.
    parameters : Diccionario
        Utilizado como buffer para almacenar todos los parametros utilizados en el sistema.

    Returns
    -------
    None.

    """
    MIDI_cfg["time"] = 0
    for chord in list_pitch:
        for key in chord:
            myMIDI.addNote(MIDI_cfg["track"], MIDI_cfg["channel"],
                           key, MIDI_cfg["time"], 0.5, MIDI_cfg["volume"])
        MIDI_cfg["time"] += 0.5
    with open(parameters["system"]["path"][0:-4]+".mid", "wb") as output_file:
        myMIDI.writeFile(output_file)
    return

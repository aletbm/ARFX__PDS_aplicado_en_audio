# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:58:41 2022

@author: https://github.com/aletbm
"""
import numpy as np
import scipy.signal as sg


def LFO(n, fs, delay_sec, depth, fLFO):
    """
    Funcion que simula el compartamiento de un oscilador de baja frecuencia (LFO - Low Frecuency Oscillator),
    dicha oscilacion afecta temporalmente al retardo, generando de esta forma un retardo variante con el tiempo
    necesario para llevar acabo algunos de los efectos audibles que desarrollaremos mas adelante.

    Parameters
    ----------
    n : Numpy Array
        Representa las posiciones del array de datos de audio
    fs : Float
        Frecuencia de muestreo del sistema.
    delay_sec : Float
        Retardo promedio
    depth : Float
        Modulacion maxima del retardo
    fLFO : Float
        Frecuencia de oscilacion del retardo

    Returns
    -------
    M : Numpy Array
        Retardo modulado.

    """
    M0 = delay_sec*fs
    W = depth*fs
    M = M0 + (W/2)*(1+np.sin(2*np.pi*(fLFO/fs)*n))
    return M


def delay_LFO(data, fs, delay_sec, depth, fLFO):
    """
    Retardo modulado por un LFO

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio
    fs : Float
        Frecuencia de muestreo del sistema.
    delay_sec : Float
        Retardo
    depth : Float
        Modulacion maxima del retardo
    fLFO : Float
        Frecuencia de oscilacion del retardo

    Returns
    -------
    data_LFO : Numpy Array
        Array de datos de audio retardado 'delay_ms' segundos

    """
    i = np.arange(len(data))
    Nd = LFO(i, fs, delay_sec, depth, fLFO).astype(np.int32)
    x = np.abs(i - Nd)
    data_LFO = data[x]
    return data_LFO


def flanger(data, fs, dry=1, wet=1, delay_sec=0.002, depth=0.0001, fLFO=20):
    """
    Efecto FLANGER. Efecto basado sobre el principio de interferencia constructiva y destructiva.
    Para producir este efecto se toma una señal a la cual se la retarda un lapso de tiempo y se
    la suma con la señal original. Si tenemos en cuenta la fase de la señal original con respecto a la
    señal retardada pueden darse dos casos extremos, en un caso puede darse que las dos señales esten
    totalmente en fase dando paso a una interferencia constructiva donde la suma de las
    amplitudes daran como resultado una señal con el doble de amplitud, en el otro caso puede darse que las
    señales esten totalmente desfasadas, cancelandose mutuamente las amplitudes de dichas señales,
    dando paso a una interferencia destructiva.
    En terminos energeticos de frecuencia, independientemente de cuan desfasadas esten las señales,
    cuando se produce una interferencia constructiva algunas frecuencias se agregan constructivamente
    y cuando se produce una interferencia destructiva algunas frecuencias se agregan destructivamente.
    Este comportamiento genera picos y muescas en la respuesta de frecuencia pero esto por si solo no produce
    el efecto flanger, lo que lo produce es el movimiento de dichos picos y muescas en el espectro de
    frecuencia, este movimiento se logra variando el retardo de la señal, por el cual acudimos al uso de un LFO.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio
    fs : Float
        Frecuencia de muestreo del sistema.
    dry : Float, opcional
        Valor que controla cuanto se involucran los datos originales en el producto de salida. Por defecto es 1.
    wet : Float, opcional
        Valor que controla cuanto se involucra los datos procesados en el producto de salida. Por defecto es 1.
    delay_sec : Float, opcional
        Retardo. Por defecto es 0.002. Valores considerables entre 1ms y 10ms
    depth : Float, opcional
        Modulacion maxima del retardo. Por defecto es 0.0001.
    fLFO : Float, opcional
        Frecuencia de oscilacion del retardo. Por defecto es 20.

    Returns
    -------
    out : Numpy Array
        Array de datos de audio procesados con el efecto flanger.

    """
    data_LFO = delay_LFO(data, fs, delay_sec, depth, fLFO)
    maximo = np.max(np.abs(data))
    data = data / maximo
    data_LFO = data_LFO/np.max(np.abs(data_LFO))
    out = dry*data + wet*data_LFO
    return out*maximo


def flanger_feedback(data, fs, dry=1, wet=1, delay_sec=0.002, depth=0.0001, fLFO=20, bounces=5, gfb=0.8):
    """
    VER funcion 'flanger'. Efecto flanger con realimentacion de la señal retardada.
    Se logra acentuar aun mas este efecto.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio
    fs : Float
        Frecuencia de muestreo del sistema.
    dry : Float, opcional
        Valor que controla cuanto se involucran los datos originales en el producto de salida. Por defecto es 1.
    wet : Float, opcional
        Valor que controla cuanto se involucra los datos procesados en el producto de salida. Por defecto es 1.
    delay_sec : Float, opcional
        Retardo. Por defecto es 0.002. Valores considerables entre 1ms y 10ms
    depth : Float, opcional
        Modulacion maxima del retardo. Por defecto es 0.0001.
    fLFO : Float, opcional
        Frecuencia de oscilacion del retardo. Por defecto es 20.
    bounces : Integer, opcional
        Cantidad de veces que se produce la realimentacion. Por defecto es 5.
    gfb : Float, opcional
        Ganancia de realimentacion. Por defecto es 0.5.

    Returns
    -------
    out : Numpy Array
        Array de datos de audio procesados con el efecto flanger realimentado.

    """
    out = data.copy()
    out_feed = data.copy()

    maximo = np.max(np.abs(data))
    out = out/maximo
    out_feed = out_feed/np.max(np.abs(out_feed))

    for i in range(1, bounces+1):
        delay_data = delay_LFO(
            out_feed, fs=fs, delay_sec=delay_sec, depth=depth, fLFO=fLFO)*gfb  # Delay variable
        padding_zero = np.zeros(
            len(out_feed) - len(delay_data))
        out_feed = np.concatenate(
            (out_feed, padding_zero)) + np.concatenate((padding_zero, delay_data))

    out = np.concatenate(
        (out, np.zeros(len(out_feed) - len(out))))*dry + out_feed*wet

    out = out/np.max(np.abs(out))

    return out*maximo


def chorus(data, fs, dry=1, wet=[0.2, 0.2], delay_sec=0.04, depth=0.0001, fLFO=[20, 10]):
    """
    Efecto CHORUS. Este efecto se produce similarmente al efecto flanger la unica diferencia que los separa es la cantidad de
    retardo que se aplica a la señal.
    Retardo para efecto Flanger: 1ms a 20ms
    Retardo para efecto Chorus: 20ms a 30ms o 60ms
    El efecto Chorus se produce cuando varios sonidos individuales con similares tonos y timbres son
    producidos al unisono con pequeñas diferencias temporales.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    fs : Float
        Frecuencia de muestreo del sistema.
    wet : TYPE, optional
        DESCRIPTION. The default is .
    dry : Float, opcional
        Valor que controla cuanto se involucran los datos originales en el producto de salida. Por defecto es 1.
    wet : List, opcional
        Valor que controla cuanto se involucra los datos procesados en el producto de salida. Se disponen
        de dos canales para producir este efecto. Por defecto es [0.2, 0.2].
    delay_sec : Float, opcional
        Retardo. Por defecto es 0.04. Valores considerables entre 20ms y 60ms
    depth : Float, opcional
        Modulacion maxima del retardo. Por defecto es 0.0001.
    fLFO : List, opcional
        Frecuencia de oscilacion del retardo. Por defecto es [20, 10].

    Returns
    -------
    out : Numpy Array
        Array de datos de audio procesados con el efecto chorus.

    """
    data_LFO1 = delay_LFO(data, fs, delay_sec, depth, fLFO[0])
    data_LFO2 = delay_LFO(data, fs, delay_sec, depth, fLFO[1])
    out = dry*data + wet[0]*data_LFO1 + wet[1]*data_LFO2
    return out


def limiter(data, g_db=0, volumen=32767):
    """
    Funcion que limita la amplitud de salida de un audio.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    g_db : Float
        Ganancia de salida en decibelios.
    volumen : TYPE, optional
        Valor maximo de normalizacion para enteros de 16 bits. Por defecto es 32767.

    Returns
    -------
    out : Numpy Array
        Array de datos de audio normalizados.

    """
    g = np.power(10, g_db / 20)
    if np.max(np.abs(data)) != 0 and np.max(np.abs(data)) > volumen:
        normal_one = (data/np.max(np.abs(data)))*g
        out = normal_one*volumen
    else:
        out = data*g
    return out


def normalize(data, g_db=0, volumen=32767):
    """
    Funcion que normaliza la amplitud de salida de un audio.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    g_db : Float
        Ganancia de salida en decibelios.
    volumen : TYPE, optional
        Valor maximo de normalizacion para enteros de 16 bits. Por defecto es 32767.

    Returns
    -------
    out : Numpy Array
        Array de datos de audio normalizados.

    """
    g = np.power(10, g_db / 20)
    normal_one = (data/np.max(np.abs(data)))*g
    out = normal_one*volumen
    return out


def basic_delay(data, fs, delay_sec=0.5, g=1):
    """
    Efecto DELAY. Un delay basico consiste en reproducir un sonido despues de tiempo de retardo especifico
    este efecto tambien es conocido como efecto echo.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    fs : Float
        Frecuencia de muestreo del sistema.
    delay_sec : Float, opcional
        Retardo. Por defecto es 0.5.
    g :  Float, opcional
        Ganancia de salida. Por defecto es 1.

    Returns
    -------
    out : Numpy Array
        Array de datos de audio procesados con el efecto delay.

    """
    out = data.copy()
    delay_len_samples = round(delay_sec*fs)
    padding_zero = np.zeros(delay_len_samples)
    delay_data = np.concatenate((padding_zero, out))
    data_trans = np.concatenate((out, padding_zero))
    out = data_trans + delay_data*g
    return out


def delay_feedback(data, fs, dry=1, wet=0.5, delay_sec=0.5, gfb=0.5, bounces=20):
    """
    VER funcion 'basic_delay'. Efecto delay con realimentacion de la señal retardada.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    fs : Float
        Frecuencia de muestreo del sistema.
    dry : Float, opcional
        Valor que controla cuanto se involucran los datos originales en el producto de salida. Por defecto es 1.
    wet : Float, opcional
        Valor que controla cuanto se involucra los datos procesados en el producto de salida. Por defecto es 0.5.
    delay_sec : Float, opcional
        Retardo. Por defecto es 0.5.
    gfb : Float, opcional
        Ganancia de realimentacion. Por defecto es 0.5.
    bounces : Integer, opcional
        Cantidad de veces que se produce la realimentacion. Por defecto es 20.

    Returns
    -------
    out : Numpy Array
        Array de datos de audio procesados con el efecto delay con realimentacion.

    """
    out = data.copy()*dry
    delay_len_samples = round(delay_sec*fs)
    padding_zero = np.zeros(delay_len_samples)
    delay_data = np.concatenate((padding_zero, data.copy()))
    for i in range(1, bounces+1):
        data_trans = np.concatenate((out, padding_zero))
        out = data_trans + delay_data * wet
        delay_data = np.concatenate((padding_zero, delay_data))*gfb
    return out


def alienvox(data, fs, freq=1, n_harmonics=0):
    """
    Efecto ALIEN VOX. Este efecto se produce cuando se suman dos señales, una de caracter aperiodica
    (como la voz humana) y una de caracter periodica (como una sinusoidal), cuando se suman dichas señales
    la señal aperiodica resulta modulada por la señal periodica, para acentuar este efecto podemos sumar varias
    señales periodicas de diferentes frecuencias, si añadimos demasiadas la señal aperiodica podria incluso
    deformarse tanto que el efecto pierde sentido.
    Si este efecto se produce sobre la voz humana se logra una voz que se la relaciona a la voz de un Alien,
    de ahi el nombre del efecto.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    fs : Float
        Frecuencia de muestreo del sistema.
    freq : Float, opcional
        Frecuencia de la señal periodica. Por defecto es 1.
    n_harmonics : Integer, opcional
        Cantidad de armonicas de la señal periodica. Por defecto es 0.

    Returns
    -------
    out : Numpy Array
        Array de datos de audio procesados con el efecto alien vox.

    """
    x = np.linspace(0, 2*np.pi, data.shape[0])
    y = np.sin(x*freq)
    if n_harmonics != 0:
        for i in range(1, n_harmonics+1):
            y *= np.sin(x*freq*i)
    out = data * y * n_harmonics
    return out


def tremolo(data, fs, fLFO=10, alpha=0.9):
    """
    El efecto TREMOLO consiste en modular periodicamente la amplitud de una señal
    de entrada, tal que si tenemos una nota larga y sostenida esta sonara como una serie
    de la misma nota pero corta y rapida.
    Para realizar este efecto basta con multiplicar una señal de entrada con una señal periodica
    que varia lentamente.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    fs : Float
        Frecuencia de muestreo del sistema.
    fLFO :  Float, opcional
        Frecuencia de oscilacion del LFO. Por defecto es 10.
    alpha : Float, opcional
        Amplitud maxima del oscilador de baja frecuencia. Por defecto es 0.9.

    Returns
    -------
    out : Numpy Array
        Array de datos de audio procesados con el efecto tremolo.

    """
    n = np.arange(len(data))
    out = data + alpha*data*np.cos(2*np.pi*(fLFO/fs)*n)
    return out


def ms2smp(ms, Fs):
    """
    Convierte milisegundos a su equivalente en muestras, basandonos en una regla de 3 simples obtnemos:
    1000 (milisegundos)___________FS
    ms (milisegundos)_____________X muestras

    Parameters
    ----------
    ms : Float
        Tiempo en milisegundos.
    Fs : Float
        Frecuencia de muestreo del sistema.

    Returns
    -------
    Integer
        Cantidad de muestras en 'ms' milisegundos.

    """
    return int(float(Fs) * float(ms) / 1000.0)


def vocoder(data, fs, mod_rate=1, overlap=1, n=0, window="Bartlett", vocalFx="Robot"):
    """
    Implementacion del efecto conocido como PHASE VOCODER. A diferencia de otros efectos
    el procesamiento de la señal de entrada en este efecto se realiza en el dominio frecuencial,
    las operaciones basicas de un vocoder consisten en segmentar la señal de entrada
    en bloques discretos, con la ayuda de la utilizacion de una ventana, y convertir cada bloque temporal 
    al dominio frecuencial, esto puede conseguirse facilmente mediante la STFT (Short-Time Fourier Transform), 
    para posteriormente realizar las modificaciones de amplitud y fase de componentes de frecuencia especificos
    y retornar cada bloque frecuencial a su dominio temporal para obtener la salida final mediante la ISTFT.

    # Robotizacion:
        Es uno de los efectos mas usados, se aplica un tono constante a la señal mientras se preserva
        los formantes vocales que determinan sonidos vocales y consonantes, lo que da como resultado
        una voz monotona similar a la de un robot.
        Una vez obtenido el bloque discreto de muestras en su dominio frecuencial, es decir la FFT de cada
        bloque de muestras, se setea la fase de cada bin de frecuencia a cero, mientras que la magnitud 
        no sufre cambios, setear la fase a cero es convertir cada bin de frecuencia en un numero real,
        cada bin de frecuencia puede representarse con la siguiente expresion: Ak+jBk. Para lograr la
        robotizacion los bin de frecuencia de salida resultaran ser: np.sqrt((Ak)**2 + (Bk)**2).

    # Whisperizacion (Susurrar):
        Tambien conocido como Whisper o Whisperization, consiste en mantenar los formantes vocales mientras
        se eliminan completamente la sensacion de tonalidad, el resultado final es similar al de una persona
        susurrando, la implementacion es similar a la de robotizacion pero en vez de setear la fase a cero se
        la setea a un valor random entre ɸ=[0, 2pi] si volvemos a tener en cuenta que los bin de frecuencia entrantes
        se pueden representar con la siguiente expresion: Ak+jBk, entonces los bin de frecuencia de salida
        para lograr la whisperizacion resultaran ser: np.sqrt((Ak)**2 + ((Bk)**2))*(cos(ɸk) + jsin(ɸk))

    # Pitch Shifter (Cambiador de tono):
        Si tomamos una señal a la cual le reducimos o aumentamos la cantidad de muestras en una determinada
        proporcion el efecto que simulamos es el de reducir o aumentar la frecuencia de muestreo resultando
        en una alteracion de la duracion temporal de la señal y a su vez resultaran afectadas las tonalidades
        de la señal, pero este tipo de procesamiento no es viable para realizar en tiempo real.
        Entonces debemos recurrir a algun tipo de procesamiento que afecte la tonalidad de la señal sin alterar
        su duracion en el tiempo, en eso consiste este tipo de efecto denominado Pitch Shifter, para lograr esto
        requerimos de 3 procesos:

            1.Phase Correction: Cuando dividimos a la señal en una determinada cantidad de frames de largo N
                pueden darse dos casos, el primer caso es que la fraccion de señal contenida en un frame 
                contenga exactamente un bin de frecuencia es decir la energia de la señal estara contenida totalmente
                en ese frame y el segundo caso es que el frame contenga un bin de frecuencia contenido entre 
                dos frames es decir la energia de la señal estara distribuida entre dos frames.
                Para el primer caso no habra diferencia de fase con el frame continuo pero para el segundo caso 
                la fase entre dos frames continuos sera distinto de cero, esto nos indicaria que la componente 
                frecuencial de la señal es mayor o menor que la frecuencia del frame, esta diferencia de fase
                puede ser usada para determinar la verdadera frecuencia asociada a un bin.

            2.Time Stretching: Cuando hablamos acerca de la STFT omitimos algunos detalles, el proceso de ventana
                altera al espectro de la señal y es por esto que utilizamos ventanas del estilo Hanning, Hamming, etc. 
                pero este proceso por si solo suaviza el inicio y fin de cada frame por lo que la señal producida 
                presentaria ausencia de sonido entre muestras, por esto es que cada frame aparte de ser multiplicada 
                por una ventana, se intenta tomar cada frame con una determinada superposicion (Overlapping), 
                esto ademas ayuda a mejorar la resolucion del espectro, la diferencia entre el inicio de dos frames 
                superpuestos se conoce como Hop size y alterando esta diferencia es posible comprimir o expandir una 
                señal en tiempo. 

            3.Resampling:
                A esta altura ya hemos realizado una correccion de fase que nos permite estirar o comprimir en tiempo 
                una señal sin afectar el tono de la misma, lo que resta realizar es remuestrear la señal para 
                volver a conseguir la duracion inicial, al hacer esto terminaremos afectando el tono de la señal pero 
                conservando su duracion.
    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    fs : Float
        Frecuencia de muestreo del sistema.
    mod_rate : Integer, opcional
        Valor multiplo de 2 que determina el tamaño de la ventana. Por defecto es 1.
    overlap : Integer, opcional
        Superposicion entre segmentos. Por defecto es 1.
    n : Float, opcional
        Tono. Por defecto es 0.
    window : String, opcional
        Tipo de ventana que se utilizara. Por defecto es "Bartlett".
    vocalFx : String, opcional
        Tipo de efecto que se desea implementar. Por defecto es "Robot".

    Returns
    -------
    X_time : Numpy Array
        Array de datos de audio procesados con el efecto vocoder.

    """
    window_size = int(1024*mod_rate)
    if len(data) < window_size:
        data = np.concatenate((data, np.zeros(window_size - len(data))))
    hop = window_size//4
    if window == "Hann":
        w_ = np.hanning(window_size)
    elif window == "Hamming":
        w_ = np.hamming(window_size)
    elif window == "Blackman":
        w_ = np.blackman(window_size)
    elif window == "Bartlett":
        w_ = np.bartlett(window_size)
    elif window == "Kaiser":
        w_ = np.kaiser(window_size, beta=14)
    elif window == "Tukey":
        w_ = sg.windows.tukey(window_size)
    elif window == "Rectangular":
        w_ = sg.windows.boxcar(window_size)

    data_pd = np.pad(data, [(hop, 0)])
    data_stft = sg.stft(x=data_pd, fs=fs, window=w_,
                        nperseg=window_size, noverlap=(window_size - hop)*overlap)

    if vocalFx == "Whisper" or vocalFx == "Robot":
        factor = 2**(1.0 * 0 / 12.0)
        hop_s = hop*factor
        data_voc = np.abs(data_stft[2])

        if vocalFx == "Whisper":
            phase = 2*np.pi * \
                (np.random.rand(data_voc.shape[0], data_voc.shape[1]))
            data_voc = data_voc*(np.cos(phase) + 1j*np.sin(phase))

    elif vocalFx == "Pitch Shifter":
        # PHASE CORRECTION AND TIME SCALING
        factor = 2**(1.0 * n / 12.0)
        hop_s = hop*factor

        bins = data_stft[2]
        bins_ = np.vstack([bins, np.zeros(bins.shape[1])])[
            1:bins.shape[0]+1, :]

        amp = np.abs(data_stft[2])

        w_bin = np.tile(data_stft[0].shape[0], [
                        data_stft[1].shape[0], 1]).transpose() * (2*np.pi)
        w_delta = (np.angle(bins) - np.angle(bins_))/(hop/fs)
        w_wrap = np.mod(w_delta - w_bin + np.pi, 2*np.pi) - np.pi
        w_true = w_bin + w_wrap

        phase_syn = np.angle(bins) + (hop_s/fs) * w_true

        data_voc = np.abs(amp)*np.exp(1j*phase_syn)

    t, data_time = sg.istft(data_voc, fs=fs, window=w_,
                            nperseg=window_size, noverlap=(window_size - hop_s)*overlap)

    if vocalFx == "Pitch Shifter":
        # RESAMPLING
        indices = np.round(np.arange(0, len(data_time), factor))
        indices = indices[indices < len(data_time)].astype(int)
        data_time = data_time[indices.astype(int)]

    return data_time

    ##################### Mi implementacion de STFT #################
    # window_size = int(1024*mod_rate)
    # hop = window_size//2

    # if window == "Hann":
    #     w_ = np.hanning(window_size)
    # elif window == "Hamming":
    #     w_ = np.hamming(window_size)
    # elif window == "Blackman":
    #     w_ = np.blackman(window_size)
    # elif window == "Bartlett":
    #     w_ = np.bartlett(window_size)
    # elif window == "Kaiser":
    #     w_ = np.kaiser(window_size, beta=14)
    # elif window == "Tukey":
    #     w_ = sg.windows.tukey(window_size)
    # elif window == "Rectangular":
    #     w_ = sg.windows.boxcar(window_size)

    # i_hop = len(data)//hop
    # hop_init = np.arange(i_hop+1)*hop
    # hop_end = hop_init[2:]
    # X_time = np.zeros(window_size)

    # for i in range(len(hop_end)-1):
    #     X_fft = np.fft.fft(data[hop_init[i]:hop_end[i]]*w_)
    #     X_fft_next = np.fft.fft(data[hop_init[i+1]:hop_end[i+1]]*w_)

    #     if vocalFx == "Robot":
    #         x_proc = np.absolute(X_fft)
    #         x_proc_next = np.absolute(X_fft_next)

    #     elif vocalFx == "Whisper":
    #         x_proc_abs = np.absolute(X_fft)
    #         x_proc_next_abs = np.absolute(X_fft_next)
    #         phase = 2*np.pi*(np.random.rand())

    #         x_proc = x_proc_abs*(np.cos(phase) + 1j*np.sin(phase))
    #         x_proc = np.concatenate(
    #             (x_proc, x_proc_abs*(np.cos(phase) - 1j*np.sin(phase))))
    #         x_proc_next =  x_proc_next_abs*(np.cos(phase) + 1j*np.sin(phase))
    #         x_proc_next = np.concatenate(
    #             (x_proc_next,  x_proc_next_abs*(np.cos(phase) + 1j*np.sin(phase))))

    #     X_time = np.concatenate((X_time, np.zeros(hop)))
    #     X_time[hop_init[i]:hop_end[i]] += np.fft.ifft(x_proc).real
    #     X_time[hop_init[i+1]:hop_end[i+1]] += np.fft.ifft(x_proc_next).real
    # return X_time


def distortion(data, g_db, distortionType, threshold):
    """
    El efecto DISTORTION es un efecto no lineal que puede ser descrito por una infinidad de curvas
    dependiendo del efecto que se desee lograr. Algunos de los efectos que se implementaran aqui
    se los puede clasificar en 3 diferentes tipos: Overdrive, distortion y fuzz, aunque generalmente estos
    terminos sean utilizados como sinonimos tienen comportamientos ligeramente distintos.

        Overdrive: Para señales con niveles bajos el comportamiento de este efecto es muy cercano a la linealidad,
        pero a medidad que la señal crece a niveles altos el comportamiento comienza a comportarse de forma no lineal.

        Distortion: Son aquellos efectos que opera principalmente en la region no lineal para todas las señales 
        de entrada.

        Fuzz: Al igual que Distortion actua en la region no lineal para todas las señales entrantes pero crea
        cambios mas dramaticos en la forma de onda de la señal.

    Hard y Soft Clipping:Efectos del tipo Overdrive. Hard Clipping genera una transicion abrupta entra la
    region de clipeo y la region de no clipeo, lo que provoca que la forma de onda de la señal de entrada 
    se asimile a la forma de onda de una señal cuadrada. Soft Clipping tiene un comportamiento similar al
    anterior pero la forma de onda de la señal de salida se asemeja a una señal cuadrada con las esquinas
    redondeadas.

    Rectificacion: Consiste en dejar pasar la region positiva de la onda sin cambios y omitir o invertir la
    region negativa de la onda, en este tipo de efectos tenemos a la rectificacion de media onda (Half-Wave Rectification)
    y la rectificacion de onda completa (Full-Wave Rectification)

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    g_db : Float
        Ganancia del preamplificador.
    distortionType : String
        Tipo de distortion que se desea aplicar.
    threshold : Float
        Valor umbral apartir del cual se produciran los efectos.

    Returns
    -------
    out :  Numpy Array
        Array de datos de audio procesados con el efecto distortion.

    """
    g = np.power(10, g_db / 20)
    level_max = np.max(np.abs(data))
    out = (data/level_max)*g

    if distortionType == "HardClipping":
        if (out > threshold).any():
            out[(out > threshold)] = 1
        if (out < -threshold).any():
            out[(out < -threshold)] = -1

    elif distortionType == "SoftClipping":
        threshold1 = threshold * 1/3
        threshold2 = threshold * 2/3

        if (out > threshold2).any():
            out[(out > threshold2)] = 1
        if (out >= threshold1).any() & (out < threshold2).any():
            out[(out >= threshold1) & (out < threshold2)] = 1 - \
                ((2-3*out[(out >= threshold1) & (out < threshold2)])/3)**2
        if (out < threshold1).any():
            out[(out < threshold1)] = 2*out[(out < threshold1)]

        if (out < -threshold2).any():
            out[(out < -threshold2)] = -1
        if (out <= -threshold1).any() & (out > -threshold2).any():
            out[(out <= -threshold1) & (out > -threshold2)] = 1 - \
                ((2-3*out[(out <= -threshold1) & (out > -threshold2)])/3)**2
        if (out > -threshold1).any():
            out[(out > -threshold1)] = 2*out[(out > -threshold1)]

    elif distortionType == "SoftClippingExponential":
        if (out > 0).any():
            out[(out > 0)] = 1 - np.exp(-np.abs(out[(out > 0)]))
        if (out <= 0).any():
            out[(out <= 0)] = -1 + np.exp(-np.abs(out[(out <= 0)]))

    elif distortionType == "FullWaveRectifier":
        out = np.abs(out)

    elif distortionType == "HalfWaveRectifier":
        out[(out <= 0)] = 0

    out = out*level_max
    return out


def hp(fs, fc, att, g, coef=101):
    """
    Filtro pasa altos de tipo FIR

    Parameters
    ----------
    fs : Float
        Frecuencia de muestreo del sistema.
    fc : Float
        Frecuencia de corte.
    att : Float
        Atenuacion de la banda de stop.
    g : Float
        Ganancia de la banda de paso
    coef : Interger, opcional
        Numero de taps del filtro FIR, debe ser un numero impar. Por defecto es 101.

    Returns
    -------
    Numpy Array
        Coeficientes del filtro FIR.

    """
    nyq = fs/2
    frec = np.array([0.0, fc, fc, nyq])
    gain = np.array([-200, -att, g, g])
    gain = 10**(gain/20)
    return sg.firls(numtaps=coef, bands=frec, desired=gain, fs=fs)


def lp(fs, fc, att, g, coef=101):
    """
    Filtro pasa bajo de tipo FIR

    Parameters
    ----------
    fs : Float
        Frecuencia de muestreo del sistema.
    fc : Float
        Frecuencia de corte.
    att : Float
        Atenuacion de la banda de stop.
    g : Float
        Ganancia de la banda de paso.
    coef : Interger, opcional
        Numero de taps del filtro FIR, debe ser un numero impar. Por defecto es 101.

    Returns
    -------
    Numpy Array
        Coeficientes del filtro FIR.

    """
    nyq = fs/2
    frec = np.array([0.0, fc, fc, nyq])
    gain = np.array([g, g, -att, -200])
    gain = 10**(gain/20)
    return sg.firls(numtaps=coef, bands=frec, desired=gain, fs=fs)


def bp(fs, fci, fcs, att, g):
    """
    Filtro pasa banda de tipo FIR

    Parameters
    ----------
    fs : Float
        Frecuencia de muestreo del sistema.
    fci : Float
        Frecuencia de corte inferior.
    fcs : Float
        Frecuencia de corte superior.
    att : Float
        Atenuacion de la banda de stop.
    g : Float
        Ganancia de la banda de paso.

    Returns
    -------
    Numpy Array
        Coeficientes del filtro FIR.

    """
    return np.polymul(lp(fs, fcs, att, g), hp(fs, fci, att, g))


def notch(fs, fc, Q, g):
    """
    Filtro pasa banda de tipo IIR

    Parameters
    ----------
    fs : Float
        Frecuencia de muestreo del sistema.
    fc : Float
        Frecuencia de corte.
    Q : TYPE
        Factor de calidad.
    g : Float
        Ganancia de la banda de paso.

    Returns
    -------
    list
        Numerator y denominador del filtro IIR.

    """
    b, a = sg.iirnotch(fc, Q, fs)
    z, p, k = sg.tf2zpk(b, a)
    b, a = sg.zpk2tf(z, p, k*(1+g))
    return [b, a]


def init_fir(parameters):
    """
    Inicializador de filtros

    Parameters
    ----------
    parameters : Dictionary
        Parametros del sistema.

    Returns
    -------
    buffer_filter : Dictionary
        Diccionario que contiene los coeficientes de los filtros.

    """
    buffer_filter = {"hp": hp(parameters["system"]["fs"], parameters["hp"]["fc"], parameters["hp"]["att"], parameters["hp"]["g"]),
                     "lp": lp(parameters["system"]["fs"], parameters["lp"]["fc"], parameters["lp"]["att"], parameters["lp"]["g"]),
                     "bp": bp(parameters["system"]["fs"], parameters["bp"]["fci"], parameters["bp"]["fcs"], parameters["bp"]["att"], parameters["bp"]["g"]),
                     "notch": notch(parameters["system"]["fs"], parameters["notch"]["fc"], parameters["notch"]["Q"], parameters["notch"]["g"])
                     }
    return buffer_filter


def apply_filter(data, buffer_filter, tipo=""):
    """
    Aplicador de filtros.

    Parameters
    ----------
    data : Float
        Frecuencia de muestreo del sistema.
    buffer_filter : Dictionary
        Diccionario que contiene los coeficientes de los filtros.
    tipo : String, opcional
        Filtro a aplicar. Por defecto es "".

    Returns
    -------
    data_out :  Numpy Array
        Array de datos de audio procesados por los filtros.

    """
    if tipo == "notch":
        data_out = sg.filtfilt(
            buffer_filter["notch"][0], buffer_filter["notch"][1], data)
    else:
        if tipo == "hp":
            bf_filter = buffer_filter["hp"]
        elif tipo == "lp":
            bf_filter = buffer_filter["lp"]
        elif tipo == "bp":
            bf_filter = buffer_filter["bp"]
        data_out = sg.filtfilt(bf_filter, 1, data, axis=0)
    return data_out

# def delay_feedback_ba(data, fs, delay_time, gff, gfb):
#     N = int(delay_time * fs)
#     b = np.zeros(N+1)
#     b[0] = 1
#     b[-1] = gff - gfb
#     a = np.zeros(N+1)
#     a[0] = 1
#     a[-1] = -gfb
#     y = sg.lfilter(b, a, data, axis=0)
#     return y


def reverb(data, fs, dry=1, wet=0.1, room_size=40, colorless=35, g_colorless=0.7, diffusion=8):
    """
    Implementacion del efecto REVERBeration, en un ambiente acustico las ondas de sonido,
    que atraviesan el espacio para llegar al oyente, producen 'copias' que rebotan en determinadas 
    superficies alargando su recorrido por el espacio provocando un retardo temporal y una atenuacion en su 
    llegada al oyente, esto provoca la percepcion espacial en el sonido y da lugar a la reverberacion.
    Como dato de color, sin la reverberacion muchos de los instrumentos que escuchamos no sonarian igual,
    ya que algunos instrumentos no irradian todas las frecuencias de igual manera en todas las direcciones,
    la reverberacion ayuda a difundir la energia de una onda de forma mas uniforme para un oyente.

    Reverberador de Schroeder:
        Schroeder nos brinda un marco para emplear un reverb con caracteristicas basicas el cual
        consiste de 3 principales componentes: Filtros comb, filtros pasa todo y una matriz de mezcla, este
        ultimo elemento con el tiempo a sido reemplazado por metodos mas sofisticados. Los filtros comb son
        un caso especial de un filtro digital IIR por que hay retroalimentacion de la salida retardada a la
        entrada, gracias a esto se logra representar el sonido reflectandose entre dos paredes paralelas que 
        produce una serie de ecos. Los ecos decaen exponencialmente y estan espaciados uniformemente en el
        tiempo.
        Los filtros pasa todo proveen ecos de alta densidad 'incoloros'. Esencialmente, estos filtros
        transforman cada muestra de entrada de la etapa anterior en una respuesta de impulso infinita completa,
        lo que da como resultado una mayor densidad de eco. Este bloque de filtro se conocen como difusores
        de impulso.

    Reverb convolucional:
        De la convolucion entre la señal de entrada y una respuesta al impulso finita, como pueden ser los
        filtros comb que nosotros emplearemos, podemos genera la señal reverberada, 
        pero en terminos de trabajo computacional la convolucion resulta muy costosa de realizar, 
        para solucionar esto recurrimos al teorema de convolucion que nos dice que la multiplicacion en el 
        dominio de Fourier es equivalente a convolucionar en el dominio temporal y viceversa,
        esta multiplicacion requiere menos trabajo computacional por lo que resulta un proceso mas veloz.

    Para simular un filtro comb nosotros utilizares un tren de deltas separados uniformemente en el tiempo donde
    cada delta sera atenuada por un valor aleatorio lo que da la sensacion natural de un reverb. Se añadio un
    filtro pasa bajos que ayuda a simular la absorcion de ese sonido metalico, caracterisco de la reverberancia en altas frecuencias,
    que naturalmente es absorbido por el aire y las paredes.

    Parameters
    ----------
    data : Numpy Array
        Array de datos de audio.
    fs : Float
        Frecuencia de muestreo del sistema.
    dry : Float, opcional
        Valor que controla cuanto se involucran los datos originales en el producto de salida. Por defecto es 1.
    wet : Float, opcional
        Valor que controla cuanto se involucra los datos procesados en el producto de salida. Por defecto es 0.1.
    room_size : Integer, opcional
        Cantidad de retardo que logra la ilusion de estar en un cuarto. Por defecto es 40.
    colorless : Integer, opcional
        Retardo que generan los filtros pasa bajos - nivel de decoloracion. Por defecto es 35.
    g_colorles : Float, opcional
        Fuerza de decoloracion. Por defecto es 0.7.
    diffusion : Integer, opcional
        Cantidad de difusores. Por defecto es 8.

    Returns
    -------
    data_rev :  Numpy Array
        Array de datos de audio procesados por el efecto reverb.

    """
    datas = np.concatenate((data, np.zeros(ms2smp(120, fs)*10)))

    S = np.fft.fft(datas)
    R = np.array(np.zeros_like(S))

    delay = np.arange(30, room_size, 10)

    for d in delay:
        h = sg.unit_impulse(datas.shape[0])
        for i in range(10):
            h[ms2smp(d, fs)*(1+i)] = np.random.uniform(0.1, (11-i)/10)
        h[0] = 0
        H = np.fft.fft(h)
        R += (H*S)*wet

    data_rev = np.fft.ifft(R).real

    fl_lp = lp(fs, 10000, 30, 0)
    data_rev = sg.filtfilt(fl_lp, 1, data_rev, axis=0)
    data_rev = datas*dry + data_rev

    data_rev = data_rev[:(datas.shape[0]//2) + ms2smp(d, fs)*(1+i)]

    d = colorless
    g = g_colorless

    b = np.zeros(d)
    b[-1] = 1
    b[0] = -g

    a = np.zeros(d)
    a[-1] = -g
    a[0] = 1
    for i in range(diffusion):
        data_rev = sg.lfilter(b, a, data_rev)

    return data_rev

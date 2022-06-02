# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:34:46 2022

@author: https://github.com/aletbm
"""

"""
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if(p.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')) > 0:
        print("Input Device id ", i, " - ",p.get_device_info_by_host_api_device_index(0, i).get('name'))
"""
# sys.path.insert(0, 'C:\\Users\\alexa\\Desktop\\Codigos\\PDS\\Nueva carpeta')




import numpy as np
from pitch_transcription import nota_musical
import matplotlib.pyplot as plt
def aspect_default(ax):
    """
    Setea el aspecto visual del Axes.

    Parameters
    ----------
    ax : clase Axes que contiene elementos de una figura

    Returns
    -------
    None.

    """
    ax.grid(alpha=0.4, color='#FFFFFF')
    ax.set_facecolor("black")
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['top'].set_color('#dddddd')
    ax.spines['right'].set_color('#dddddd')
    ax.spines['left'].set_color('#dddddd')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    return


def aspect_piano(ax):
    frec_ref = 440
    pitch_init = 0
    colors = {"hot": ["#FF1604", "#FF2C09", "#FF420E", "#FF5714", "#FF6B19", "#FF7F20", "#FF9326", "#FFA52E", "#FFB736", "#FFC840", "#FFD94B", "#FFE75A"],
              "piano": ["#A0CBE8", "#4E79A7", "#A0CBE8", "#4E79A7", "#A0CBE8", "#A0CBE8", "#4E79A7", "#A0CBE8", "#4E79A7", "#A0CBE8", "#4E79A7", "#A0CBE8"]}
    i_color = 0
    for pitch in range(pitch_init, 12*15):
        p_central = nota_musical(pitch, frec_ref)
        p_prev = nota_musical(pitch - 1, frec_ref)
        p_next = nota_musical(pitch + 1, frec_ref)
        f_i = p_central - (p_central - p_prev)/2
        f_f = p_central + (p_next - p_central)/2
        rect = plt.Rectangle((f_i, 0), f_f - f_i, 5500,
                             color=colors["piano"][i_color], ec='black', lw=0.1)
        i_color += 1
        i_color %= 12
        ax.add_patch(rect)
    return


def init_transcription():
    """
    Se instancian plots para generar espacio en la aplicacion de Kivy

    Returns
    -------
    fig : clase matplotlib.figure
    ax : clase matplotlib.axes.Axes
        Axes para presentar un Espectrograma
    ax2 : clase matplotlib.axes.Axes
        Axes para presentar la diferencia espectral
    ax3 : clase matplotlib.axes.Axes
        Axes para presentar los datos contenidos en un archivo .wav

    """
    fig, (ax, ax2, ax3) = plt.subplots(3)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.11,
        right=0.981,
        hspace=0.55,
        wspace=0.2
    )

    ax.plot(0, 0)
    aspect_default(ax)

    ax2.plot(0, 0)
    aspect_default(ax2)

    ax3.plot(0, 0)
    aspect_default(ax3)

    fig.patch.set_facecolor('xkcd:black')

    return fig, ax, ax2, ax3


def init_visualizer(fs, FRAMES):
    """
    Se instancian plots para generar espacio en la aplicacion de Kivy

    Parameters
    ----------
    fs : TYPE
        DESCRIPTION.
    FRAMES : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : Float
        Frecuencia de muestreo del sistema.
    ax : clase matplotlib.axes.Axes
        Axes para presentar los datos de audio en tiempo real
    ax1 : clase matplotlib.axes.Axes
        Axes para presentar la transformada de Fourier de los datos de audio en tiempo real
    line : Line2D
        Linea que representa los datos ploteados.
    line_fft : Line2D
        Linea que representa los datos ploteados.
    line_fft_max : Line2D
        Linea que representa los datos ploteados.

    """
    fig, (ax, ax1) = plt.subplots(2)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.11,
        right=0.981,
        hspace=0.35,
        wspace=0.2
    )

    x_audio = np.arange(0, FRAMES, 1)
    x_fft = np.linspace(0, fs, FRAMES)

    line, = ax.plot(x_audio, np.random.rand(FRAMES), '#00FF00')
    line_fft, = ax1.semilogx(x_fft, np.random.rand(
        FRAMES), '#000000', linewidth=0.8)
    line_fft_max, = ax1.plot(0, 0, '#FF0000', marker="x", ls='None')
    ax.set_ylim(-40000, 40000)
    ax.set_xlim = (0, FRAMES)
    ax.set_title("Real-Time Audio", color="white")
    ax.set_xlabel(f"Frames - {FRAMES}", color="white")
    ax.set_ylabel("Amplitude", color="White")

    Fmin = 15
    Fmax = fs//2
    ax1.set_xlim(Fmin, Fmax)
    ax1.set_ylim(0, 5500)
    ax1.set_title("Real-Time FFT", color="white")
    ax1.set_xlabel("Frecuency [Hz]", color="white")
    ax1.set_ylabel("Normalized |FFT|", color="white")
    ax1.legend(loc="upper right")

    aspect_default(ax)

    aspect_piano(ax1)

    ax1.set_facecolor("black")
    ax1.spines['bottom'].set_color('#dddddd')
    ax1.spines['top'].set_color('#dddddd')
    ax1.spines['right'].set_color('#dddddd')
    ax1.spines['left'].set_color('#dddddd')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    fig.patch.set_facecolor('xkcd:black')

    # Asignamos los datos a la curva de la variación temporal
    line.set_ydata(np.zeros(FRAMES))
    line_fft.set_ydata(np.zeros(FRAMES))
    return fig, ax, ax1, line, line_fft, line_fft_max


def visualizer(q_graph, q_param, q_control, parameters, filters):
    """
    Funcion utilizada por el proceso secundario, 
    creado por el programa principal para instanciar una aplicacion Kivy y llevar
    a cabo tareas de visualizacion y obtencion de datos de forma grafica.

    Parameters
    ----------
    q_graph : clase multiprocessing.Queue
        Cola compartida por el proceso principal para intercambiar datos de visualizacion.
    q_param : clase multiprocessing.Queue
        Cola compartida por el proceso principal para intercambiar el intercambio de parametros.
    q_control : q_param : clase multiprocessing.Queue
        Cola compartida por el proceso principal para intercambiar señales de control.
    parameters : Diccionario
        Utilizado como buffer para almacenar todos los parametros utilizados en el sistema.
    filters : Diccionario
        Utilizado como buffer para almacenar los coeficientes de los filtros, ayuda en la optimizacion
        de la reproduccion de audio, ya que ahorra tiempo de computo en el calculo de dichos coeficientes

    Returns
    -------
    None.

    """
    from app import MainApp, init_graphics_filters, Fig2KivyCanvas
    fig, ax, ax1, line, line_fft, line_fft_max = init_visualizer(
        parameters["system"]["fs"], parameters["system"]["frames"])
    cv_visualizer = Fig2KivyCanvas(fig)
    fig2, ax, ax2, ax3 = init_transcription()
    cv_transcription = Fig2KivyCanvas(fig2)
    canvasAll = init_graphics_filters(parameters, filters)
    canvasAll["visualizer"] = {"canvas": cv_visualizer, "line": line,
                               "line_fft": line_fft, "line_fft_max": line_fft_max, "time": ax, "frec": ax1}
    canvasAll["transcription"] = {"canvas": cv_transcription, "spectrogram": ax,
                                  "spectralDif": ax2, "song": ax3, "minX": 0, "maxX": 0, "minY": 0, "maxY": 0}

    app = MainApp()
    app.setParameters(parameters, canvasAll, q_graph, q_param, q_control)
    app.run()


def processing_data(data_int16, parameters, FILTER):
    """
    Funcion que se encargar de llevar a cabo el procesamiento de datos para la aplicacion de efectos.

    Parameters
    ----------
    data_int16 : Numpy Array de int16
        Contiene los datos de audio casteados a int16
    parameters : Diccionario
        Utilizado como buffer para almacenar todos los parametros utilizados en el sistema.
    FILTER : Diccionario
        Utilizado como buffer para almacenar los coeficientes de los filtros, ayuda en la optimizacion
        de la reproduccion de audio, ya que ahorra tiempo de computo en el calculo de dichos coeficientes

    Returns
    -------
    None.

    """
    data_int32 = data_int16.astype(np.float32)

    nframes = len(data_int32)/1024
    if nframes > np.floor(nframes):
        len_add_zeros = np.ceil(nframes)*1024 - len(data_int32)
        data_int32 = np.concatenate((data_int32, np.zeros(int(len_add_zeros))))

    if parameters["system"]["fx"] == "delay":
        data_out = delay_feedback(data_int32,
                                  parameters["system"]["fs"],
                                  parameters["delay"]["dry"],
                                  parameters["delay"]["wet"],
                                  parameters["delay"]["delay_sec"],
                                  parameters["delay"]["gfb"],
                                  parameters["delay"]["bounces"])

    elif parameters["system"]["fx"] == "flanger":
        data_out = flanger_feedback(data_int32,
                                    parameters["system"]["fs"],
                                    parameters["flanger"]["dry"],
                                    parameters["flanger"]["wet"],
                                    parameters["flanger"]["delay_sec"],
                                    parameters["flanger"]["depth"],
                                    parameters["flanger"]["fLFO"],
                                    parameters["flanger"]["bounces"],
                                    parameters["flanger"]["gfb"])

    elif parameters["system"]["fx"] == "chorus":
        data_out = chorus(data_int32,
                          parameters["system"]["fs"],
                          parameters["chorus"]["dry"],
                          parameters["chorus"]["wet"],
                          parameters["chorus"]["delay_sec"],
                          parameters["chorus"]["depth"],
                          parameters["chorus"]["fLFO"])

    elif parameters["system"]["fx"] == "reverb":
        data_out = reverb(data_int32,
                          parameters["system"]["fs"],
                          parameters["reverb"]["dry"],
                          parameters["reverb"]["wet"],
                          parameters["reverb"]["room_size"],
                          parameters["reverb"]["colorless"],
                          parameters["reverb"]["g_colorless"],
                          parameters["reverb"]["diffusion"])

    elif parameters["system"]["fx"] == "alien":
        data_out = alienvox(data_int32,
                            parameters["system"]["fs"],
                            parameters["alien"]["freq"],
                            parameters["alien"]["n_harmonics"])

    elif parameters["system"]["fx"] == "distortion":
        data_out = distortion(data_int32,
                              parameters["distortion"]["g_db"],
                              parameters["distortion"]["distortionType"],
                              parameters["distortion"]["threshold"])

    elif parameters["system"]["fx"] == "tremolo":
        data_out = tremolo(data_int32,
                           parameters["system"]["fs"],
                           parameters["tremolo"]["fLFO"],
                           parameters["tremolo"]["alpha"])

    elif parameters["system"]["fx"] == "vocoder":
        data_out = vocoder(data_int32,
                           parameters["system"]["fs"],
                           parameters["vocoder"]["mod_rate"],
                           parameters["vocoder"]["overlap"],
                           parameters["vocoder"]["n"],
                           parameters["vocoder"]["window"],
                           parameters["vocoder"]["vocalFx"])

    elif parameters["system"]["fx"] == "hp" or parameters["system"]["fx"] == "lp" or parameters["system"]["fx"] == "bp" or parameters["system"]["fx"] == "notch":
        data_out = apply_filter(data_int32, FILTER, parameters["system"]["fx"])

    else:
        data_out = data_int32

    return data_out


def buffer_audio(data_out, parameters):
    global FILTER_BACK, COUNTER, BUFFER_OUT
    if COUNTER == 0 or FILTER_BACK != parameters["system"]["fx"]:
        FILTER_BACK = parameters["system"]["fx"]
        BUFFER_OUT = 0
        BUFFER_OUT = data_out
        COUNTER += parameters["system"]["frames"]
    else:
        BUFFER_OUT = np.concatenate(
            (BUFFER_OUT, np.zeros(parameters["system"]["frames"])))
        if len(BUFFER_OUT) > len(data_out):
            data_out = np.concatenate(
                (data_out, np.zeros(len(BUFFER_OUT) - len(data_out))))
        if len(BUFFER_OUT) < len(data_out):
            BUFFER_OUT = np.concatenate(
                (BUFFER_OUT, np.zeros(len(data_out)-len(BUFFER_OUT))))
        BUFFER_OUT += data_out
    data_out = BUFFER_OUT[:parameters["system"]["frames"]]
    BUFFER_OUT = BUFFER_OUT[parameters["system"]["frames"]:]
    return data_out


def callback(in_data, fram_count, time_info, status):
    """
    Handler de pyaudio que se encargar del flujo de datos de los datos de audio.

    """
    global parameters, FILTER, q_graph, q_param
    if parameters["system"]["live"] == False:
        data = wf.readframes(parameters["system"]["frames"])
        if wf.getnchannels() == 2:
            data_int16 = np.frombuffer(data, dtype=np.int16)[::2]
        else:
            data_int16 = np.frombuffer(data, dtype=np.int16)
    else:
        # Leemos paquetes de longitud FRAMES
        data_int16 = np.frombuffer(in_data, dtype=np.int16)

    parameters2 = parameters.copy()

    try:
        parameters2 = q_param.get_nowait()
        FILTER = init_fir(parameters2)
    except:
        parameters2 = parameters
    finally:
        parameters = parameters2

        data_out = processing_data(data_int16, parameters, FILTER)

        if parameters["system"]["fx"] == "delay" or parameters["system"]["fx"] == "chorus" or parameters["system"]["fx"] == "flanger" or parameters["system"]["fx"] == "vocoder" or parameters["system"]["fx"] == "reverb":
            data_out = buffer_audio(data_out, parameters)

        data_out = limiter(data_out, parameters["system"]["gain"])

        data_int16 = data_out.astype(np.int16)
        q_graph.put(data_int16)
        data_bytes = data_int16.tobytes()

        return data_bytes, pa.paContinue


if __name__ == '__main__':

    import pyaudio as pa
    import scipy.signal as sg
    import scipy.io.wavfile as wavfile
    import wave
    from midiutil import MIDIFile
    from multiprocessing import Process, Queue, freeze_support
    import pathlib
    from pitch_transcription import spectral_difference, w_phase_deviation, onset_detection, time_to_samples, create_mel, save_MIDI
    from effects import limiter, chorus, reverb, flanger_feedback, delay_feedback, alienvox, distortion, tremolo, vocoder, init_fir, apply_filter

    parameters = {"system": {"device_input": 1,
                             "fs": 44100,
                             "format": pa.paInt16,
                             "channels": 1,
                             "frames": 1024*8,
                             "fx": "piano",
                             "live": False,
                             "path": str(pathlib.Path().absolute()),
                             "preview": True,
                             "gain": 0,
                             "PLAY": "PLAY",
                             "STOP": "STOP",
                             "END": "END",
                             "PITCH": "TRANSCRIPTION",
                             "POISON_PILL": "KILLAPP"},
                  "delay": {"dry": 1,
                            "wet": 0.5,
                            "delay_sec": 0.5,
                            "gfb": 0.5,
                            "bounces": 20},
                  "flanger": {"dry": 1,
                              "wet": 1,
                              "delay_sec": 0.002,
                              "depth": 0.0001,
                              "fLFO": 10,
                              "bounces": 3,
                              "gfb": 0.5},
                  "chorus": {"dry": 1,
                             "wet": [.5, .5],
                             "delay_sec": 0.002,
                             "depth": 0.0001,
                             "fLFO": [20, 10]},
                  "reverb": {"dry": 1,
                             "wet": 0.5,
                             "room_size": 40,
                             "colorless": 35,
                             "g_colorless": 0.7,
                             "diffusion": 5},
                  "alien": {"freq": 1,
                            "n_harmonics": 1},
                  "distortion": {"g_db": 1,
                                 "distortionType": "HardClipping",
                                 "threshold": 1},
                  "tremolo": {"fLFO": 10,
                              "alpha": 0.9},
                  "vocoder": {"mod_rate": 1/2,
                              "overlap": 1,
                              "n": 0,
                              "window": "Hann",
                              "vocalFx": "Robot"},
                  "hp": {"fc": 1000,
                         "att": 30,
                         "g": 0},
                  "lp": {"fc": 1000,
                         "att": 30,
                         "g": 0},
                  "bp": {"fci": 500,
                         "fcs": 3500,
                         "att": 30,
                         "g": 0},
                  "notch": {"fc": 1000,
                            "Q": 2,
                            "g": 0},
                  "transcription": {"nchannels": 1,
                                    "sampwidth": 2,
                                    "window_len": 3,
                                    "waveform": "sine",
                                    "type_onset": "percussive",
                                    "shift": 1}
                  }

    COUNTER = 0
    BUFFER_OUT = []
    FILTER = init_fir(parameters)
    FILTER_BACK = parameters["system"]["fx"]

    p = pa.PyAudio()

    q_graph = Queue()
    q_param = Queue()
    q_control = Queue()
    freeze_support()
    visualProcss = Process(None, visualizer, args=(
        q_graph, q_param, q_control, parameters, FILTER,))
    visualProcss.daemon = True
    visualProcss.start()

    while True:

        parameters2 = parameters.copy()

        try:
            parameters2 = q_param.get_nowait()
            FILTER = init_fir(parameters2)
        except:
            parameters2 = parameters
        finally:
            parameters = parameters2

        try:
            ctrl = q_control.get_nowait()
        except:
            ctrl = ""

        if ctrl == parameters["system"]["PITCH"]:
            wf_synth = wave.open(parameters["system"]["path"], 'rb')
            data_synth = wf_synth.readframes(wf_synth.getnframes())

            if wf_synth.getnchannels() == 2:
                data_synth = np.frombuffer(data_synth, dtype=np.int16)[::2]
            else:
                data_synth = np.frombuffer(data_synth, dtype=np.int16)

            fs_synth = wf_synth.getframerate()

            window_len = 1024*parameters["transcription"]["window_len"]
            window_size = window_len
            hop = window_size//2
            w_ = sg.cosine(window_size)

            if parameters["transcription"]["type_onset"] == 'percussive':
                sf, data_stft = spectral_difference(
                    data_synth, hop_length=hop, window=w_, fs=fs_synth)
            else:
                sf, data_stft = w_phase_deviation(
                    data_synth, hop_length=hop, window=w_, fs=fs_synth)

            peak_frames, sf = onset_detection(sf, fs_synth)
            peak_samples = time_to_samples(peak_frames, hop)
            peak_times = peak_samples/fs_synth
            sf = sf/np.max(sf)

            lapse = np.sum(np.diff(peak_times))/peak_times.shape[0]
            # lapse __________ 1 beat
            # 60sec __________ x beat
            BPM = 60/lapse
            BPM = round(BPM, 3)

            song, list_tones, list_pitch = create_mel(
                data_synth, peak_samples, fs_synth, waveform=parameters["transcription"]["waveform"], shiftOct=parameters["transcription"]["shift"], fadeOut=0.85)
            # song_bytes = song.astype("<h").tobytes()
            wavfile.write(parameters["system"]["path"][:-4] +
                          "_synth.wav", fs_synth, song.astype(np.int16))

            MIDI_cfg = {"track": 0, "time": 0,
                        "channel": 0, "tempo": BPM, "volume": 100}
            myMIDI = MIDIFile(1)
            myMIDI.addTempo(MIDI_cfg["track"],
                            MIDI_cfg["time"], MIDI_cfg["tempo"])
            save_MIDI(list_pitch, myMIDI, MIDI_cfg, parameters)

            p_transcription = {"data": data_synth, "data_stft": data_stft, "spectral_flux": sf,
                               "window": w_, "peaks_f": peak_frames, "peaks_s": peak_samples, "tones": list_tones}
            q_graph.put(p_transcription)

        if ctrl == parameters["system"]["PLAY"]:
            if parameters["system"]["live"]:
                stream = p.open(                                  # Abrimos el canal de audio con los parámeteros de configuración
                    format=parameters["system"]["format"],
                    channels=parameters["system"]["channels"],
                    rate=parameters["system"]["fs"],
                    input_device_index=parameters["system"]["device_input"],
                    input=True,
                    output=True,
                    frames_per_buffer=parameters["system"]["frames"],
                    stream_callback=callback
                )

                stream.start_stream()
                # print("Escuchando...")
                while stream.is_active():
                    try:
                        ctrl = q_control.get_nowait()
                        if ctrl == parameters["system"]["STOP"] or ctrl == parameters["system"]["POISON_PILL"]:
                            break
                    except:
                        pass
                stream.stop_stream()

            else:
                wf = wave.open(parameters["system"]["path"], 'rb')
                stream = p.open(                                  # Abrimos el canal de audio con los parámeteros de configuración
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=1,
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=parameters["system"]["frames"],
                    stream_callback=callback
                )

                stream.start_stream()
                # print("Escuchando...")
                while stream.is_active():
                    try:
                        ctrl = q_control.get_nowait()
                        if ctrl == parameters["system"]["STOP"] or ctrl == parameters["system"]["POISON_PILL"]:
                            break
                    except:
                        pass

                stream.stop_stream()

            q_graph.put(parameters["system"]["END"])
            stream.close()

            if ctrl == parameters["system"]["POISON_PILL"]:
                # print("Cerrando...")
                p.terminate()
                visualProcss.join()
                break

        elif ctrl == parameters["system"]["POISON_PILL"]:
            # print("Cerrando...")
            p.terminate()
            visualProcss.join()
            break
    # print("Listo")

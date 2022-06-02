# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:35:39 2022

@author: https://github.com/aletbm
"""

from pitch_transcription import detect_key
import pandas as pd
import matplotlib
import copy
from effects import lp, hp, bp, notch
import re
import scipy.signal as sg
import numpy as np
import matplotlib.pyplot as plt
from kivy.core.window import Window
from kivy.config import Config
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, StringProperty
from kivy.garden.knob import Knob
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
import kivy

kivy.require('1.9.0')
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
Config.set('kivy', 'exit_on_escape', '0')
Config.set('graphics', 'resizable', 0)
Window.size = (800, 750)


def Fig2KivyCanvas(fig):
    return FigureCanvasKivyAgg(fig)


def cfg_plot(fs=None, freq=None, h=None, fc=None, label=None, ylim=None, xlim=None, tipo="Freq", x=None, y=None):
    fig, ax = plt.subplots()
    line_fc = []
    if tipo == "Freq":
        nyq = fs/2
        line, = ax.semilogx(freq*fs/(2*np.pi), 20 * np.log10(abs(h)),
                            'r', label=label, linewidth='0.8', color="#04F2F2")

        for i in fc:
            line_fc.append(ax.axvline(x=i, color="red"))
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(1, nyq)

    if tipo == "Time":
        line, = ax.plot(x, y, linewidth='0.8', color="#04F2F2")
        ax.set_ylim(-np.max(np.abs(y[(y <= 0)])), np.max(np.abs(y[(y >= 0)])))

    ax.grid(which="both", ls="-")
    ax.grid(alpha=0.4, color='#FFFFFF')
    ax.set_facecolor("black")
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['top'].set_color('#dddddd')
    ax.spines['right'].set_color('#dddddd')
    ax.spines['left'].set_color('#dddddd')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    fig.patch.set_facecolor('#303030')
    return fig, ax, line, line_fc


def init_graphics_filters(parameters, filters):
    canvasAll = {"lp": {},
                 "hp": {},
                 "bp": {},
                 "notch": {},
                 "alien": {}}
    # LP
    freq, h = sg.freqz(filters["lp"], fs=2*np.pi)
    fig, ax, line, line_fc = cfg_plot(fs=parameters["system"]["fs"], freq=freq, h=h, fc=[
                                      parameters["lp"]["fc"]], label="Lowpass Filter", ylim=[-60, 20], xlim=10)
    canvas = Fig2KivyCanvas(fig)
    canvasAll["lp"] = {"canvas": canvas, "line": line, "lineFC": line_fc[0]}

    # HP
    freq, h = sg.freqz(filters["hp"], fs=2*np.pi)
    fig, ax, line, line_fc = cfg_plot(fs=parameters["system"]["fs"], freq=freq, h=h, fc=[
                                      parameters["hp"]["fc"]], label="Highpass Filter", ylim=[-60, 20], xlim=10)
    canvas = Fig2KivyCanvas(fig)
    canvasAll["hp"] = {"canvas": canvas, "line": line, "lineFC": line_fc[0]}

    # BP
    freq, h = sg.freqz(filters["bp"], fs=2*np.pi)
    fig, ax, line, line_fc = cfg_plot(fs=parameters["system"]["fs"], freq=freq, h=h, fc=[
                                      parameters["bp"]["fci"], parameters["bp"]["fcs"]], label="Bandpass Filter", ylim=[-60, 20], xlim=10)
    canvas = Fig2KivyCanvas(fig)
    canvasAll["bp"] = {"canvas": canvas, "line": line,
                       "lineFCI": line_fc[0], "lineFCS": line_fc[1]}

    # NOTCH
    freq, h = sg.freqz(filters["notch"][0], filters["notch"][1], fs=2*np.pi)
    fig, ax, line, line_fc = cfg_plot(fs=parameters["system"]["fs"], freq=freq, h=h, fc=[
                                      parameters["notch"]["fc"]], label="Notch Filter", ylim=[-60, 20], xlim=10)
    canvas = Fig2KivyCanvas(fig)
    canvasAll["notch"] = {"canvas": canvas, "line": line, "lineFC": line_fc[0]}

    # ALIEN
    x = np.linspace(0, 2*np.pi, 1024)
    y = np.sin(x*parameters["alien"]["freq"])
    if parameters["alien"]["n_harmonics"] > 1:
        for i in range(2, parameters["alien"]["n_harmonics"]+1):
            y *= np.sin(x*parameters["alien"]["freq"]*i)
    fig, ax, line, line_fc = cfg_plot(tipo="Time", x=x, y=y)
    canvas = Fig2KivyCanvas(fig)
    canvasAll["alien"] = {"canvas": canvas, "ax": ax, "line": line}
    return canvasAll.copy()


class FileChoosePopup(Popup):
    load = ObjectProperty()


class myTabPiano(BoxLayout):

    def __init__(self, transcription, q_ctrl, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters
        self.q_ctrl = q_ctrl
        self.transcription = transcription

        self.nota_counters = {"C": 0, "C#": 0, "D": 0, "D#": 0, "E": 0,
                              "F": 0, "F#": 0, "G": 0, "G#": 0, "A": 0, "A#": 0, "B": 0}

        nota_list = list(self.nota_counters.keys())
        self.scales = {}
        self.scales["Major"] = {}
        for i in range(len(nota_list)):
            self.scales["Major"][nota_list[0]] = [nota_list[0], nota_list[2],
                                                  nota_list[4], nota_list[5], nota_list[7], nota_list[9], nota_list[11]]
            nota_list.append(nota_list[0])
            nota_list = nota_list[1:]

        self.scales["Minor"] = {}
        for i in range(len(nota_list)):
            self.scales["Minor"][nota_list[0]] = [nota_list[0], nota_list[2],
                                                  nota_list[3], nota_list[5], nota_list[7], nota_list[8], nota_list[10]]
            nota_list.append(nota_list[0])
            nota_list = nota_list[1:]

        self.scales["Chromatic"] = list(self.nota_counters.keys())

        self.currentScale = self.scales["Chromatic"]

        self.spinnerScale = self.ids.spinnerScale
        self.spinnerTonic = self.ids.spinnerTonic

        self.octaveText = self.ids['octaveText']
        self.scalePredict = self.ids['scalePredict']

        self.keys = {"C": self.ids['C'],
                     "C#": self.ids['C_sharp'],
                     "D": self.ids['D'],
                     "D#": self.ids['D_sharp'],
                     "E": self.ids['E'],
                     "F": self.ids['F'],
                     "F#": self.ids['F_sharp'],
                     "G": self.ids['G'],
                     "G#": self.ids['G_sharp'],
                     "A": self.ids['A'],
                     "A#": self.ids['A_sharp'],
                     "B": self.ids['B']}

        self.spinnerWindow = self.ids.spinnerWindow
        self.spinnerWaveform = self.ids.spinnerWaveform
        self.spinnerShift = self.ids.spinnerShift
        self.spinnerOnset = self.ids.spinnerOnset

        self.TranscribeBtn = self.ids.TranscribeBtn
        self.UpBtn = self.ids.UpBtn
        self.LBtn = self.ids.LBtn
        self.RBtn = self.ids.RBtn
        self.DownBtn = self.ids.DownBtn
        self.PlusBtn = self.ids.PlusBtn
        self.MinusBtn = self.ids.MinusBtn

        self.zoom = 1.5

        self.posx_min = 0
        self.posx_max = 0
        self.posy_min = 0
        self.posy_max = 0

    def on_spinner_scale(self):
        if self.spinnerScale.text == "Chromatic":
            self.currentScale = self.scales["Chromatic"]
        else:
            self.currentScale = self.scales[self.spinnerScale.text][self.spinnerTonic.text]
        for key in self.scales["Chromatic"]:
            if key in self.currentScale:
                self.keys[key].background_disabled_normal = ''
            else:
                self.keys[key].background_disabled_normal = 'atlas://data/images/defaulttheme/button_disabled'
        return

    def reset_keys(self):
        for key in self.keys.keys():
            if '#' in key:
                self.keys[key].background_color = [78/255, 121/255, 167/255, 1]
            else:
                self.keys[key].background_color = [
                    160/255, 203/255, 232/255, 1]
        return

    def key_change_color(self, key):
        self.keys[key].background_color = [1, 0, 0, 1]
        self.nota_counters[key] += 1
        return

    def octave(self, octava):
        self.octaveText.text = str(octava)
        return

    def reset_counters(self):
        for key in self.nota_counters.keys():
            self.nota_counters[key] = 0
        return

    def predicted_scale(self):
        data = {'key': list(self.nota_counters.keys()),
                'counter': list(self.nota_counters.values())}
        new = pd.DataFrame.from_dict(data)

        notas_detect = new.sort_values('counter', ascending=False)[
            0:7].sort_index()

        notas_detect = notas_detect[notas_detect.counter > 0].key.tolist()

        count = pd.DataFrame.from_dict(
            {"scale": self.nota_counters.keys(), "coinMajor": [0]*12, "coinMinor": [0]*12})

        nota_list = list(self.nota_counters.keys())
        for i in range(len(nota_list)):
            for key in notas_detect:
                if key in self.scales["Major"][nota_list[i]]:
                    count.loc[count.scale == nota_list[i], "coinMajor"] += 1
                if key in self.scales["Minor"][nota_list[i]]:
                    count.loc[count.scale == nota_list[i], "coinMinor"] += 1
                #print("Escala encontrada "+ nota_list[i])
        if count.coinMajor.max() > count.coinMinor.max():
            self.scalePredict.text = str(
                count[count.coinMajor == count.coinMajor.max()].scale.tolist()[0])+"Maj"
        else:
            self.scalePredict.text = str(
                count[count.coinMinor == count.coinMinor.max()].scale.tolist()[0])+"Min"
        return

    def transcribe(self):
        self.q_ctrl.put(self.parameters["system"]["PITCH"])
        return

    def on_spinner_transcribe(self):
        self.parameters["transcription"]["window_len"] = int(
            self.spinnerWindow.text)
        self.parameters["transcription"]["waveform"] = self.spinnerWaveform.text.lower(
        )
        self.parameters["transcription"]["type_onset"] = self.spinnerOnset.text[8:].lower()
        self.parameters["transcription"]["shift"] = float(
            self.spinnerShift.text)
        return

    def update_param_zoom(self):
        self.posx_min = self.transcription["minX"]
        self.posx_max = self.transcription["maxX"]
        self.posy_min = self.transcription["minY"]
        self.posy_max = self.transcription["maxY"]
        return

    def zoom_in(self):
        w_win = self.posx_max - self.posx_min
        w_hop = w_win/2
        h_win = self.posy_max - self.posy_min
        h_hop = h_win/2
        self.posx_max -= w_hop/4
        self.posy_max -= h_hop/4
        self.posx_min += w_hop/4
        self.posy_min += h_hop/4

        self.transcription["spectrogram"].set_xlim(
            self.posx_min, self.posx_max)
        self.transcription["spectrogram"].set_ylim(
            self.posy_min, self.posy_max)
        self.transcription["spectralDif"].set_xlim(
            self.posx_min, self.posx_max)
        self.transcription["song"].set_xlim(self.posx_min, self.posx_max)
        self.transcription["canvas"].draw()
        return

    def zoom_out(self):
        w_win = self.posx_max - self.posx_min
        w_hop = w_win/2
        h_win = self.posy_max - self.posy_min
        h_hop = h_win/2

        self.posx_max += w_hop/4
        self.posy_max += h_hop/4
        self.posx_min -= w_hop/4
        self.posy_min -= h_hop/4

        if self.posx_max > self.transcription["maxX"]:
            self.posx_max = self.transcription["maxX"]

        if self.posx_min < self.transcription["minX"]:
            self.posx_min = self.transcription["minX"]

        if self.posy_max > self.transcription["maxY"]:
            self.posy_max = self.transcription["maxY"] + 100

        if self.posy_min < self.transcription["minY"]:
            self.posy_min = self.transcription["minY"]

        self.transcription["spectrogram"].set_xlim(
            self.posx_min, self.posx_max)
        self.transcription["spectralDif"].set_xlim(
            self.posx_min, self.posx_max)
        self.transcription["song"].set_xlim(self.posx_min, self.posx_max)

        self.transcription["spectrogram"].set_ylim(
            self.posy_min, self.posy_max)

        self.transcription["canvas"].draw()
        return

    def shiftR(self):
        if self.posx_max < self.transcription["maxX"]:
            w_win = self.posx_max - self.posx_min
            shift = w_win * 0.1
            self.posx_max += shift
            self.posx_min += shift
            if self.posx_max > self.transcription["maxX"]:
                self.posx_max = self.transcription["maxX"]
                self.posx_min = self.posx_max - w_win

            self.transcription["spectrogram"].set_xlim(
                self.posx_min, self.posx_max)
            self.transcription["spectralDif"].set_xlim(
                self.posx_min, self.posx_max)
            self.transcription["song"].set_xlim(self.posx_min, self.posx_max)

            self.transcription["canvas"].draw()
        return

    def shiftL(self):
        if self.posx_min > self.transcription["minX"]:
            w_win = self.posx_max - self.posx_min
            shift = w_win * 0.1
            self.posx_max -= shift
            self.posx_min -= shift
            if self.posx_min < self.transcription["minX"]:
                self.posx_min = self.transcription["minX"]
                self.posx_max = self.posx_min + w_win

            self.transcription["spectrogram"].set_xlim(
                self.posx_min, self.posx_max)
            self.transcription["spectralDif"].set_xlim(
                self.posx_min, self.posx_max)
            self.transcription["song"].set_xlim(self.posx_min, self.posx_max)

            self.transcription["canvas"].draw()
        return

    def shiftUp(self):
        if self.posy_max < self.transcription["maxY"]:
            h_win = self.posy_max - self.posy_min
            shift = h_win * 0.1
            self.posy_max += shift
            self.posy_min += shift
            if self.posy_max > self.transcription["maxY"] + 100:
                self.posy_max = self.transcription["maxY"] + 100
                self.posy_min = self.posy_max - h_win

            self.transcription["spectrogram"].set_ylim(
                self.posy_min, self.posy_max)

            self.transcription["canvas"].draw()
        return

    def shiftDown(self):
        if self.posy_min > self.transcription["minY"]:
            h_win = self.posy_max - self.posy_min
            shift = h_win * 0.1
            self.posy_max -= shift
            self.posy_min -= shift
            if self.posy_min < self.transcription["minY"]:
                self.posy_min = self.transcription["minY"]
                self.posy_max = self.posy_min + h_win

            self.transcription["spectrogram"].set_ylim(
                self.posy_min, self.posy_max)

            self.transcription["canvas"].draw()
        return

    def control_plots(self, state=True):
        self.UpBtn.disabled = state
        self.LBtn.disabled = state
        self.RBtn.disabled = state
        self.DownBtn.disabled = state
        self.PlusBtn.disabled = state
        self.MinusBtn.disabled = state


class myTabVocoder(BoxLayout):

    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)
        self.spinnerWindow = self.ids.spinnerWindow.text
        self.spinnerModRate = self.ids.spinnerModRate.text
        self.spinnerVocal = self.ids.spinnerVocal.text

        self.knobOver = self.ids.knobOver
        self.valueOver = self.ids.valueOver
        self.knobOver.value = 75

        self.valuePitch = self.ids.valuePitch
        self.valuePitch.value = 0

        self.parameters = parameters

    def on_spinner_select(self, text):
        self.parameters["vocoder"]["window"] = self.spinnerWindow = self.ids.spinnerWindow.text
        self.parameters["vocoder"]["mod_rate"] = int(
            self.ids.spinnerModRate.text[0])/int(self.ids.spinnerModRate.text[2])
        self.spinnerModRate = self.ids.spinnerModRate.text
        self.parameters["vocoder"]["vocalFx"] = self.spinnerVocal = self.ids.spinnerVocal.text

        if self.parameters["vocoder"]["vocalFx"] == 'Pitch Shifter':
            self.valuePitch.disabled = False
            self.knobOver.show_marker = False
            self.valueOver.disabled = True
        else:
            self.valuePitch.disabled = True
            self.knobOver.show_marker = True
            self.valueOver.disabled = False

        return

    def filters_knob(self, knobParam):
        if self.parameters["vocoder"]["vocalFx"] != 'Pitch Shifter':
            if knobParam == "knobOver":
                value = self.knobOver.value
                self.valueOver.text = str(int(value))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.-]+$", value):
            if(value[0] == "-" and len(value) == 1):
                return
            if knobParam == "knobOver":
                if int(value) < 1:
                    value = 1
                elif int(value) > 99:
                    value = 99
            if knobParam == "knobPitch":
                if int(value) < -20:
                    value = -20
                elif int(value) > 20:
                    value = 20
        else:
            return

        if knobParam == "knobPitch":
            self.valuePitch.value = self.parameters["vocoder"]["n"] = round(
                float(value), 1)

        if knobParam == "knobOver":
            self.knobOver.value = int(value)
            window_size = int(1024*self.parameters["vocoder"]["mod_rate"])
            hop = window_size//4
            self.parameters["vocoder"]["overlap"] = self.knobOver.value * \
                (window_size/(window_size - hop))/100
        return


class myTabDistortion(BoxLayout):

    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters

        self.knobThresh = self.ids.knobThresh
        self.knobPreAmp = self.ids.knobPreAmp

        self.spinnerDistype = self.ids.spinnerDistype.text

        self.valueThresh = self.ids.valueThresh
        self.valuePreAmp = self.ids.valuePreAmp

        self.knobThresh.value = 100
        self.knobPreAmp.value = 1

    def filters_knob(self, knobParam):
        if knobParam == "knobThresh":
            value = self.knobThresh.value
            self.valueThresh.text = str(int(value))
        elif knobParam == "knobPreAmp":
            value = self.knobPreAmp.value
            self.valuePreAmp.text = str(round(float(value), 1))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):
            if knobParam == "knobThresh":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobPreAmp":
                if float(value) < 1:
                    value = 1
                elif float(value) > 50:
                    value = 50
        else:
            return

        if knobParam == "knobThresh":
            self.knobThresh.value = float(value)
            self.parameters["distortion"]["threshold"] = self.knobThresh.value/100
        elif knobParam == "knobPreAmp":
            self.parameters["distortion"]["g_db"] = self.knobPreAmp.value = float(
                value)
        return

    def on_spinner_select(self, text):
        self.parameters["distortion"]["distortionType"] = self.ids.spinnerDistype.text.replace(
            " ", "")
        self.spinnerDistype = self.ids.spinnerDistype.text
        return


class myTabTremolo(BoxLayout):

    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters

        self.knobLFO = self.ids.knobLFO
        self.knobAmp = self.ids.knobAmp

        self.valueLFO = self.ids.valueLFO
        self.valueAmp = self.ids.valueAmp

        self.knobLFO.value = 10
        self.knobAmp.value = 1

    def filters_knob(self, knobParam):
        if knobParam == "knobLFO":
            value = self.knobLFO.value
            self.valueLFO.text = str(int(value))
        elif knobParam == "knobAmp":
            value = self.knobAmp.value
            self.valueAmp.text = str(round(float(value), 1))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):
            if knobParam == "knobLFO":
                if int(value) < 1:
                    value = 1
                elif int(value) > 5000:
                    value = 5000

            elif knobParam == "knobAmp":
                if float(value) < 1:
                    value = 1
                elif float(value) > 10:
                    value = 10
        else:
            return

        if knobParam == "knobLFO":
            self.parameters["tremolo"]["fLFO"] = self.knobLFO.value = int(
                value)
        elif knobParam == "knobAmp":
            self.parameters["tremolo"]["alpha"] = self.knobAmp.value = round(
                float(value), 1)
        return


class myTabAlien(BoxLayout):

    def __init__(self, canvas, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters
        self.canvasFig = canvas["canvas"]
        self.line = canvas["line"]
        self.ax = canvas["ax"]

        self.graph = self.ids.graph
        self.graph.add_widget(self.canvasFig)

        self.knobF = self.ids.knobF
        self.knobHarm = self.ids.knobHarm

        self.valueF = self.ids.valueF
        self.valueHarm = self.ids.valueHarm

        self.knobF.value = 1
        self.knobHarm.value = 1

    def filters_knob(self, knobParam):
        if knobParam == "knobF":
            value = self.knobF.value
            self.valueF.text = str(round(float(value), 1))
        elif knobParam == "knobHarm":
            value = self.knobHarm.value
            self.valueHarm.text = str(int(value))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):
            if knobParam == "knobF":
                if float(value) < 1:
                    value = 1
                elif float(value) > 50:
                    value = 50

            elif knobParam == "knobHarm":
                if int(value) < 1:
                    value = 1
                elif int(value) > 10:
                    value = 10
        else:
            return

        if knobParam == "knobF":
            self.parameters["alien"]["freq"] = self.knobF.value = float(value)
            self.plot_alien()
        elif knobParam == "knobHarm":
            self.parameters["alien"]["n_harmonics"] = self.knobHarm.value = int(
                value)
            self.plot_alien()
        return

    def plot_alien(self):
        x = np.linspace(0, 2*np.pi, 1024)
        y = np.sin(x*self.parameters["alien"]["freq"])
        if self.parameters["alien"]["n_harmonics"] > 1:
            for i in range(2, self.parameters["alien"]["n_harmonics"]+1):
                y *= np.sin(x*self.parameters["alien"]["freq"]*i)

        y = y/np.max(np.abs(y))
        self.line.set_ydata(y)
        self.ax.set_ylim(-np.max(np.abs(y[(y <= 0)])),
                         np.max(np.abs(y[(y >= 0)])))

        self.canvasFig.draw()
        return


class myTabReverb(BoxLayout):
    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters

        self.knobDry = self.ids.knobDry
        self.knobWet = self.ids.knobWet
        self.knobRoom = self.ids.knobRoom
        self.knobColorless = self.ids.knobColorless
        self.knobGColorless = self.ids.knobGColorless
        self.knobDiff = self.ids.knobDiff

        self.valueDry = self.ids.valueDry
        self.valueWet = self.ids.valueWet
        self.valueRoom = self.ids.valueRoom
        self.valueColorless = self.ids.valueColorless
        self.valueGColorless = self.ids.valueGColorless
        self.valueDiff = self.ids.valueDiff

        self.knobDry.value = 100
        self.knobWet.value = 50
        self.knobRoom.value = 40
        self.knobColorless.value = 35
        self.knobGColorless.value = 70
        self.knobDiff.value = 5

    def filters_knob(self, knobParam):
        if knobParam == "knobDry":
            value = self.knobDry.value
            self.valueDry.text = str(int(value))
        elif knobParam == "knobWet":
            value = self.knobWet.value
            self.valueWet.text = str(int(value))
        elif knobParam == "knobRoom":
            value = self.knobRoom.value
            self.valueRoom.text = str(int(value))
        elif knobParam == "knobColorless":
            value = self.knobColorless.value
            self.valueColorless.text = str(int(value))
        elif knobParam == "knobGColorless":
            value = self.knobGColorless.value
            self.valueGColorless.text = str(int(value))
        elif knobParam == "knobDiff":
            value = self.knobDiff.value
            self.valueDiff.text = str(int(value))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):

            if knobParam == "knobDry":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobWet":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobRoom":
                if int(value) < 40:
                    value = 40
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobColorless":
                if int(value) < 1:
                    value = 1
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobGColorless":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobDiff":
                if int(value) < 1:
                    value = 1
                elif int(value) > 10:
                    value = 10

        else:
            return

        if knobParam == "knobDry":
            self.parameters["reverb"]["dry"] = int(value)/100
            self.knobDry.value = int(value)
        elif knobParam == "knobWet":
            self.parameters["reverb"]["wet"] = (int(value))/100
            self.knobWet.value = int(value)
        elif knobParam == "knobRoom":
            self.parameters["reverb"]["room_size"] = int(value)
            self.knobRoom.value = int(value)
        elif knobParam == "knobColorless":
            self.parameters["reverb"]["colorless"] = int(value)
            self.knobColorless.value = int(value)
        elif knobParam == "knobGColorless":
            self.parameters["reverb"]["g_colorless"] = (int(value))/100
            self.knobGColorless.value = int(value)
        elif knobParam == "knobDiff":
            self.parameters["reverb"]["diffusion"] = self.knobDiff.value = int(
                value)

        return


class myTabChorus(BoxLayout):
    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters

        self.knobDry = self.ids.knobDry
        self.knobWet1 = self.ids.knobWet1
        self.knobWet2 = self.ids.knobWet2
        self.knobDelay = self.ids.knobDelay
        self.knobDepth = self.ids.knobDepth
        self.knobLFO1 = self.ids.knobLFO1
        self.knobLFO2 = self.ids.knobLFO2

        self.valueDry = self.ids.valueDry
        self.valueWet1 = self.ids.valueWet1
        self.valueWet2 = self.ids.valueWet2
        self.valueDelay = self.ids.valueDelay
        self.valueDepth = self.ids.valueDepth
        self.valueLFO1 = self.ids.valueLFO1
        self.valueLFO2 = self.ids.valueLFO2

        self.knobDry.value = 100
        self.knobWet1.value = 50
        self.knobWet2.value = 50
        self.knobDelay.value = 2.0
        self.knobDepth.value = 0.1
        self.knobLFO1.value = 20.0
        self.knobLFO2.value = 10.0

    def filters_knob(self, knobParam):
        if knobParam == "knobDry":
            value = self.knobDry.value
            self.valueDry.text = str(int(value))
        elif knobParam == "knobWet1":
            value = self.knobWet1.value
            self.valueWet1.text = str(int(value))
        elif knobParam == "knobWet2":
            value = self.knobWet2.value
            self.valueWet2.text = str(int(value))
        elif knobParam == "knobDelay":
            value = self.knobDelay.value
            self.valueDelay.text = str(round(float(value), 3))
        elif knobParam == "knobDepth":
            value = self.knobDepth.value
            self.valueDepth.text = str(round(float(value), 3))
        elif knobParam == "knobLFO1":
            value = self.knobLFO1.value
            self.valueLFO1.text = str(round(float(value), 3))
        elif knobParam == "knobLFO2":
            value = self.knobLFO2.value
            self.valueLFO2.text = str(round(float(value), 3))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):

            if knobParam == "knobDry":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobWet1":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobWet2":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobDelay":
                if float(value) < 0:
                    value = 0
                elif float(value) > 20.0:
                    value = 20.0

            elif knobParam == "knobDepth":
                if float(value) < 0:
                    value = 0
                elif float(value) > 10.0:
                    value = 10.0

            elif knobParam == "knobLFO1":
                if float(value) < 0:
                    value = 0
                elif float(value) > 20.0:
                    value = 20.0

            elif knobParam == "knobLFO2":
                if float(value) < 0:
                    value = 0
                elif float(value) > 20.0:
                    value = 20.0
        else:
            return

        if knobParam == "knobDry":
            self.parameters["chorus"]["dry"] = int(value)/100
            self.knobDry.value = int(value)
        elif knobParam == "knobWet1":
            self.parameters["chorus"]["wet"][0] = int(value)/100
            self.knobWet1.value = int(value)
        elif knobParam == "knobWet2":
            self.parameters["chorus"]["wet"][1] = int(value)/100
            self.knobWet2.value = int(value)
        elif knobParam == "knobDelay":
            self.parameters["chorus"]["delay_sec"] = float(value)/1000
            self.knobDelay.value = float(value)
        elif knobParam == "knobDepth":
            self.parameters["chorus"]["depth"] = float(value)/1000
            self.knobDepth.value = float(value)
        elif knobParam == "knobLFO1":
            self.parameters["chorus"]["fLFO"][0] = self.knobLFO1.value = float(
                value)
        elif knobParam == "knobLFO2":
            self.parameters["chorus"]["fLFO"][1] = self.knobLFO2.value = float(
                value)
        return


class myTabFlanger(BoxLayout):
    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters

        self.knobDry = self.ids.knobDry
        self.knobWet = self.ids.knobWet
        self.knobDelay = self.ids.knobDelay
        self.knobDepth = self.ids.knobDepth
        self.knobLFO = self.ids.knobLFO

        self.valueDry = self.ids.valueDry
        self.valueWet = self.ids.valueWet
        self.valueDelay = self.ids.valueDelay
        self.valueDepth = self.ids.valueDepth
        self.valueLFO = self.ids.valueLFO

        self.knobDry.value = 100
        self.knobWet.value = 100
        self.knobDelay.value = 2.0
        self.knobDepth.value = 0.1
        self.knobLFO.value = 10.0

    def filters_knob(self, knobParam):
        if knobParam == "knobDry":
            value = self.knobDry.value
            self.valueDry.text = str(int(value))
        elif knobParam == "knobWet":
            value = self.knobWet.value
            self.valueWet.text = str(int(value))
        elif knobParam == "knobDelay":
            value = self.knobDelay.value
            self.valueDelay.text = str(round(float(value), 3))
        elif knobParam == "knobDepth":
            value = self.knobDepth.value
            self.valueDepth.text = str(round(float(value), 3))
        elif knobParam == "knobLFO":
            value = self.knobLFO.value
            self.valueLFO.text = str(round(float(value), 3))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):

            if knobParam == "knobDry":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobWet":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobDelay":
                if float(value) < 0:
                    value = 0
                elif float(value) > 60.0:
                    value = 60.0

            elif knobParam == "knobDepth":
                if float(value) < 0:
                    value = 0
                elif float(value) > 20.0:
                    value = 20.0

            elif knobParam == "knobLFO":
                if float(value) < 0:
                    value = 0
                elif float(value) > 20.0:
                    value = 20.0
        else:
            return

        if knobParam == "knobDry":
            self.parameters["flanger"]["dry"] = int(value)/100
            self.knobDry.value = int(value)
        elif knobParam == "knobWet":
            self.parameters["flanger"]["wet"] = int(value)/100
            self.knobWet.value = int(value)
        elif knobParam == "knobDelay":
            self.parameters["flanger"]["delay_sec"] = float(value)/1000
            self.knobDelay.value = float(value)
        elif knobParam == "knobDepth":
            self.parameters["flanger"]["depth"] = float(value)/1000
            self.knobDepth.value = float(value)
        elif knobParam == "knobLFO":
            self.parameters["flanger"]["fLFO"] = self.knobLFO.value = float(
                value)
        return


class myTabDelay(BoxLayout):
    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters

        self.knobDry = self.ids.knobDry
        self.knobWet = self.ids.knobWet
        self.knobOffset = self.ids.knobOffset
        self.knobGFeed = self.ids.knobGFeed
        self.knobRoom = self.ids.knobRoom

        self.valueDry = self.ids.valueDry
        self.valueWet = self.ids.valueWet
        self.valueOffset = self.ids.valueOffset
        self.valueGFeed = self.ids.valueGFeed
        self.valueRoom = self.ids.valueRoom

        self.knobDry.value = 100
        self.knobWet.value = 50
        self.knobOffset.value = 500.0
        self.knobGFeed.value = 50.000
        self.knobRoom.value = 20

    def filters_knob(self, knobParam):
        if knobParam == "knobDry":
            value = self.knobDry.value
            self.valueDry.text = str(int(value))
        elif knobParam == "knobWet":
            value = self.knobWet.value
            self.valueWet.text = str(int(value))
        elif knobParam == "knobOffset":
            value = self.knobOffset.value
            self.valueOffset.text = str(round(float(value), 3))
        elif knobParam == "knobGFeed":
            value = self.knobGFeed.value
            self.valueGFeed.text = str(int(value))
        elif knobParam == "knobRoom":
            value = self.knobRoom.value
            self.valueRoom.text = str(int(value))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):

            if knobParam == "knobDry":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobWet":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobOffset":
                if float(value) < 0:
                    value = 0
                elif float(value) > 500:
                    value = 500

            elif knobParam == "knobGFeed":
                if int(value) < 0:
                    value = 0
                elif int(value) > 100:
                    value = 100

            elif knobParam == "knobRoom":
                if int(value) < 0:
                    value = 0
                elif int(value) > 20:
                    value = 20
        else:
            return

        if knobParam == "knobDry":
            self.parameters["delay"]["dry"] = int(value)/100
            self.knobDry.value = int(value)
        elif knobParam == "knobWet":
            self.parameters["delay"]["wet"] = int(value)/100
            self.knobWet.value = int(value)
        elif knobParam == "knobOffset":
            self.parameters["delay"]["delay_sec"] = round(float(value), 3)/1000
            self.knobOffset.value = float(value)
        elif knobParam == "knobGFeed":
            self.parameters["delay"]["gfb"] = int(value)/100
            self.knobGFeed.value = int(value)
        elif knobParam == "knobRoom":
            self.parameters["delay"]["bounces"] = self.knobRoom.value = int(
                value)
        return


class myTabN(BoxLayout):
    def __init__(self, canvas, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters
        self.canvasFig = canvas["canvas"]
        self.line = canvas["line"]
        self.lineFC = canvas["lineFC"]

        self.graph = self.ids.graph
        self.graph.add_widget(self.canvasFig)

        self.knobFc = self.ids.knobFc
        self.knobQ = self.ids.knobQ
        self.knobGain = self.ids.knobGain

        self.valueFc = self.ids.valueFc
        self.valueQ = self.ids.valueQ
        self.valueGain = self.ids.valueGain

        self.knobFc.value = 1000
        self.knobQ.value = 2
        self.knobGain.value = 0

    def filters_knob(self, knobParam):
        if knobParam == "knobFc":
            value = self.knobFc.value
            self.valueFc.text = str(round(float(value), 1))
        elif knobParam == "knobQ":
            value = self.knobQ.value
            self.valueQ.text = str(round(float(value), 1))
        elif knobParam == "knobGain":
            value = self.knobGain.value
            self.valueGain.text = str(round(float(value), 1))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):

            if knobParam == "knobFc":
                if float(value) < 20:
                    value = 20
                elif float(value) > 20000:
                    value = 20000

            elif knobParam == "knobQ":
                if float(value) > 10:
                    value = 10
                elif float(value) < 0:
                    value = 0

            elif knobParam == "knobGain":
                if float(value) < 0:
                    value = 0
                elif float(value) > 10:
                    value = 10
        else:
            return

        if knobParam == "knobFc":
            self.parameters["notch"]["fc"] = self.knobFc.value = float(value)
            self.filters(float(value), float(self.valueQ.text),
                         float(self.valueGain.text))
        elif knobParam == "knobQ":
            self.parameters["notch"]["Q"] = self.knobQ.value = float(value)
            self.filters(float(self.valueFc.text), float(
                value), float(self.valueGain.text))
        elif knobParam == "knobGain":
            self.parameters["notch"]["g"] = self.knobGain.value = float(value)
            self.filters(float(self.valueFc.text), float(
                self.valueQ.text), float(value))
        return

    def filters(self, fc, Q, g):
        ba = notch(self.parameters["system"]["fs"], self.parameters["notch"]
                   ["fc"], self.parameters["notch"]["Q"], self.parameters["notch"]["g"])
        freq, h = sg.freqz(ba[0], ba[1], fs=2*np.pi)
        self.line.set_ydata(20 * np.log10(abs(h)))
        self.lineFC.set_xdata(self.parameters["notch"]["fc"])

        self.canvasFig.draw()
        return


class myTabBP(BoxLayout):
    def __init__(self, canvas, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters
        self.canvasFig = canvas["canvas"]
        self.line = canvas["line"]
        self.lineFCI = canvas["lineFCI"]
        self.lineFCS = canvas["lineFCS"]

        self.graph = self.ids.graph
        self.graph.add_widget(self.canvasFig)

        self.knobFci = self.ids.knobFci
        self.knobFcs = self.ids.knobFcs
        self.knobAtt = self.ids.knobAtt
        self.knobGain = self.ids.knobGain

        self.valueFci = self.ids.valueFci
        self.valueFcs = self.ids.valueFcs
        self.valueAtt = self.ids.valueAtt
        self.valueGain = self.ids.valueGain

        self.knobFci.value = 500
        self.knobFcs.value = 3500
        self.knobAtt.value = 30
        self.knobGain.value = 0

    def filters_knob(self, knobParam):
        if knobParam == "knobFci":
            value = self.knobFci.value
            self.valueFci.text = str(round(float(value), 1))
        elif knobParam == "knobFcs":
            value = self.knobFcs.value
            self.valueFcs.text = str(round(float(value), 1))
        elif knobParam == "knobAtt":
            value = self.knobAtt.value
            self.valueAtt.text = str(round(float(value), 1))
        elif knobParam == "knobGain":
            value = self.knobGain.value
            self.valueGain.text = str(round(float(value), 1))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):

            if knobParam == "knobFci":
                if float(value) < 20:
                    value = 20
                elif float(value) >= round(float(self.valueFcs.text), 1):
                    value = round(float(self.valueFcs.text) - 1, 1)

            elif knobParam == "knobFcs":
                if float(value) > 20000:
                    value = 20000
                elif float(value) <= round(float(self.valueFci.text), 1):
                    value = round(float(self.valueFci.text) + 1, 1)

            elif knobParam == "knobAtt":
                if float(value) < 0:
                    value = 0
                elif float(value) > 100:
                    value = 100

            elif knobParam == "knobGain":
                if float(value) < 0:
                    value = 0
                elif float(value) > 10:
                    value = 10
        else:
            return

        if knobParam == "knobFci":
            self.parameters["bp"]["fci"] = self.knobFci.value = float(value)
            self.filters()
        elif knobParam == "knobFcs":
            self.parameters["bp"]["fcs"] = self.knobFcs.value = float(value)
            self.filters()
        elif knobParam == "knobAtt":
            self.parameters["bp"]["att"] = self.knobAtt.value = float(value)
            self.filters()
        elif knobParam == "knobGain":
            self.parameters["bp"]["g"] = self.knobGain.value = float(value)
            self.filters()
        return

    def filters(self):
        buffer_filter = bp(self.parameters["system"]["fs"], self.parameters["bp"]["fci"],
                           self.parameters["bp"]["fcs"], self.parameters["bp"]["att"], self.parameters["bp"]["g"]/2)
        freq, h = sg.freqz(buffer_filter, fs=2*np.pi)
        self.line.set_ydata(20 * np.log10(abs(h)))
        self.lineFCI.set_xdata(self.parameters["bp"]["fci"])
        self.lineFCS.set_xdata(self.parameters["bp"]["fcs"])

        self.canvasFig.draw()
        return


class myTabLPHP(BoxLayout):

    def __init__(self, canvas, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters
        self.canvasFig = canvas["canvas"]
        self.line = canvas["line"]
        self.lineFC = canvas["lineFC"]
        self.graph = self.ids.graph
        self.graph.add_widget(self.canvasFig)

        self.knobFc = self.ids.knobFc
        self.knobAtt = self.ids.knobAtt
        self.knobGain = self.ids.knobGain

        self.valueFc = self.ids.valueFc
        self.valueAtt = self.ids.valueAtt
        self.valueGain = self.ids.valueGain

        self.knobFc.value = 1000
        self.knobAtt.value = 30
        self.knobGain.value = 0

    def filters_knob(self, knobParam):
        if knobParam == "knobFc":
            value = self.knobFc.value
            self.valueFc.text = str(round(float(value), 1))
        elif knobParam == "knobAtt":
            value = self.knobAtt.value
            self.valueAtt.text = str(round(float(value), 1))
        elif knobParam == "knobGain":
            value = self.knobGain.value
            self.valueGain.text = str(round(float(value), 1))
        return

    def filter_inputs(self, value, knobParam):
        if re.match("^[0-9.]+$", value):
            if knobParam == "knobFc":
                if float(value) < 20:
                    value = 20
                elif float(value) > 20000:
                    value = 20000

            elif knobParam == "knobAtt":
                if float(value) < 0:
                    value = 0
                elif float(value) > 100:
                    value = 100

            elif knobParam == "knobGain":
                if float(value) < 0:
                    value = 0
                elif float(value) > 10:
                    value = 10
        else:
            return

        if knobParam == "knobFc":
            self.parameters["lp"]["fc"] = self.parameters["hp"]["fc"] = self.knobFc.value = float(
                value)
            self.filters()
        elif knobParam == "knobAtt":
            self.parameters["lp"]["att"] = self.parameters["hp"]["att"] = self.knobAtt.value = float(
                value)
            self.filters()
        elif knobParam == "knobGain":
            self.parameters["lp"]["g"] = self.parameters["hp"]["g"] = self.knobGain.value = float(
                value)
            self.filters()
        return

    def filters(self):
        if self.parameters["system"]["fx"] == 'hp':
            buffer_filter = hp(self.parameters["system"]["fs"], self.parameters["hp"]
                               ["fc"], self.parameters["hp"]["att"], self.parameters["hp"]["g"])
        else:
            buffer_filter = lp(self.parameters["system"]["fs"], self.parameters["lp"]
                               ["fc"], self.parameters["lp"]["att"], self.parameters["lp"]["g"])
        freq, h = sg.freqz(buffer_filter, fs=2*np.pi)
        self.line.set_ydata(20 * np.log10(abs(h)))
        self.lineFC.set_xdata(self.parameters["hp"]["fc"])

        self.canvasFig.draw()
        return


class Matty(FloatLayout):
    def __init__(self, parameters, canvas, q_graph, q_param, q_control, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters
        self.parameters_back = copy.deepcopy(parameters)

        self.visualizer = canvas["visualizer"]
        self.transcription = canvas["transcription"]

        self.cv_visualizer = canvas["visualizer"]["canvas"]
        self.cv_transcription = canvas["transcription"]["canvas"]

        self.q_graph = q_graph
        self.q_param = q_param
        self.q_control = q_control

        self.cv_lp = canvas["lp"]
        self.cv_hp = canvas["hp"]
        self.cv_bp = canvas["bp"]
        self.cv_notch = canvas["notch"]
        self.cv_alien = canvas["alien"]

        self.box1 = self.ids.box1
        self.box1.add_widget(self.cv_visualizer)

        self.box2 = self.ids.box2
        self.box2.add_widget(self.cv_transcription)

        self.tabPiano = myTabPiano(
            self.transcription, self.q_control, self.parameters)
        self.ids.tabPiano.add_widget(self.tabPiano)

        self.ids.tablp.add_widget(myTabLPHP(self.cv_lp, self.parameters))
        self.ids.tabhp.add_widget(myTabLPHP(self.cv_hp, self.parameters))

        self.tabBP = myTabBP(self.cv_bp, self.parameters)
        self.ids.tabbp.add_widget(self.tabBP)

        self.ids.tabn.add_widget(myTabN(self.cv_notch, self.parameters))
        self.ids.tabdelay.add_widget(myTabDelay(self.parameters))
        self.ids.tabflanger.add_widget(myTabFlanger(self.parameters))
        self.ids.tabChorus.add_widget(myTabChorus(self.parameters))
        self.ids.tabRev.add_widget(myTabReverb(self.parameters))
        self.ids.tabAlien.add_widget(
            myTabAlien(self.cv_alien, self.parameters))
        self.ids.tabTremolo.add_widget(myTabTremolo(self.parameters))
        self.ids.tabDistortion.add_widget(myTabDistortion(self.parameters))
        self.ids.tabVocoder.add_widget(myTabVocoder(self.parameters))

        self.playBtn = self.ids.playBtn
        self.switchMic = self.ids.switchMic

        self.valueGain = self.ids.valueGain

        self.ids.get_file.text = self.parameters["system"]["path"]
        self.file_path = StringProperty("No file chosen")
        self.the_popup = ObjectProperty(None)

        self.currentTab("Piano")

        Clock.schedule_interval(self.drawGraph, 0.01)

    def open_popup(self):
        self.the_popup = FileChoosePopup(load=self.load)
        self.the_popup.ids.filechooser.path = self.parameters["system"]["path"]
        self.the_popup.open()

    def load(self, selection):
        self.file_path = str(selection[0])
        self.the_popup.dismiss()
        self.parameters["system"]["path"] = self.file_path

        self.visualizer["time"].set_title(self.parameters["system"]["path"][self.parameters["system"]["path"].rfind(
            '\\')+1:]+" in Real-Time", color="white")

        if ".wav" in self.parameters["system"]["path"]:
            self.tabPiano.TranscribeBtn.disabled = False
        else:
            self.tabPiano.TranscribeBtn.disabled = True
        # check for non-empty list i.e. file selected
        if self.file_path:
            self.ids.get_file.text = self.file_path

    def play(self):
        if ".wav" in self.parameters["system"]["path"] or self.parameters["system"]["live"]:
            if self.playBtn.text == "P L A Y" or self.playBtn.text == "L I S T E N":
                self.playBtn.text = "S T O P"
                self.playBtn.background_color = (230/255, 57/255, 63/255, 1)
                self.tabPiano.TranscribeBtn.disabled = True
                self.tabPiano.control_plots(True)
                self.switchMic.disabled = True
                self.tabPiano.reset_counters()
                self.q_control.put(self.parameters["system"]["PLAY"])
            elif self.parameters["system"]["live"]:
                self.playBtn.text = "L I S T E N"
                self.tabPiano.TranscribeBtn.disabled = True
                self.tabPiano.control_plots(True)
                self.playBtn.background_color = (254/255, 97/255, 0, 1)
                self.q_control.put(self.parameters["system"]["STOP"])
            else:
                self.playBtn.text = "P L A Y"
                self.tabPiano.TranscribeBtn.disabled = False
                self.tabPiano.control_plots(False)
                self.playBtn.background_color = (0, 166/255, 102/255, 1)
                self.q_control.put(self.parameters["system"]["STOP"])
        return

    def on_switch(self):
        if self.switchMic.active:
            self.playBtn.text = "L I S T E N"
            self.playBtn.background_color = (254/255, 97/255, 0, 1)
            self.parameters["system"]["live"] = bool(self.switchMic.active)
        else:
            self.playBtn.text = "P L A Y"
            self.playBtn.background_color = (0, 166/255, 102/255, 1)
            self.parameters["system"]["live"] = bool(self.switchMic.active)
        return

    def volumen(self, gain):
        self.parameters["system"]["gain"] = gain
        self.valueGain.text = str(round(gain, 1)) + " dB"
        return

    def drawGraph(self, dt):
        if self.parameters_back != self.parameters:
            self.q_param.put(self.parameters)
            self.parameters_back = copy.deepcopy(self.parameters)

        try:
            dataInt = self.q_graph.get_nowait()
        except:
            return

        if type(dataInt) is str and dataInt == self.parameters["system"]["END"]:
            self.endPlay()
            return

        if type(dataInt) is dict and len(dataInt) > 0:
            self.updatePTrans(dataInt)

        elif type(dataInt) is not str and len(dataInt) > 0:
            self.updateVisualizer(dataInt)

        return

    def currentTab(self, text):
        fx = {"Low Pass": "lp", "High Pass": "hp", "Band Pass": "bp", "Notch": "notch", "Delay": "delay", "Flanger": "flanger",
              "Chorus": "chorus", "Reverb": "reverb", "Alien Vox": "alien", "Tremolo": "tremolo", "Distortion": "distortion", "Vocoder": "vocoder", "Piano": "piano"}
        self.parameters["system"]["fx"] = fx[text]
        return

    def detect_key_(self, nota):
        return detect_key(nota)

    def endPlay(self):
        if self.parameters["system"]["live"]:
            self.playBtn.text = "L I S T E N"
            self.playBtn.background_color = (254/255, 97/255, 0, 1)
        else:
            self.playBtn.text = "P L A Y"
            self.playBtn.background_color = (0, 166/255, 102/255, 1)
        self.tabPiano.reset_keys()
        self.tabPiano.predicted_scale()
        self.switchMic.disabled = False
        return

    def updatePTrans(self, dataInt):
        p_transcription = dataInt
        data = p_transcription["data"]
        data_stft = p_transcription["data_stft"]
        sf = p_transcription["spectral_flux"]

        fs = self.parameters["system"]["fs"]
        w_ = p_transcription["window"]
        window_len = w_.shape[0]

        peak_frames = p_transcription["peaks_f"]
        peak_samples = p_transcription["peaks_s"]
        peak_times = peak_samples/fs

        list_tones = p_transcription["tones"]

        data_norm = data/np.max(data)
        duration = len(data)/fs
        time = np.arange(0, duration, 1/fs)

        lapse = np.sum(np.diff(peak_times))/peak_times.shape[0]
        BPM = 60/lapse
        BPM = round(BPM, 3)

        frec_max = 0

        self.box2.disabled = False
        self.tabPiano.control_plots(False)
        self.transcription["spectrogram"].clear()
        self.transcription["song"].clear()
        self.transcription["spectralDif"].clear()

        # plt.pcolormesh(data_stft[1], data_stft[0], np.abs(data_stft[2]), vmin=0, vmax=1, shading='nearest')
        self.transcription["spectrogram"].specgram(
            data, NFFT=window_len, Fs=fs, window=w_)
        for i in range(len(peak_times)-1):
            frec_list = list(list_tones[i].values())
            tones = list(list_tones[i].keys())
            if np.max(frec_list) > frec_max:
                frec_max = np.max(frec_list)
            self.transcription["spectrogram"].hlines(
                frec_list, peak_times[i], peak_times[i+1], lw=0.5)
            for j, frec in enumerate(frec_list):
                self.transcription["spectrogram"].annotate(
                    tones[j], (peak_times[i], frec+4), fontsize=6)

        min_x = -0.1
        max_x = duration
        min_y = 0
        max_y = frec_max

        self.transcription["spectrogram"].set_ylim(min_y, max_y + 100)
        self.transcription["spectrogram"].set_xlim(min_x, max_x)

        # colors = ["#0D3C55", "#0F5B78", "#117899", "#1395BA", "#5CA793", "#A2B86C", "#EBC844", "#ECAA38", "#EF8B2C", "#F16C20", "#D94E1F", "#C02E1D"]
        self.transcription["song"].plot(
            time, data_norm*max_y, color='#00FF00', label=f"BPM: {BPM}")
        self.transcription["song"].vlines(
            peak_samples/fs, -max_y, max_y, color='#04F2F2', lw=1)
        self.transcription["song"].set_ylim(-max_y, max_y)
        self.transcription["song"].set_xlim(min_x, max_x)
        self.transcription["song"].legend(loc='upper right')

        self.transcription["spectralDif"].plot(
            data_stft[1][:-1], sf*max_y, color='#00FF00', lw=1)
        self.transcription["spectralDif"].plot(
            peak_samples/fs, sf[peak_frames]*max_y, ls='None', marker="x", color='#04F2F2')
        self.transcription["spectralDif"].set_xlim(min_x, max_x)
        self.transcription["spectralDif"].set_ylim(min_y, max_y)

        self.transcription["spectrogram"].set_title(
            self.parameters["system"]["path"][self.parameters["system"]["path"].rfind('\\')+1:]+" - Spectrogram", color="white")
        self.transcription["spectrogram"].set_xlabel(
            "Time [sec]", color="white")
        self.transcription["spectrogram"].set_ylabel(
            "Frecuency [Hz]", color="white")

        self.transcription["song"].set_title(self.parameters["system"]["path"][self.parameters["system"]["path"].rfind(
            '\\')+1:]+" - Detection Onsets", color="white")
        self.transcription["song"].set_xlabel("Time [sec]", color="white")
        self.transcription["song"].set_ylabel(
            "Normalized Amplitude", color="white")

        if self.parameters["transcription"]["type_onset"] == 'percussive':
            self.transcription["spectralDif"].set_title(self.parameters["system"]["path"][self.parameters["system"]["path"].rfind(
                '\\')+1:]+" - Spectral Difference", color="white")
        else:
            self.transcription["spectralDif"].set_title(
                self.parameters["system"]["path"][self.parameters["system"]["path"].rfind('\\')+1:]+" - Phase Deviation", color="white")
        self.transcription["spectralDif"].set_xlabel(
            "Time [sec]", color="white")
        self.transcription["spectralDif"].set_ylabel(
            "Normalized Difference", color="white")

        self.transcription["minX"] = min_x
        self.transcription["maxX"] = max_x
        self.transcription["minY"] = min_y
        self.transcription["maxY"] = max_y

        self.tabPiano.update_param_zoom()
        self.cv_transcription.draw()
        return

    def updateVisualizer(self, dataInt):
        if len(dataInt) < self.parameters["system"]["frames"]:
            dataInt = np.concatenate(
                (dataInt, np.zeros(self.parameters["system"]["frames"] - len(dataInt))))

        # Asignamos los datos a la curva de la variación temporal
        self.visualizer["line"].set_ydata(dataInt)

        # Calculamos la FFT y la Magnitud de la FFT del paqute de datos
        M_gk = abs(np.fft.fft(dataInt)/self.parameters["system"]["frames"])

        if np.max(M_gk) != 0:
            # Asigmanos la Magnitud de la FFT a la curva del espectro
            self.visualizer["line_fft"].set_ydata((M_gk/np.max(M_gk))*5000)
        else:
            self.visualizer["line_fft"].set_ydata(M_gk)

        # Creamos el vector de frecuencia para encontrar la frecuencia dominante
        F = (self.parameters["system"]["fs"]/self.parameters["system"]
             ["frames"])*np.arange(0, self.parameters["system"]["frames"])
        # Tomamos la mitad del espectro para encontrar la Frecuencia Dominante
        M_gk = M_gk[0:self.parameters["system"]["frames"]]

        if np.max(M_gk) != 0:
            M_gk = (M_gk/np.max(M_gk))*5000
            prueba, l = sg.find_peaks(M_gk, height=5000)
            self.visualizer["line_fft_max"].set_ydata(M_gk[prueba])
            self.visualizer["line_fft_max"].set_xdata(F[prueba])
            if F[prueba].size > 0:
                self.visualizer["frec"].legend([self.visualizer["line_fft_max"]], [
                                               f'{round(F[prueba][0], 2)} Hz'], loc='upper right')

            if self.parameters["system"]["fx"] == "piano":
                self.tabPiano.reset_keys()
                for F_nota in F[prueba]:
                    if F_nota > 0 and F_nota < 20000:
                        pitch_detect, key, octava = self.detect_key_(
                            F_nota)
                        self.tabPiano.key_change_color(key)
                        self.tabPiano.octave(octava)

        self.cv_visualizer.draw()
        return


class MainApp(MDApp):
    def build(self):
        self.MDapp_ = MDApp
        self.theme_cls.theme_style = "Dark"
        self.icon = 'img/Logo.png'
        self.title = 'ARFX'
        # self.theme_cls.primary_palette = "BlueGray"
        Builder.load_file('app.kv')
        Window.bind(on_request_close=self.on_request_close)
        return Matty(self.parameters, self.canvasFig, self.q_graph, self.q_param, self.q_control)

    def setParameters(self, parameters, canvas, q_graph, q_param, q_control):
        self.parameters = parameters
        self.canvasFig = canvas
        self.q_graph = q_graph
        self.q_param = q_param
        self.q_control = q_control
        return

    def on_request_close(self, MDApp, *args):
        self.q_control.put(self.parameters["system"]["POISON_PILL"])
        self.MDapp_.get_running_app().stop()
        return True

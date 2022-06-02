<div align="center"><img src="img/Logo.png" width="200" /></div>

<h3 align="center">Procesamiento digital de señales orientado al audio.</h3>

<div align="center">
    <img src="https://img.shields.io/badge/Python-%E2%89%A53.9-blue?logo=python&logoColor=white"/>
    <img src="https://img.shields.io/badge/Kivy-v2.1.0-0c4b33?logo=kivy" />
</div>

------

## 💡 Descripción

Aplicación de escritorio para procesamiento digital de audio, que cuenta con múltiples efectos, filtros, y un transcriptor de melodías. Fue desarrollado para simular el funcionamiento de plug-ins de audio populares en la industria de la música. Realizado como proyecto final de la materia "Procesamiento digital de señales" dictada en la *Universidad Tecnológica Nacional* de Argentina.

## 💻 Lanzar proyecto

### 📋 Pre-requisitos

* El proyecto fue desarrollado sobre [Python v.3.9](https://www.python.org/downloads/release/python-390/) o superiores.
* Se utilizó el *framework* [Kivy v.2.1.0](https://kivy.org/#home) para desplegar la interfaz gráfica.
* Además, se utilizaron las siguientes librerías:
    - [PyAudio 0.2.11](http://people.csail.mit.edu/hubert/pyaudio/)
    - [Matplotlib 3.1.3](https://matplotlib.org)
    - [Scipy 1.7.3](https://scipy.org)
    - [Numpy 1.19.3](https://numpy.org)
    - [MIDIutil](https://midiutil.readthedocs.io/en/1.2.1/)
    - [Pandas 1.3.4](https://pandas.pydata.org)
    
### 🔧 Instalación

Procedemos a instalar los paquetes:

```
pip install "kivy[base]" kivy_examples
pip install kivy-garden
pip install kivymd
pip install PyAudio
pip install matplotlib==3.1.3
pip install scipy
pip install numpy
pip install MIDIUtil
pip install pandas
```

En caso de tener problemas de instalación con PyAudio podemos instalarlo a partir de su [distribución wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

### 🎧 Ejecución

```
python procesamiento.py
```

---

## 📖 La teoria detras del codigo

Para entender más sobre el código implementado detrás de cada efecto, le sugiero echar un vistazo al siguiente [documento](https://nbviewer.org/github/aletbm/ARFX__PDS_aplicado_en_audio/blob/master/docs/PDS_aplicado_en_audio.ipynb).

## ⭐️ Vista previa de la aplicacion

![GIF](./img/preview.gif)


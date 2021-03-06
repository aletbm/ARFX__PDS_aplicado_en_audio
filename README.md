<div align="center"><img src="img/Logo.png" width="200" /></div>

<h3 align="center">Procesamiento digital de se帽ales orientado al audio.</h3>

<div align="center">
    <img src="https://img.shields.io/badge/Python-%E2%89%A53.9-blue?logo=python&logoColor=white"/>
    <img src="https://img.shields.io/badge/Kivy-v2.1.0-0c4b33?logo=kivy" />
</div>

------

## 馃挕 Descripci贸n

Aplicaci贸n de escritorio para procesamiento digital de audio, que cuenta con m煤ltiples efectos, filtros, y un transcriptor de melod铆as. Fue desarrollado para simular el funcionamiento de plug-ins de audio populares en la industria de la m煤sica. Realizado como proyecto final de la materia "Procesamiento digital de se帽ales" dictada en la *Universidad Tecnol贸gica Nacional* de Argentina.

## 馃捇 Lanzar proyecto

### 馃搵 Pre-requisitos

* El proyecto fue desarrollado sobre [Python v.3.9](https://www.python.org/downloads/release/python-390/) o superiores.
* Se utiliz贸 el *framework* [Kivy v.2.1.0](https://kivy.org/#home) para desplegar la interfaz gr谩fica.
* Adem谩s, se utilizaron las siguientes librer铆as:
    - [PyAudio 0.2.11](http://people.csail.mit.edu/hubert/pyaudio/)
    - [Matplotlib 3.1.3](https://matplotlib.org)
    - [Scipy 1.7.3](https://scipy.org)
    - [Numpy 1.19.3](https://numpy.org)
    - [MIDIutil](https://midiutil.readthedocs.io/en/1.2.1/)
    - [Pandas 1.3.4](https://pandas.pydata.org)
    
### 馃敡 Instalaci贸n

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

En caso de tener problemas de instalaci贸n con PyAudio podemos instalarlo a partir de su [distribuci贸n wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
Ademas debemos descomprimir el archivo *garden.rar* en la siguiente direccion 

```
C:\Users\usr\.kivy\garden\
```
para poder visualizar los Knobs de Kivy y que los plots se puedan integrar correctamente en la interfaz grafica.

### 馃帶 Ejecuci贸n

```
python procesamiento.py
```

---

## 馃摉 La teoria detras del codigo

Para entender m谩s sobre el c贸digo implementado detr谩s de cada efecto, le sugiero echar un vistazo al siguiente [documento](https://nbviewer.org/github/aletbm/ARFX__PDS_aplicado_en_audio/blob/master/docs/PDS_aplicado_en_audio.ipynb).

## 猸愶笍 Vista previa de la aplicacion

![GIF](./img/preview.gif)


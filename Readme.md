# Dynamic Image Generation of Formosat-8 Frame Sensor
## Prerequisites
**1.** Install required packages.
```shell
pip install -r Requirements.txt
```
**2.**  Download gdal wheel from [Python Extension Packages for Windows](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
```shell
python -m pip install GDAL-3.4.2-cp37-cp37m-win_amd64.whl
```

## Introduction
<div align="center">
<img src=docs/flowchart.png height="160.5" width="279.75">
</div>
To generate simulated images of Formosat-8 frame sensor, the satellite position, velocity, and time data are used to simulate ephemeris in advance, so that the object-space coordinate of the satellite observation area can be calculated.
In summary, simulated ephemeris, DEM and ortho-image are inputs for generating simulated images. Where the purpose of the input DEM is to provide gridded object space 3D coordinates, and the ortho-image input is to provide gray value to simulated image so that the presentation of simulated images can be close to real image.    


### Ray Tracing Method
<div align="center">
<img src=docs/ray-tracing-method.png height="140.125" width="141">
</div>
With simulated ephemeris, i.e. internal and external orientation are known, one could get the gray scale of each pixel as follow steps.

**Step 1.** Gizen the initial value of elevation $Z^0$

**Step 2.** Input $Z^0$ to topdown collinear equation [^註腳] to get approximate object space coordinate $(X^0,Y^0)$

[^註腳]: 

$$
X = X_A + (Z-Z_A){m_{11}(x-x_0)+ m_{21}(y-y_0) + m_{31}(-f)\over m_{13}(x-x_0)+ m_{23}(y-y_0) + m_{33}(-f)}
$$

$$
Y = Y_A + (Z-Z_A){m_{12}(x-x_0)+ m_{22}(y-y_0) + m_{32}(-f)\over m_{13}(x-x_0)+ m_{23}(y-y_0) + m_{33}(-f)}
$$
Where $(x_0, y_0, -f)$ are internal orientation parameters of the satellite, $(x, y)$ is the , 

**Step 3.**  The DEM elevation $Z^1$ can be interpolated by $(X^0,Y^0)$

**Step 4.** Input $Z^1$ to collinear equation to get (X^1,Y^1)$

**Step 5.** Repeat Step 3 and Step 4 until convergence. The object space coordinate of each specific pixel is then $(X^n,Y^n,Z^n)$

**Step 6.** 
Interpolate the ortho-image with object space coordinate $(X^n,Y^n,Z^n)$ to obtain the gray value of each pixel


## Quick Start
<div align="center">
<img src=docs/gui.png height="301.5" width="241">
</div>

**1.** Execute main.py to activate user interface
```shell
python main.py
```
**2.** Load DEM, ortho-image, ephemeris data through the "Browse" button. Input the latitude and longtitude of target acquisition area, also, an ouput folder to save the generated simulated images and video.

**3.**
Add Gaussian noise to each attitude angle by giving standard deviation if needed.

**4.** Click "Launch" to start generating simulated images. The generation progress will be shown in the gray textbox. Note that, the outputs of this code are as follows:<br>
(1) Pickle files(.pkl) of each input DEM, ortho-image, and ephemeris data.</br>
(2) Every single simulated image and a simulation.mp4 converted from the image sequences.
(3) A metedata.txt which describes the flight time, satellite position in both ECEF and TM2 coordinate systems, satellite attitude as well as the time spent to generate each simulated image.




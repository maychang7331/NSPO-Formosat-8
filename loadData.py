import os
import pickle
import numpy as np  # print (np.__version__) gives 1.18.4
from osgeo import gdal
import cv2
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
import pandas as pd

from linecache import getline
from math import radians, sin, cos
from utilities import rot
from color_balance import simplest_cb


class DEM:
    def __init__(self, filePath):
        # initialize
        self.filePath = filePath
        self.cols = 0.0
        self.rows = 0.0
        self.xlucorner = 0.0
        self.ylucorner = 0.0
        self.cellsize = 0.0
        self.nodata_value = 0.0
        self.avg = 0.0  # 平均高程

    def loadDem(self):
        # Open the file:
        (dirName, extension) = os.path.splitext(self.filePath)

        # deal with headers   
        hdr = [getline(self.filePath, i) for i in range(1, 7)]  # type(hdr) gives <class 'list'>
        values = [float(h.split(" ")[-1].strip()) for h in hdr]
        # split h with " "(space), get the rightmost value and delete unnecessary space with strip

        self.cols, self.rows, xllcorner, yllcorner, self.cellsize, self.nodata_value = values
        self.xlucorner = xllcorner
        self.ylucorner = yllcorner + self.cellsize * self.rows

        # asc to .pkl <class 'numpy.ndarray'>
        while True:
            try:  # if .pkl exists
                with open(dirName + '.pkl', 'rb') as f:
                    DEM = pickle.load(f)
                    # print(type(DEM)) gives <class 'numpy.ndarray'>    # DEM.shape gives (5555, 5922)
                    self.avg = np.nanmean(DEM)  # gives 109.46928947586392
                    f.close()
                    break

            except IOError:  # if not
                arr = np.loadtxt(self.filePath,
                                 skiprows=6)  # Load the asc ndarray into a numpy array   # arr.shape gives (5555, 5922)
                arr = np.where(arr == -9999.0, np.nan,
                               arr)  # replace -9999.0 with nan in order to calculate the average of DEM

                with open(dirName + '.pkl', 'wb') as f:  # then save it as .pkl
                    pickle.dump(arr, f)
                    f.close()

        return DEM  # <class 'numpy.ndarray'>

    """ 在main程式中
        DEM.getDemdict('cols')
        可以被  DEM.__dict__['cols'] 取代
    """
    # def getDemdict(self, arg):
    #     DEMdict = {
    #         'cols' : self.DEMcols,
    #         'rows' : self.DEMrows,
    #         'xlucorner' : self.DEMxlucorner,
    #         'ylucorner' : self.DEMylucorner,
    #         'cellsize' : self.DEMcellsize,
    #         'nodata_value' : self.DEMnodata_value,
    #         'avg' : self.DEMavg
    #     }
    #     return DEMdict.get(arg, lambda: 'Invalid arguments')


class Tif:
    def __init__(self, filePath):
        # initialize
        self.filePath = filePath
        self.cols = 0.0
        self.rows = 0.0
        self.bands = 0.0
        self.xlucorner = 0.0
        self.ylucorner = 0.0
        self.cellsize = 0.0
        self.nodata_value = 0.0

    def loadTif(self):
        # Open the file:
        (dirName, extension) = os.path.splitext(self.filePath)
        tiff = gdal.Open(self.filePath)  # type(tiff) gives <class 'osgeo.gdal.Dataset'>

        # Dimensions
        self.cols = tiff.RasterXSize  # 7402
        self.rows = tiff.RasterYSize  # 6944
        self.bands = tiff.RasterCount  # Number of bands gives 4
        self.xlucorner = tiff.GetGeoTransform()[0]  # gives 195680.0
        self.ylucorner = tiff.GetGeoTransform()[3]  # gives 2695132.0
        self.cellsize = tiff.GetGeoTransform()[1]  # gives 4.0

        # tiff to .pkl <class 'numpy.ndarray'>
        while True:
            try:  # if .pkl exists
                with open(dirName + '.pkl', 'rb') as f:
                    Tif = pickle.load(f)  # type(Tif) gives <class 'numpy.ndarray'>
                    # plt.imshow(Tif)
                    # plt.show()
                    f.close()
                    break

            except IOError:  # if not

                b = tiff.GetRasterBand(1).ReadAsArray()     # np.max(b) gives 16
                g = tiff.GetRasterBand(2).ReadAsArray()
                r = tiff.GetRasterBand(3).ReadAsArray()

                # normalize color to 0~255  # b.dtype gives float64
                b = (b / np.amax(b)) * 255
                g = (g / np.amax(g)) * 255
                r = (r / np.amax(r)) * 255

                bgr = np.dstack((b, g, r))
                bgr = bgr.astype(np.uint8)  # float64 => unit8
                bgr = simplest_cb(bgr)      # color balance    # bgr.shape gives (6944, 7402, 3)

                rgb = bgr[:, :, ::-1]       # cv2 讀圖片時是BGR, 而 matplotlib 是RGB
                # plt.imshow(rgb)
                # plt.show()

                with open(dirName + '.pkl', 'wb') as f:  # save the RGB ndarray as .pkl
                    pickle.dump(rgb, f)
                    f.close()

        return Tif  # type(Tif) gives <class 'numpy.ndarray'>   # rgb

    """ 在main程式中
        Tif.getDemdict('cols')
        可以被  Tif.__dict__['cols'] 取代
    """
    # def getTifdict(self, arg):
    #     Tifdict = {
    #         'cols' : self.Tifcols,
    #         'rows' : self.Tifrows,
    #         'xlucorner' : self.Tifxlucorner,
    #         'ylucorner' : self.Tifylucorner,
    #         'cellsize' : self.Tifcellsize,
    #         'band' : self.bands
    #     }
    #     return Tifdict.get(arg, lambda: 'Invalid arguments')


class Eph:
    def __init__(self, filePath):
        self.filePath = filePath

    def loadEph(self):  # load eph information in .dim
        from coordinateSystem import CoordinateSystem
        cs = CoordinateSystem()

        meta_tree = ET.parse(self.filePath)
        root = meta_tree.getroot()

        ecef_List = []
        eph = pd.DataFrame(columns=['time', 'Lat', 'Lon', 'sat_h', 'TM2_X', 'TM2_Y'])

        # read position values of sat pos (in ecef) and time in .dim file then append to list
        for points in root.findall('Data_Strip/Ephemeris/Corrected_Ephemeris/Point_List/'):
            ecef_List.append([float(points.find('Location/X').text),
                              float(points.find('Location/Y').text),
                              float(points.find('Location/Z').text)])
            eph = eph.append({'time': points.find('TIME').text}, ignore_index=True)

        # coordinate transformation
        for i in range(len(ecef_List)):
            # ecef X Y Z  to llh then save to dataframe eph
            lat, lon, h = cs.ecef_to_llh(ecef_List[i])
            eph.at[i, 'Lat'] = lat
            eph.at[i, 'Lon'] = lon
            eph.at[i, 'sat_h'] = h  # 橢球高 (要加上大地起伏N 和 LVD offset 才是在TWD97座標下的 衛星高度 )

            # llh to TWD97 TM2 then save to dataframe eph
            TM2_X, TM2_Y = cs.LatLon_To_TWD97TM2(lat, lon)
            eph.at[i, 'TM2_X'] = TM2_X
            eph.at[i, 'TM2_Y'] = TM2_Y

        # read orientation values of sat pos (in ecef)
        for idx, row in eph.iterrows():
            for angle in root.findall('Data_Strip/Attitudes/Corrected_Attitudes/ECF_Attitude/Angle_List/'):
                if angle.find('TIME').text == row['time']:

                    R = float(angle.find('ROLL').text)
                    P = float(angle.find('PITCH').text)
                    Y = float(angle.find('YAW').text)
                    rotRPY = rot(Y, 3).dot(rot(P, 2).dot(rot(R, 1)))    # form rotation matrix

                    # turn rotation matrix from ecef coordinate system to local TWD97 TM2
                    Z = np.array(ecef_List[idx])        # position of s in ecef (X、Y、Z)
                    unit_Z = Z / np.linalg.norm(Z)      # unit Z vector

                    satLat_rad, satLon_rad = radians(row['Lat']), radians(row['Lon'])
                    X = np.array([- sin(satLon_rad), cos(satLon_rad), np.linalg.norm(Z) * sin(satLat_rad)])
                    unit_X = X / np.linalg.norm(X)

                    Y = np.cross(unit_Z, unit_X)
                    unit_Y = Y / np.linalg.norm(Y)

                    ecef_X = np.array([1, 0, 0])
                    ecef_Y = np.array([0, 1, 0])
                    ecef_Z = np.array([0, 0, 1])

                    R = np.array([[unit_X.dot(ecef_X), unit_X.dot(ecef_Y), unit_X.dot(ecef_Z)],
                                  [unit_Y.dot(ecef_X), unit_Y.dot(ecef_Y), unit_Y.dot(ecef_Z)],
                                  [unit_Z.dot(ecef_X), unit_Z.dot(ecef_Y), unit_Z.dot(ecef_Z)]
                                  ])

                    rotOPK = np.transpose(R).dot(rotRPY)
                    # then save all the elements to dataframe eph
                    eph.at[idx, 'r11'] = rotOPK[0, 0]
                    eph.at[idx, 'r12'] = rotOPK[0, 1]
                    eph.at[idx, 'r13'] = rotOPK[0, 2]
                    eph.at[idx, 'r21'] = rotOPK[1, 0]
                    eph.at[idx, 'r22'] = rotOPK[1, 1]
                    eph.at[idx, 'r23'] = rotOPK[1, 2]
                    eph.at[idx, 'r31'] = rotOPK[2, 0]
                    eph.at[idx, 'r32'] = rotOPK[2, 1]
                    eph.at[idx, 'r33'] = rotOPK[2, 2]

        return eph  # type(eph) gives <class 'pandas.core.frame.DataFrame'>


if __name__ == '__main__':
    DEM = DEM('C:/Users/ChihYu/Desktop/ToNCKU_imagedata/台中.asc')
    Tif = Tif('D:/shortcut/pleiades2017_ms_twd97-2.tif')
    # 'D:/shortcut/FS5_G010_MS_L4TWD97_20191108_030233.tif'

    Eph = Eph(
        'C:/Users/ChihYu/Desktop/ToNCKU_imagedata/FS5_20191108/MS_L1A/FS5_G010_MS_L1A_20191108_030233'
        '/FS5_G010_MS_L1A_20191108_030233.dim')

    dem = DEM.loadDem()
    tif = Tif.loadTif()
    eph = Eph.loadEph()
    # print(eph)

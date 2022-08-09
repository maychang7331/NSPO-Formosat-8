import numpy as np
from numpy import floor
import imageio
import os
from datetime import datetime
import pandas as pd
import sys
from math import degrees
import time


def rot(angle_in_rad, axis):  # input radian
    # 右手坐標系，逆時針為正
    a = float(angle_in_rad)
    if a > 2 * np.pi:
        raise Exception('Make sure to input your angle in radian (0~2*pi), not in degree !')

    if axis == 1:
        R = np.array([[1, 0, 0],
                      [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a), np.cos(a)]])

    elif axis == 2:
        R = np.array([[np.cos(a), 0, np.sin(a)],
                      [0, 1, 0],
                      [-np.sin(a), 0, np.cos(a)]])

    elif axis == 3:
        R = np.array([[np.cos(a), -np.sin(a), 0],
                      [np.sin(a), np.cos(a), 0],
                      [0, 0, 1]])

    else:
        raise Exception('Invalid input axis ! X = 1, Y = 2, Z = 3')
    # print(R)
    return R


def get_rc(x, y, className):
    xlucorner = className.__dict__['xlucorner']
    ylucorner = className.__dict__['ylucorner']
    numcol = className.__dict__['cols']
    numrow = className.__dict__['rows']
    cellsize = className.__dict__['cellsize']

    x_min = xlucorner
    y_max = ylucorner
    x_max = x_min + numcol * cellsize
    y_min = y_max - numrow * cellsize

    if not (x_min <= x <= x_max) or not (y_min <= y <= y_max):
        raise ValueError('(x, y) not within the rectangle')

    # calculate row, column bound indices from point
    row_from_bottom = floor((y - y_min) / cellsize)
    row = int(numrow - row_from_bottom)
    col = int(floor((x - x_min) / cellsize))
    return row, col


def get_rc_Mat(X_Mat, Y_Mat, className):
    """Note that the inputs are the center coordinate of each pixel"""

    xlucorner = className.__dict__['xlucorner']
    ylucorner = className.__dict__['ylucorner']
    numcol = className.__dict__['cols']
    numrow = className.__dict__['rows']
    cellsize = className.__dict__['cellsize']

    # corner
    x_min = xlucorner
    y_max = ylucorner
    x_max = x_min + numcol * cellsize
    y_min = y_max - numrow * cellsize

    statement = all([(x_min < X_Mat).all(), (X_Mat < x_max).all(), (y_min < Y_Mat).all(), (Y_Mat < y_max).all()])
    # 就會產生False如果不在範圍裡
    # numpy.matrix.all() checks whether all matrix elements along a given axis evaluate to True
    # all() checks whether all values in a list interpret to True, if at least one of them is False, it'll return False
    if statement is False:
        raise ValueError('(x, y) not within the rectangle')

    # calculate row, column bound indices from point
    row_from_bottom = floor(((Y_Mat - y_min) / cellsize).ravel())  # 先用ravel() 變成 1D array 才能取floor
    row = numrow - 1 - row_from_bottom
    col = floor(((X_Mat - x_min) / cellsize).ravel())
    row = row.reshape(Y_Mat.shape).astype(int)  # 在reshape回CCD大小
    col = col.reshape(X_Mat.shape).astype(int)
    return row, col


def get_datum(row, col, dem):
    if np.isnan(dem[row, col]):
        dem[row, col] = -9999.0
    return dem[row, col]


def get_datumn_Mat(row_Mat, col_Mat, dem):
    Z = dem[row_Mat, col_Mat]
    np.where(Z == np.nan, -9999.0, Z)
    return Z


def getlowestdatumn(target_r_dem, target_c_dem, halfGD_h, halfGD_w, classDEM):
    cellsize = classDEM.__dict__['cellsize']
    halfDEMpixel_r = floor(halfGD_h / cellsize)
    halfDEMpixel_c = floor(halfGD_w / cellsize)
    # print(halfDEMpixel_r, halfDEMpixel_c)
    r_vector = np.arange(target_r_dem - halfDEMpixel_r, target_r_dem + halfDEMpixel_r)
    c_vector = np.arange(target_c_dem - halfDEMpixel_c, target_c_dem + halfDEMpixel_r)
    r = np.broadcast_to(r_vector[:, None], (int(halfDEMpixel_r * 2), int(halfDEMpixel_c * 2))).astype(int)
    c = np.broadcast_to(c_vector, (int(halfDEMpixel_r * 2), int(halfDEMpixel_c * 2))).astype(int)
    dem = classDEM.loadDem()
    lowestdatumn = np.nanmin(dem[r, c])
    return lowestdatumn


def get_color(row, col, tif):
    B = int(tif[row, col, 0])
    G = int(tif[row, col, 1])
    R = int(tif[row, col, 2])
    return B, G, R


def get_color_Mat(row_Mat, col_Mat, tif):
    return (tif[row_Mat, col_Mat]).astype(int)


def calculateOPK(satpos, satpos_next, target):
    # 衛星轉到target
    X_differ = satpos[0] - target[0]
    Y_differ = satpos[1] - target[1]
    Z_differ = satpos[2] - target[2]

    Z_direct = np.array([X_differ, Y_differ, Z_differ])
    unit_Z = Z_direct / np.linalg.norm(Z_direct)

    Y_Up = - unit_Z[1] / unit_Z[2]  # 和Z軸相互垂直所得出來的關係
    dir_ratio = (satpos[1] - satpos_next[1]) / (satpos[0] - satpos_next[0])  # 飛行方向在TM2 座標下的關係
    Y_direct = np.array([1, dir_ratio, Y_Up])
    unit_Y = Y_direct / np.linalg.norm(Y_direct)

    X_direct = np.cross(unit_Y, unit_Z)
    unit_X = X_direct / np.linalg.norm(X_direct)

    TM2_E = np.array([1, 0, 0])
    TM2_N = np.array([0, 1, 0])
    TM2_U = np.array([0, 0, 1])

    R = np.array([[unit_X.dot(TM2_E), unit_X.dot(TM2_N), unit_X.dot(TM2_U)],
                  [unit_Y.dot(TM2_E), unit_Y.dot(TM2_N), unit_Y.dot(TM2_U)],
                  [unit_Z.dot(TM2_E), unit_Z.dot(TM2_N), unit_Z.dot(TM2_U)]
                  ])
    return R


def camera(satpos, satpos_next, target):
    # 衛星轉到target
    X_differ = satpos[0] - target[0]
    Y_differ = satpos[1] - target[1]
    Z_differ = satpos[2] - target[2]

    Z_direct = np.array([X_differ, Y_differ, Z_differ])
    unit_Z = Z_direct / np.linalg.norm(Z_direct)

    # dir_ratio = (satpos[1] - satpos_next[1]) / (satpos[0] - satpos_next[0])  # 飛行方向在TM2座標的 E and N 關係
    # = np.array([0, 1, 0])
    Y_Up = -(unit_Z[0] * (satpos[0] - satpos_next[0]) + unit_Z[1] * (satpos[1] - satpos_next[1])) / unit_Z[2]     # 和Z軸相互垂直所得出來的關係 ZE*YE + ZN*YN + ZU*YU = 0
    # - unit_Z[1] / unit_Z[2]
    Y_direct = np.array([(satpos[0] - satpos_next[0]), (satpos[1] - satpos_next[1]), Y_Up])
    unit_Y = Y_direct / np.linalg.norm(Y_direct)

    X_direct = np.cross(unit_Y, unit_Z)
    unit_X = X_direct / np.linalg.norm(X_direct)

    TM2_E = np.array([1, 0, 0])
    TM2_N = np.array([0, 1, 0])
    TM2_U = np.array([0, 0, 1])

    R = np.array([[unit_X.dot(TM2_E), unit_X.dot(TM2_N), unit_X.dot(TM2_U)],
                  [unit_Y.dot(TM2_E), unit_Y.dot(TM2_N), unit_Y.dot(TM2_U)],
                  [unit_Z.dot(TM2_E), unit_Z.dot(TM2_N), unit_Z.dot(TM2_U)]
                  ])
    return R


def addGaussianNoise(sigma_O=0, sigma_P=0, sigma_K=0):  # 設定sigma default = 0, 產生出來的 noise 就都是0，即沒有加誤差
    """回傳的值視為degree"""
    mu = 0  # 平均值
    noise_O = np.random.normal(mu, sigma_O)
    noise_P = np.random.normal(mu, sigma_P)
    noise_K = np.random.normal(mu, sigma_K)
    return noise_O, noise_P, noise_K


def frameinterpolation(eph, framespersec):
    sattime0 = datetime.strptime(eph.at[0, 'time'], "%Y-%m-%d %H:%M:%S.%f")
    sattime1 = datetime.strptime(eph.at[1, 'time'], "%Y-%m-%d %H:%M:%S.%f")
    eph_timeinterval = (sattime1 - sattime0).total_seconds()  # 每間隔0.25有一個星曆
    eph_frequency = 1 / eph_timeinterval  # satpos per seconds      # 所以每1秒有 4 (=1/0.25) 個星曆
    num_of_interpolation = int(framespersec / eph_frequency) - 1  # 若希望每1秒有12筆，則需內插 2 筆 (12/4 -1 = 2)

    # 建立有2列的dataframe s，並設值為nan  np.arange(2) = 0、1
    s = pd.DataFrame(index=[np.arange(num_of_interpolation)], columns=eph.columns)

    # 以index為group的單位，才可以在每一列後面內插s
    new_eph = pd.concat([
        pd.concat([grp, s], axis=0, ignore_index=True) for key, grp in eph.groupby(eph.index)]).reset_index(drop=True)

    # 把加在最後一列後面的s 拿掉
    new_eph.drop(new_eph.tail(num_of_interpolation).index, inplace=True)  # drop last n rows

    # 轉換資料型態，以利進行內插
    for col in new_eph:
        if col == 'time':
            new_eph['time'] = pd.to_datetime(new_eph['time'])  # new_eph['time'].dtypes gives datetime64[ns]
        else:
            new_eph[col] = pd.to_numeric(new_eph[col], errors='coerce')  # new_eph['TM2_X'].dtypes gives float64

    # 時間的內插
    # Cast date to seconds (also recast the NaT to Nan)，並save在新增的column 'seconds'
    # 但因為timetuple() 會把小數點去掉，所以用t.microsecond / 1000000.0 加回去
    new_eph['seconds'] = [time.mktime(t.timetuple()) + (t.microsecond / 1000000.0) if t is not pd.NaT else float('nan')
                          for t in new_eph['time']]
    # 內插
    new_eph['interpolated'] = new_eph['seconds'].interpolate()
    # 存回去
    new_eph['time'] = [datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f") for t in new_eph['interpolated']]

    # 位置的內插
    new_eph['TM2_X'] = new_eph['TM2_X'].interpolate(method='linear', limit_direction='forward', axis=0)
    new_eph['TM2_Y'] = new_eph['TM2_Y'].interpolate(method='linear', limit_direction='forward', axis=0)
    new_eph['sat_h'] = new_eph['sat_h'].interpolate(method='linear', limit_direction='forward', axis=0)

    # 拿掉'seconds'和'interpolated'
    new_eph = new_eph.drop(['seconds', 'interpolated'], 1)  # where 1 is the axis number (0 for rows and 1 for columns.)

    # 把內插得到的TM2_X, TM2_Y 換算回Lat, Lon 並填回 new_eph
    from coordinateSystem import CoordinateSystem
    cs = CoordinateSystem()

    for idx, row in new_eph.iterrows():
        if np.isnan(row['Lat']) or np.isnan(row['Lon']):
            new_eph.at[idx, 'Lat'], new_eph.at[idx, 'Lon'] = cs.TWD97TM2_To_LatLon(row['TM2_X'], row['TM2_Y'])

    return new_eph, num_of_interpolation


def image2gif(outputdir, fps):
    frame_array = []
    filenamelist = [f for f in os.listdir(path=outputdir) if f.endswith('.png')]  # ['1.png', '10.png', '11.png'....]
    writer = imageio.get_writer(outputdir + '/simulation.mp4', fps=fps)
    for i in range(len(filenamelist)):
        writer.append_data(imageio.imread(outputdir + '/' + str(i) + '.png'))
    writer.close()
    sys.stdout.write('\r' + 'process done... \n file saved in the output directory\n')

# if __name__ == '__main__':
#     from loadData import DEM, Tif, Eph
#
#     DEM = DEM('C:/Users/Chih Yu/Desktop/ToNCKU_imagedata/台中.asc')
#     Tif = Tif('C:/Users/Chih Yu/Desktop/ToNCKU_imagedata/FS5_20191108/MS_L4/FS5_G010_MS_L4TWD97_20191108_030233'
#               '/FS5_G010_MS_L4TWD97_20191108_030233.tif')
#     Eph = Eph(
#         'C:/Users/ChihYu/Desktop/ToNCKU_imagedata/FS5_20191108/MS_L1A/FS5_G010_MS_L1A_20191108_030233'
#         '/FS5_G010_MS_L1A_20191108_030233.dim')

# dem = DEM.loadDem()  # 須先loadDEM才會更新init的參數
# r, c = get_rc(210000, 2680000, DEM)
# # print(r, c)
#
# tif = Tif.loadTif()
# # print(get_color(r, c, tif))
#
# dem = DEM.loadDem()
# # print(get_datum(r, c, dem))
#
# R = rot(np.pi / 2, 1)
# # print(R)

# image2gif('C:/Users/ChihYu/Desktop/harbor48/', 12)
# frameinterpolation(Eph.loadEph(), 12)

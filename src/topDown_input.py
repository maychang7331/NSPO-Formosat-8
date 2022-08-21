import time
import numpy as np
# np.show_config()
import warnings
import matplotlib.pyplot as plt
from math import atan2, sqrt, degrees, radians
from utilities import rot, get_rc_Mat, get_datumn_Mat, get_color_Mat, calculateOPK, addGaussianNoise, \
    frameinterpolation, get_rc, getlowestdatumn, get_datum
from coordinateSystem import CoordinateSystem

cs = CoordinateSystem()


class TopDown:
    # initialize
    pixelSize = 0
    x0 = 0
    y0 = 0
    f = 0

    def __init__(self, camera_specification, io, DEM, Tif, Eph):
        """ Note that all the parameters should be in meters in advance
        """
        self.pixelSize = camera_specification[0]  # 4.5 um 換成 以 m 為單位
        self.sensorSize_h = camera_specification[1]
        self.sensorSize_w = camera_specification[2]

        self.x0 = io[0]  # 相機內方位參數(需以公尺為單位!!!)
        self.y0 = io[1]
        self.f = io[2]  # mm 換算成 m

        self.DEM = DEM  # class
        self.Tif = Tif  # class
        self.Eph = Eph  # class

    # bottom Up沒用到
    def bottomUp(self, satpos, target, R):
        XA = satpos[0]
        YA = satpos[1]
        ZA = satpos[2]
        sub_X = target[0] - XA
        sub_Y = target[1] - YA
        sub_Z = target[2] - ZA
        denominator = (R[2, 0] * sub_X + R[2, 1] * sub_Y + R[2, 2] * sub_Z)
        x = self.x0 + self.f * (R[0, 0] * sub_X + R[0, 1] * sub_Y + R[0, 2] * sub_Z) / denominator
        y = self.y0 + self.f * (R[1, 0] * sub_X + R[1, 1] * sub_Y + R[1, 2] * sub_Z) / denominator
        return x, y

    def topDown(self, satpos, R):

        XA = satpos[0]
        YA = satpos[1]
        ZA = satpos[2]

        """vectorize all parameters in order to speed up"""
        # 將每個CCD的r,c座標換算成x,y 並存到矩陣x、y
        h_vector = np.arange(self.sensorSize_h)
        w_vector = np.arange(self.sensorSize_w)
        h = np.broadcast_to(h_vector[:, None], (self.sensorSize_h, self.sensorSize_w))
        w = np.broadcast_to(w_vector, (self.sensorSize_h, self.sensorSize_w))

        x = (0.5 + w - self.sensorSize_w / 2) * self.pixelSize  # center of each col
        y = (self.sensorSize_h / 2 - h - 0.5) * self.pixelSize  # center of each row

        # 共線式
        sub_x = x - self.x0
        sub_y = y - self.y0
        denominator = (R[0, 2] * sub_x + R[1, 2] * sub_y + R[2, 2] * (-self.f))
        X_cal = (R[0, 0] * sub_x + R[1, 0] * sub_y + R[2, 0] * (-self.f)) / denominator
        Y_cal = (R[0, 1] * sub_x + R[1, 1] * sub_y + R[2, 1] * (-self.f)) / denominator

        # given initial height value as average elevation of Dem in vectorized form
        Z = np.empty([self.sensorSize_h, self.sensorSize_w])
        Z.fill(self.DEM.__dict__['avg'])

        tmp_rc = np.empty([2, self.sensorSize_h, self.sensorSize_w])
        tmp_rc.fill(-1)  # print(tmp_rc)

        # 設置迭代初始值
        tmp_r_differ = self.sensorSize_h * self.sensorSize_w + 1  # 假設 前一次r 和 後一次r 不一樣的個數 遠大於pixel個數(不可能)
        tmp_c_differ = self.sensorSize_h * self.sensorSize_w + 1

        count = 0
        while True:
            count += 1              # 迭代次數

            sub_Z = Z - ZA
            X = XA + sub_Z * X_cal  # print(X[0, 0], Y[0, 0])
            Y = YA + sub_Z * Y_cal  # print(X[5119, 5119], Y[5119, 5119])

            r_in_dem, c_in_dem = get_rc_Mat(X, Y, self.DEM)

            """
            compare whether former and next r matrix are equal, return true false matrix
            than calculate the amount of pixel that differs, e.g. which returns false
            num of false(0) = size of whole matrix - num of true(1)
            """
            compare_r = (r_in_dem == tmp_rc[0])  # 比較前後次迭代出來的r,c矩陣是否相同, 回傳值為true false 矩陣
            compare_c = (c_in_dem == tmp_rc[1])
            r_differ = np.size(compare_r) - np.count_nonzero(compare_r)  # 計算 值不同的個數(false) 的個數
            c_differ = np.size(compare_c) - np.count_nonzero(compare_c)

            # 設定迭代結束條件：　(1) 大於10圈  (2) 後一次 row 和 col false 個數總和 > 前次
            # or (r_differ == tmp_r_differ and c_differ == tmp_c_differ):
            if count >= 10 or ((r_differ + c_differ) >= (tmp_r_differ + tmp_c_differ)):
                # print(X[0, 0], Y[0, 0])               # 檢驗影像大小是否合理
                # print(X[5119, 5119], Y[5119, 5119])

                if count == 10:
                    warnings.warn(" Iteration over 10 loops!")
                Max_diff_row = (r_in_dem - tmp_rc[0]).max()
                Max_diff_col = (c_in_dem - tmp_rc[1]).max()

                r_in_tif, c_in_tif = get_rc_Mat(X, Y, self.Tif)     # 取得經迭代後後之X、Y座標在tif中對應的r,c座標
                break

            Z = get_datumn_Mat(r_in_dem, c_in_dem, self.DEM.loadDem())

            # update
            tmp_rc[0] = r_in_dem
            tmp_rc[1] = c_in_dem

            tmp_r_differ = r_differ
            tmp_c_differ = c_differ

        return r_in_tif, c_in_tif, count, Max_diff_row, Max_diff_col

    # Code starts from here !!!
    def colinearityEquation(self, fps, outputdir, targetLat, targetLon, sigma_O, sigma_P, sigma_K):
        dem = self.DEM.loadDem()  # 須要先load資料參數才會更新，不然都會是初始值
        tif = self.Tif.loadTif()
        eph = self.Eph.loadEph()

        """
        若要在runtime 指定衛星起始位置、target point，請註解這段，並將後面 satpos[0] -= 383.9800、satpos[1] -= 1655.0 一並註解讓衛星移動
        """
        # satLat, satLon = input("Input satellite position in degree (lat, lon): ").split(',')
        # XA, YA = cs.LatLon_To_TWD97TM2(satLat, satLon)
        # ZA = 725000-22               # eph.at[i, 'sat_h']      # change 橢球高 to TM2 的高
        # satpos = [XA, YA, ZA]
        # targetLat, targetLon = input("Input target position in degree (lat, lon): ").split(',')

        """
        若希望能夠讓高程較高的地方位移的情況比平地明顯，可透過減去取向範圍之最低高程(註解下面)辦到。(但大肚區高差不到，影響不明顯)
        """
        # calculate GSD in order to get the minimum datumn of the frame
        # GSD = satpos[2] * self.pixelSize / self.f
        # halfGD_w = GSD * self.sensorSize_w / 2  # 半張影像的寬(m)
        # halfGD_h = GSD * self.sensorSize_h / 2  # 半張影像的高(m)
        # lowestdatumn = getlowestdatumn(target_r_dem, target_c_dem, halfGD_h, halfGD_w, self.DEM)

        targetX, targetY = cs.LatLon_To_TWD97TM2(targetLat, targetLon)
        target_r_dem, target_c_dem = get_rc(targetX, targetY, self.DEM)     # target point 在DEM的row、col
        targetZ = get_datum(target_r_dem, target_c_dem, dem)                # target point 的高程值
        target = [targetX, targetY, targetZ]

        # 內插衛星位置，並回傳新的星曆，及需內插的個數
        new_eph, num_of_interpolation = frameinterpolation(eph, fps)

        # 先建一個空list用來存放資料，以便之後存入dataFrame
        omegaList_deg = []
        phiList_deg = []
        kappaList_deg = []
        numOfIterList = []
        MaxDifferList_row = []
        MaxDifferList_column = []
        TimeSpentList = []

        # 為了要拿到這個row 和下一個row所以加下面兩行在for迴圈之前
        # row_iterator = new_eph.iterrows()
        # _, last = next(row_iterator)  # take idx=1 from row_iterator

        for idx, row in new_eph.iterrows():
            tic = time.time()

            satpos = [row['TM2_X'], row['TM2_Y'], row['sat_h']]  # 給定星曆中的衛星位置

            # 最後一筆衛星沒有下一筆星曆可以算行進方向，所以用前一筆
            if idx == (len(new_eph) - 1):
                satpos_next = - (np.array([new_eph.loc[idx-1, 'TM2_X'], new_eph.loc[idx-1, 'TM2_Y'], new_eph.loc[idx-1, 'sat_h']]) - satpos) + satpos
            else:
                satpos_next = [new_eph.loc[idx+1, 'TM2_X'], new_eph.loc[idx+1, 'TM2_Y'], new_eph.loc[idx+1, 'sat_h']]

            R = calculateOPK(satpos, satpos_next, target)  # 計算衛星位置與target的旋轉矩陣(衛星轉至TM2)
            noise_O, noise_P, noise_K = addGaussianNoise(sigma_O, sigma_P, sigma_K)     # 指定1 sigma值，以加入高斯常態分佈雜訊

            omega = atan2(-R[2, 1], R[2, 2]) + radians(noise_O)                         # 由旋轉矩陣算出旋轉角omega、phi、kappa
            phi = atan2(R[2, 0], sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)) + radians(noise_P)  # 單位：radians
            kappa = atan2(-R[1, 0], R[0, 0]) + radians(noise_K)

            # 再換回 rotation matrix R (不知道為甚麼要加負號才會跟上面算出來的R較一致@@)
            R = np.dot(rot(-kappa, axis=3), np.dot(rot(-phi, axis=2), rot(-omega, axis=1)))
            r_in_tif, c_in_tif, count, Max_diff_row, Max_diff_col = self.topDown(satpos, R)     # 執行topDown

            img = np.empty([self.sensorSize_h, self.sensorSize_w, 3], dtype=int)        # 開一個空的矩陣以儲存影像
            img = get_color_Mat(r_in_tif, c_in_tif, tif)                                # 順序為rgb
            plt.imsave(outputdir + '/' + str(idx) + '.png', img.astype('uint8'))        # 儲存影像
            # plt.imshow(img)
            # plt.show()

            toc = time.time()

            # 輸出所有相關資料到metadata.txt
            omegaList_deg.append(degrees(omega))
            phiList_deg.append(degrees(phi))
            kappaList_deg.append(degrees(kappa))
            numOfIterList.append(count)
            MaxDifferList_row.append(Max_diff_row)
            MaxDifferList_column.append(Max_diff_col)
            TimeSpentList.append(str((toc - tic)))

            if idx % (num_of_interpolation+1) != 0:  # 印出原本有的eph就好，內插出來的則印interpolating...
                print('interpolating...')
            else:
                print("\nsat pos", str(idx//(num_of_interpolation+1)))
                print("{:.2f}".format(new_eph.at[idx, 'Lat']),
                      "{:.2f}".format(new_eph.at[idx, 'Lon']), "{:.2f}".format(new_eph.at[idx, 'sat_h']))
                print("omega：{:.2f}".format(degrees(omega)), "deg")
                print("phi：{:.2f}".format(degrees(phi)), "deg")
                print("kappa：{:.2f}".format(degrees(kappa)), "deg")
                print("Iter：", count)
                print("Maximum difference in row：", Max_diff_row, "pixels")
                print("Maximum difference in col：", Max_diff_col, "pixels")
                print("Time Spent：{:.2f}".format(toc - tic), "sec", end='\n\n')

            # satpos[0] -= 383.9800
            # satpos[1] -= 1655.0
            last = row

        new_eph['omega'] = omegaList_deg
        new_eph['phi'] = phiList_deg
        new_eph['kappa'] = kappaList_deg
        new_eph['num of Iter'] = numOfIterList
        new_eph['Max difference in row'] = MaxDifferList_row
        new_eph['Max difference in col'] = MaxDifferList_column
        new_eph['TimeSpent'] = TimeSpentList

        # 指print出header的這些column, 其他不輸出
        header = ['time', 'Lat', 'Lon', 'sat_h', 'TM2_X', 'TM2_Y', 'omega', 'phi', 'kappa', 'num of Iter',
                  'Max difference in row', 'Max difference in col', 'TimeSpent']
        new_eph.to_csv(outputdir + '/metadata.txt', sep='\t', columns=header)


if __name__ == '__main__':
    from loadData import DEM, Tif, Eph

    DEM = DEM('C:/Users/ChihYu/Desktop/ToNCKU_imagedata/台中.asc')
    Tif = Tif(
        'D:/shortcut/pleiades2017_ms_twd97-2.tif')
    # 'D:/shortcut/FS5_G010_MS_L4TWD97_20191108_030233.tif')

    Eph = Eph(
        'C:/Users/ChihYu/Desktop/ToNCKU_imagedata/FS5_20191108/MS_L1A/FS5_G010_MS_L1A_20191108_030233'
        '/FS5_G010_MS_L1A_20191108_030233.dim')

    t = TopDown([4.5 * 10 ** (-6), 2560, 2560], [0, 0, 3927 * 10 ** (-3)], DEM, Tif, Eph)
    t.colinearityEquation(fps=12, outputdir='D:/pycharm/code/test', targetLat=24.17972, targetLon=120.60333,
                          sigma_O=0, sigma_P=0, sigma_K=0)

    # satellite：
    # 1:      25.50138, 121.16944
    # 13:　　 23.7080853, 120.7151677
    # 30:    21.1649357, 120.0917452
    
    # target：
    # 最多tie point：24.17972, 120.60333

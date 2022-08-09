import numpy as np
from math import tan, atan, atan2, sin, cos, radians, degrees, pi, sqrt
from decimal import Decimal


class CoordinateSystem():
    """This object provide method for converting lat/lon coordinate to TWD97
    coordinate

    the formula reference to
    http://www.uwgb.edu/dutchs/UsefulData/UTMFormulas.htm (there is lots of typo)
    http://www.offshorediver.com/software/utm/Converting UTM to Latitude and Longitude.doc

    Parameters reference to
    http://rskl.geog.ntu.edu.tw/team/gis/doc/ArcGIS/WGS84%20and%20TM2.htm
    http://blog.minstrel.idv.tw/2004/06/taiwan-datum-parameter.html
    """

    def __init__(self):
        # Equatorial radius 赤道半徑 (meters)   # 長軸半徑
        self.a = 6378137.0
        # Polar radius 兩極半徑 (meters)        # 短軸半徑
        self.b = 6356752.314245
        # central meridian of zone 中央經線
        self.long0 = radians(121)
        # scale along long0 延著long0的縮放比例
        self.k0 = 0.9999
        # delta x in meter 座標平移量 (meters)
        self.dx = 250000
        # delta y in meter 無座標平移(即起點為赤道)
        self.dy = 0

    def LatLon_To_TWD97TM2(self, lat_deg, lon_deg):
        """Convert lat lon to twd97 tm2
        input degrees
        calculate with radius
        """
        lat = radians(Decimal(lat_deg))
        lon = radians(Decimal(lon_deg))

        a = self.a
        b = self.b
        long0 = self.long0
        k0 = self.k0
        dx = self.dx

        e = (1 - b ** 2 / a ** 2) ** 0.5
        e2 = e ** 2 / (1 - e ** 2)
        n = (a - b) / (a + b)
        nu = a / (1 - (e ** 2) * (sin(lat) ** 2)) ** 0.5
        p = lon - long0

        A = a * (1 - n + (5 / 4.0) * (n ** 2 - n ** 3) + (81 / 64.0) * (n ** 4 - n ** 5))
        B = (3 * a * n / 2.0) * (1 - n + (7 / 8.0) * (n ** 2 - n ** 3) + (55 / 64.0) * (n ** 4 - n ** 5))
        C = (15 * a * (n ** 2) / 16.0) * (1 - n + (3 / 4.0) * (n ** 2 - n ** 3))
        D = (35 * a * (n ** 3) / 48.0) * (1 - n + (11 / 16.0) * (n ** 2 - n ** 3))
        E = (315 * a * (n ** 4) / 51.0) * (1 - n)

        S = A * lat - B * sin(2 * lat) + C * sin(4 * lat) - D * sin(6 * lat) + E * sin(8 * lat)

        # 計算Y值
        K1 = S * k0
        K2 = k0 * nu * sin(2 * lat) / 4.0
        K3 = (k0 * nu * sin(lat) * (cos(lat) ** 3) / 24.0) * (
                5 - tan(lat) ** 2 + 9 * e2 * (cos(lat) ** 2) + 4 * (e2 ** 2) * (cos(lat) ** 4))
        y = K1 + K2 * (p ** 2) + K3 * (p ** 4)

        # 計算X值
        K4 = k0 * nu * cos(lat)
        K5 = (k0 * nu * (cos(lat) ** 3) / 6.0) * (1 - tan(lat) ** 2 + e2 * (cos(lat) ** 2))

        x = K4 * p + K5 * (p ** 3) + self.dx
        return x, y

    def TWD97TM2_To_LatLon(self, TM2_X, TM2_Y):
        a = self.a
        b = self.b
        long0 = self.long0
        k0 = self.k0
        TM2_X -= self.dx
        TM2_Y -= self.dy

        e = (1 - b ** 2 / a ** 2) ** 0.5
        # Calculate the Meridional Arc
        M = TM2_Y / k0

        # Calculate Footprint Latitude
        mu = M / (a * (1.0 - (e ** 2) / 4.0 - 3 * (e ** 4) / 64.0 - 5 * (e ** 6) / 256.0))
        e1 = (1.0 - ((1.0 - (e ** 2)) ** 0.5)) / (1.0 + ((1.0 - (e ** 2)) ** 0.5))

        J1 = 3 * e1 / 2 - 27 * (e1 ** 3) / 32.0
        J2 = 21 * (e1 ** 2) / 16 - 55 * (e1 ** 4) / 32.0
        J3 = 151 * (e1 ** 3) / 96.0
        J4 = 1097 * (e1 ** 4) / 512.0

        fp = mu + J1 * sin(2 * mu) + J2 * sin(4 * mu) + J3 * sin(6 * mu) + J4 * sin(8 * mu)

        # Calculate Latitude and Longitude
        e2 = (e * a / b) ** 2
        C1 = (e2 * cos(fp)) ** 2
        T1 = tan(fp) ** 2
        R1 = a * (1 - (e ** 2)) / ((1 - (e ** 2) * (sin(fp) ** 2)) ** (3.0 / 2.0))
        N1 = a / ((1 - (e ** 2) * (sin(fp) ** 2)) ** 0.5)

        D = TM2_X / (N1 * k0)

        # 計算緯度
        Q1 = N1 * tan(fp) / R1
        Q2 = (D ** 2) / 2.0
        Q3 = (5 + 3 * T1 + 10 * C1 - 4 * (C1 ** 2) - 9 * e2) * (D ** 4) / 24.0
        Q4 = (61 + 90 * T1 + 298 * C1 + 45 * (T1 ** 2) - 3 * (C1 ** 2) - 252 * e2) * (D ** 6) / 720.0
        lat = degrees(fp - Q1 * (Q2 - Q3 + Q4))

        # 計算經度
        Q5 = D
        Q6 = (1 + 2 * T1 + C1) * (D ** 3) / 6
        Q7 = (5 - 2 * C1 + 28 * T1 - 3 * (C1 ** 2) + 8 * e2 + 24 * (T1 ** 2)) * (D ** 5) / 120.0
        lon = degrees(long0 + (Q5 - Q6 + Q7) / cos(fp))

        return lat, lon

    def ecef_to_llh(self, ecef_km):
        # WGS-84 ellipsoid parameters
        a = self.a
        b = self.b

        p = sqrt(ecef_km[0] ** 2 + ecef_km[1] ** 2)
        thet = atan(ecef_km[2] * a / (p * b))
        esq = 1.0 - (b / a) ** 2
        epsq = (a / b) ** 2 - 1.0

        lat = atan((ecef_km[2] + epsq * b * sin(thet) ** 3) / (p - esq * a * cos(thet) ** 3))
        lon = atan2(ecef_km[1], ecef_km[0])
        n = a * a / sqrt(a * a * cos(lat) ** 2 + b ** 2 * sin(lat) ** 2)
        h = p / cos(lat) - n

        lat = degrees(lat)
        lon = degrees(lon)
        return lat, lon, h

    def llh_to_ecef(self, lat_deg, lon_deg, height):
        a = self.a
        flattening = 1 / 298.257223563
        NAV_E2 = (2 - flattening) * flattening  # also e**2

        slat = sin(radians(lat_deg))
        clat = cos(radians(lat_deg))
        r_n = a / sqrt(1 - NAV_E2 * slat * slat)
        xyz = np.array([(r_n + height) * clat * cos(radians(lon_deg)),
                        (r_n + height) * clat * sin(radians(lon_deg)),
                        (r_n * (1 - NAV_E2) + height) * slat])

        if (lat_deg < -90.0) or (lat_deg > +90.0) or (lon_deg < -180.0) or (lon_deg > +360.0):
            raise ValueError('WGS lat or lon out of range')
        return xyz

    def origin_of_TM2(self):
        # ori_lon = radians(120.982)
        origin_lat_rad = radians(23.97387)
        origin_lon_rad = radians(121)
        return origin_lat_rad, origin_lon_rad


if __name__ == '__main__':
    cs = CoordinateSystem()
    # lon, lat = input("Input lon & lat in degrees：").split(" ")
    # x, y = cs.LatLon_To_TWD97TM2(lat, lon)
    # print(x, y)
    # # verify if  (lat, lon)  ==>  (x, y)  ==>  (lat, lon) equals
    # nlat, nlon = cs.TWD97TM2_To_LatLon(x, y)
    # print(nlat, nlon)

    X, Y, Z = input("Input X, Y, Z in meters：").split(" ")
    ecef_km = [float(X) / 1000.0, float(Y) / 1000.0, float(Z) / 1000.0]
    lat, lon, h = cs.ecef_to_llh(ecef_km)
    print(lat, lon, h)
    xyz = cs.llh_to_ecef(lat, lon, h)
    print(xyz)

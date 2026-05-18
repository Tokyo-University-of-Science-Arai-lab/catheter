import math
import numpy as np


def deg_to_rad(degree_list):
    if len(degree_list) != 7:
        raise ValueError("The input list must contain exactly 6 elements.")
    
    radian_list = [math.radians(angle) for angle in degree_list]
    return radian_list


def rad_to_deg(radian_list):
    if len(radian_list) != 7:
        raise ValueError("The input list must contain exactly 6 elements.")
    
    degree_list = [math.degrees(angle) for angle in radian_list]
    return degree_list


def toBytes(str):
    return bytes(str.encode())


def rv2rpy(rx, ry, rz): # UR RPY
    theta = np.sqrt(rx*rx + ry*ry + rz*rz)
    kx = rx/theta
    ky = ry/theta
    kz = rz/theta
    cth = np.cos(theta)
    sth = np.sin(theta)
    vth = 1 - np.cos(theta)

    r11 = kx*kx*vth + cth
    r12 = kx*ky*vth - kz*sth
    r13 = kx*kz*vth + ky*sth
    r21 = kx*ky*vth + kz*sth
    r22 = ky*ky*vth + cth
    r23 = ky*kz*vth - kx*sth
    r31 = kx*kz*vth - ky*sth
    r32 = ky*kz*vth + kx*sth
    r33 = kz*kz*vth + cth

    beta = np.arctan2(-r31, np.sqrt(r11*r11 + r21*r21))

    if beta > np.deg2rad(89.99):
        beta = np.deg2rad(89.99)
        alpha = 0
        gamma = np.arctan2(r12, r22)
    elif beta < -np.deg2rad(89.99):
        beta = -np.deg2rad(89.99)
        alpha = 0
        gamma = -np.arctan2(r12, r22)
    else:
        cb = np.cos(beta)
        alpha = np.arctan2(r21 / cb, r11 / cb)
        gamma = np.arctan2(r32 / cb, r33 / cb)

    rpy = [gamma, beta, alpha]
    return rpy


def rpy2rv(roll, pitch, yaw):
    alpha = yaw
    beta = pitch
    gamma = roll

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)

    r11 = ca * cb
    r12 = ca * sb * sg - sa * cg
    r13 = ca * sb * cg + sa * sg
    r21 = sa * cb
    r22 = sa * sb * sg + ca * cg
    r23 = sa * sb * cg - ca * sg
    r31 = -sb
    r32 = cb * sg
    r33 = cb * cg

    theta = np.arccos((r11 + r22 + r33 - 1) / 2)
    sth = np.sin(theta)
    kx = (r32 - r23) / (2 * sth)
    ky = (r13 - r31) / (2 * sth)
    kz = (r21 - r12) / (2 * sth)

    rv = [theta * kx, theta * ky, theta * kz]
    return rv
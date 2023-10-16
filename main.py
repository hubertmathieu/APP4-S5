import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy import signal
from zplane import zplane
import random
import os


def to_db(data):
    return 20 * np.log10(np.abs(data))

# Correction des abérations
def correct_image(img):
    num = np.poly([0, -0.99, -0.99, 0.8])
    den = np.poly([0.95 * np.exp(1j * np.pi / 8), 0.95 * np.exp(-1j * np.pi / 8), 0.9 * np.exp(1j * np.pi / 2),
                   0.9 * np.exp(-1j * np.pi / 2)])

    # afficher p & z
    zplane(num, den)

    # image corrigée
    img_filtered = signal.lfilter(num, den, img)

    return img_filtered

def rotate_img(img):
    img_size = len(img)
    fact_x = -1
    fact_y = 1

    zero_matrix = np.zeros((img_size, img_size))
    matrice_transformation = np.array([[0, fact_x],[fact_y, 0]])


    for x in range(img_size):
        for y in range(img_size):
            point_original = np.array([x, y])
            point_transforme = np.dot(matrice_transformation, point_original)
            x_transforme, y_transforme = point_transforme
            pixel_value = img[y][x]
            zero_matrix[y_transforme][x_transforme + img_size - 1] = pixel_value

    return zero_matrix

def rotation_vect_base(img):
    img_size = len(img)

    zero_matrix = np.zeros((img_size, img_size))

    for u1 in range(img_size):
        for u2 in range(img_size):
            e1 = -u2
            e2 = u1

            zero_matrix[e2][e1] = img[u2][u1]

    return zero_matrix

def create_filter():
    fs = 1600
    wp = 500
    ws = 750
    gpass = 0.2
    gstop = 60

    ord, wn = signal.buttord(wp, ws, gpass, gstop, fs=fs)
    print('Butter')
    print('ordre: ', ord)
    print('freq à 3dB: ', wn, ' Hz')

    ord, wn = signal.cheb1ord(wp, ws, gpass, gstop, fs=fs)
    print('Cheb 1')
    print('ordre: ', ord)
    print('freq à 3dB: ', wn, ' Hz')

    ord, wn = signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)
    print('Cheb 2')
    print('ordre: ', ord)
    print('freq à 3dB: ', wn, ' Hz')

    ord, wn = signal.ellipord(wp, ws, gpass, gstop, fs=fs)
    print('ellip')
    print('ordre: ', ord)
    print('freq à 3dB: ', wn, ' Hz')

    num, den = signal.ellip(ord, gpass, gstop, wn, 'lowpass', False, fs=fs, output='ba')
    zplane(num, den)

    return num, den



def bilinear_transform(image):
    fc = 500
    fe = 1600
    wd = 2*np.pi*fc/fe
    wa = 2*fe*np.tan(wd/2)
    s = 2*fe  # without z+1/z-1

    x = 2.39
    y = 1.108
    z = 0.502

    num = [1, 2, 1]
    den = [x, y, z]

    output = signal.lfilter(num, den, image)
    plt.imshow(output, cmap='gray')
    plt.show()







if __name__ == '__main__':
    img_aberated = np.load("goldhill_aberrations.npy")

    img_filtered = correct_image(img_aberated)

    plt.imshow(img_filtered, cmap='gray')
    plt.show()

    img_rotate = img.imread('goldhill_rotate.png')
    img_gris = np.mean(img_rotate, -1)

    img_rotated = rotate_img(img_gris)
    plt.imshow(img_filtered, cmap='gray')
    plt.show()
    # img_rotated = rotation_vect_base(img_gris)
    # plt.imshow(img_filtered, cmap='gray')
    # plt.show()

    bilinear_transform(img_rotated)

    num, den = create_filter()
    freqz, response = signal.freqz(num, den)

    plt.figure()
    plt.plot(freqz, to_db(response))
    plt.title('Réponse en freq')
    plt.xlabel('freqs')
    plt.ylabel('db')
    plt.grid()
    plt.show()

    output = signal.lfilter(num, den, img_rotated)
    plt.imshow(output, cmap='gray')
    plt.show()



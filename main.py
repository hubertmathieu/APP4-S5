import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy import signal
from zplane import zplane
import random
import os

def save_figure(name):
    # Create the child folder if it doesn't exist
    child_folder = 'figures'
    if not os.path.exists(child_folder):
        os.makedirs(child_folder)

    # Specify the filename with the child folder path and format (e.g., PNG)
    file_name = os.path.join(child_folder, name + '.png')

    plt.savefig(file_name)


def to_db(data):
    return 20 * np.log10(np.abs(data))


def lowest_ord_by_type(wp, ws, gpass, gstop, fs):
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

    return ord, wn

def show_img(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    save_figure(title)
    plt.show()

def plot(x, y, title, x_label, y_label):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    save_figure(title)
    plt.show()

# Correction des abérations
def remove_aberations(img):
    num = np.poly([0, -0.99, -0.99, 0.8])
    den = np.poly([0.95 * np.exp(1j * np.pi / 8), 0.95 * np.exp(-1j * np.pi / 8), 0.9 * np.exp(1j * np.pi / 2),
                   0.9 * np.exp(-1j * np.pi / 2)])

    # afficher p & z
    zplane(den, num)

    # Nombre de lignes
    num_rows = img.shape[0]
    filtered_img = np.zeros((num_rows, num_rows))

    # image corrigée
    for row in range(num_rows):
        filtered_row = signal.lfilter(num, den, img[row])
        filtered_img[row] = filtered_row

    return filtered_img

def rotate_img(img):
    img_size = len(img)

    fact_x = -1
    fact_y = 1

    final_matrix = np.zeros((img_size, img_size))
    matrice_transformation = np.array([[0, fact_x], [fact_y, 0]])

    for x in range(img_size):
        for y in range(img_size):
            point_original = np.array([x, y])
            point_transforme = np.dot(matrice_transformation, point_original)
            x_transforme, y_transforme = point_transforme
            pixel_value = img[y][x]
            final_matrix[y_transforme][x_transforme + img_size - 1] = pixel_value

    return final_matrix

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

    ord, wn = lowest_ord_by_type(wp, ws, gpass, gstop, fs)

    num, den = signal.ellip(ord, gpass, gstop, wn, 'lowpass', False, fs=fs, output='ba')
    zplane(num, den, filename='z-p-ellip filter')

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

    zplane(np.array(num), np.array(den), filename='z-p bilinear')

    freqz, response = signal.freqz(num, den)
    plot(freqz, to_db(response), 'Module de la réponse en freq transformation bilinéaire', 'freqs normalisées', 'dB')

    img_bilinear_trans = signal.lfilter(num, den, image)

    show_img(img_bilinear_trans, title='img_bilinear_trans')

def filter_img_py_funct(img_rotated):
    num, den = create_filter()
    freqz, response = signal.freqz(num, den)

    plot(freqz, to_db(response), 'Module de la réponse en freq filtre elip', 'freqs normalisées', 'dB')

    img_elip_filtered = signal.lfilter(num, den, img_rotated)
    show_img(img_elip_filtered, title='img_elip_filtered')

    return img_elip_filtered


def compress_img(image, ratio):
    # matrice de covariance
    mat_cov = np.cov(image)

    # valeur, vecteur propres
    values, vectors = np.linalg.eig(mat_cov)

    # trier décroissant par valeurs propres
    sorted_index = np.argsort(values)[::-1]
    sorted_vectors = vectors[:, sorted_index]

    matrice_passage = np.transpose(sorted_vectors)

    # ligne à mettre à 0 (% à éliminer)
    img_v = matrice_passage.dot(image)
    size_img_v = len(img_v)
    line_at_zero = int(ratio * size_img_v)

    for i in range(line_at_zero):
        img_v[size_img_v - i - 1] = 0

    matrice_passage_inv = np.linalg.inv(matrice_passage)
    img_o = matrice_passage_inv.dot(img_v)
    show_img(img_o, title='img_compressed' + str(ratio))

def correct_aberations(img_aberated):
    #img_aberated = np.load("goldhill_aberrations.npy")
    show_img(img_aberated, title='img_aberated')

    img_without_aberations = remove_aberations(img_aberated)

    show_img(img_without_aberations, title='img-without-aberations')

    return img_without_aberations

def rotations(img_to_rotate):
    #img_to_rotate = np.mean(img.imread('goldhill_rotate.png'), -1)
    show_img(img_to_rotate, title='img_to_rotate')

    img_rotated = rotate_img(img_to_rotate)
    show_img(img_rotated, title='img_rotated')
    # img_rotated = rotation_vect_base(img_gris)
    # show_img(img_rotated, title='img_rotated')

    return img_rotated

def filter_img(img_rotated):
    bilinear_transform(img_rotated)

    return filter_img_py_funct(img_rotated)

if __name__ == '__main__':
    img_complete = np.load("image_complete.npy")
    img_without_aberations = correct_aberations(img_complete)

    img_rotated = rotations(img_without_aberations)

    filtered_img = filter_img(img_rotated)

    compress_img(filtered_img, 0.70)
    compress_img(filtered_img, 0.5)




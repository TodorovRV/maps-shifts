import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
import sys
sys.path.insert(0, '/home/rtodorov/jetpol/ve/vlbi_errors')
from spydiff import modelfit_core_wo_extending


def get_core(img, noise):
    core = np.argmax(img)
    y_max, x_max = np.unravel_index(core, img.shape)
    dx = 0
    dy = 0
    while (img[y_max + dy][x_max + dx] > noise and
           img[y_max - dy][x_max + dx] > noise and
           img[y_max + dy][x_max - dx] > noise and
           img[y_max - dy][x_max - dx] > noise):
        dx += 1
        dy += 1
    return x_max, y_max, dx

def get_spec_ind(data):
    pass


if __name__ == "__main__":
    # loading data
    map1 = np.loadtxt('/home/rtodorov/wandering-jet/results/3C454.3/helcon_pitch70_los_0.5_coneha_0.125_delta_0.1_T_20.0_phi0_0.000000/stack_i.txt')
    map2 = np.loadtxt('/home/rtodorov/wandering-jet/results/3C454.3/helcon_pitch70_los_0.5_coneha_0.125_delta_0.1_T_20.0_phi0_0.000000/stack_p.txt')

    core = modelfit_core_wo_extending('/home/rtodorov/maps-shifts/1641+399.j.2006_06_15.uvf', 
                                      (512, 0.1), [1], path_to_script='/home/rtodorov/maps-shifts/script_clean_rms.txt',
                                      dump_json_result=False)

    print(core)
    
    
    
    mapsize = map1.shape
    if map2.shape != mapsize:
        raise Exception("Incomparable maps")
    data = [map1, map2]

    noise = np.std(data[0]) + np.std(data[1])

    x_max, y_max, radius = get_core(data[0], noise)

    mask = np.ones(mapsize)
    for x in np.arange(mapsize[0]):
        for y in np.arange(mapsize[1]):
            if np.hypot(x_max - x, y_max - y) <= radius:
                mask[y, x] = 0

    detected_shift = phase_cross_correlation(data[0], data[1], reference_mask=mask)
    print(detected_shift)








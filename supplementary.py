import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from main import registrate_images, get_core_img_mask
import sys
sys.path.insert(0, '/home/rtodorov/jetpol/ve/vlbi_errors')
from spydiff import get_uvrange
from utils import find_bbox, find_image_std
from image import plot as iplot


def process_mojave_shift_data(data_dir, save_dir, mapsize):
    with open(os.path.join(data_dir, 'core_shifts_mojave.txt')) as f:
        lines = f.readlines()
    
    shifts = {'sourse':[],
              'core_shift_dec_81':[],
              'core_shift_ra_81':[],
              'core_shift_dec_84':[],
              'core_shift_ra_84':[],
              'core_shift_dec_121':[],
              'core_shift_ra_121':[]}
    for line in lines:
        line = line.split()
        sourse = line[1]
        date = line[2]
        pa = float(line[3])
        shifts['sourse'].append(sourse)
        if line[4] != '-':
            shifts['core_shift_dec_81'].append(float(line[5])*np.sin(float(line[4])+pa))
            shifts['core_shift_ra_81'].append(float(line[5])*np.cos(float(line[4])+pa))
        else:
            shifts['core_shift_dec_81'].append('-')
            shifts['core_shift_ra_81'].append('-')
        if line[7] != '-':
            shifts['core_shift_dec_84'].append(float(line[8])*np.sin(float(line[7])+pa))
            shifts['core_shift_ra_84'].append(float(line[8])*np.cos(float(line[7])+pa))
        else:
            shifts['core_shift_dec_84'].append('-')
            shifts['core_shift_ra_84'].append('-')
        if line[10] != '-':
            shifts['core_shift_dec_121'].append(float(line[11])*np.sin(float(line[10])+pa))
            shifts['core_shift_ra_121'].append(float(line[11])*np.cos(float(line[10])+pa))
        else:
            shifts['core_shift_dec_121'].append('-')
            shifts['core_shift_ra_121'].append('-')
    data = pd.DataFrame(shifts)
    with open(os.path.join(save_dir, 'core_shifts_mojave_proc.txt'), 'w') as fo:
        fo.write(data.to_string())


def read_shift_data(data_file, data_dir):
    with open(os.path.join(data_dir, data_file)) as f:
        lines = f.readlines()
    data_dict = {}
    dict_keys = lines[0].split()
    if 'sourse' in dict_keys:
        dict_keys.remove('sourse')
    else:
        raise Exception("Incorrect data type!")
    
    for line in lines:
        if line == lines[0]:
            continue
        line = line.split()
        shift_dict = {}
        for i, key in enumerate(dict_keys):
            if line[i+2] == '-':
                shift_dict[key] = None
            else:
                shift_dict[key] = float(line[i+2])
        data_dict[line[1]] = shift_dict
    return data_dict


def plot_shiftdata_diff(data_dict1, data_dict2, freqs, mapsize):
    shifts1 = {}
    shifts2 = {}
    for sourse in data_dict1:
        if sourse in data_dict2:
            data1 = data_dict1[sourse]
            data2 = data_dict2[sourse]
            for freq in freqs:
                if not freq in shifts1:
                    shifts1[freq] = []
                if not freq in shifts2:
                    shifts2[freq] = []
                if data1['core_shift_dec_{}'.format(freq).replace('.', '')] is not None and \
                        data2['core_shift_dec_{}'.format(freq).replace('.', '')] is not None:
                    shifts1[freq].append(np.hypot(data1['core_shift_dec_{}'.format(freq).replace('.', '')],
                                                 data1['core_shift_ra_{}'.format(freq).replace('.', '')]))
                    shifts2[freq].append(np.hypot(data2['core_shift_dec_{}'.format(freq).replace('.', '')],
                                                 data2['core_shift_ra_{}'.format(freq).replace('.', '')]))
    gridsize = 0.01

    for freq in freqs:
        plotrange = max(np.max(shifts1[freq]), np.max(shifts2[freq]))
        plotrange = 0.5
        plotrange = (plotrange//gridsize+2)*gridsize
        ax = np.linspace(0, plotrange, int(plotrange//gridsize))
        data_arr = np.zeros((ax.size, ax.size))
        for shift1, shift2 in zip(shifts1[freq], shifts2[freq]):
            try:
                data_arr[int(shift1//gridsize), int(shift2//gridsize)] += 1 
            except:
                pass
        
        data_arr = np.flip(data_arr, axis=0)
        plt.imshow(data_arr, extent=[0, plotrange, 0, plotrange])
        plt.title("My median shift = {}, Mojave median shift = {}".format(
            round(np.median(shifts1[freq]), 2), round(np.median(shifts2[freq]), 2)))
        plt.ylabel('My shift, mas')
        plt.xlabel('Mojave shift, mas')
        plt.plot(ax, ax)
        plt.savefig('data_comp_{}.png'.format(freq), bbox_inches='tight')
        plt.close()


def test_script(uvf1, uvf2, target, base_dir):
    mapsize = (1024, 0.1)
    deep_clean = False
    uvfs = [uvf1, uvf2]

    if True:
        # getting images, core parameters and masks according to cores
        imgs = []
        cores = []
        masks = []
        beams = []
        beam = None

        uv_min = 0
        uv_max = np.inf
        for uvf in uvfs:
            uv_min_, uv_max_ = get_uvrange(uvf)
            if uv_min_ > uv_min:
                uv_min = uv_min_
            if uv_max_ < uv_max:
                uv_max = uv_max_                                

        with open(os.path.join(base_dir, "script_clean_rms.txt")) as f:
            lines = f.readlines()

        lines.insert(86, 'uvrange {}, {}'.format(uv_min, uv_max))
        if deep_clean:
            lines[90] = 'float overclean_coef; overclean_coef = 3.0\n'
        else:
            lines[90] = 'float overclean_coef; overclean_coef = 1.0\n'

        with open(os.path.join(base_dir, 'scripts/script_clean_rms_totest.txt'), 'w') as f:
            f.writelines(lines)

        for uvf in uvfs:
            img, core, mask, beam = get_core_img_mask(uvf, mapsize, [1], 
                                                path_to_script=os.path.join(base_dir, 'scripts/script_clean_rms_totest.txt'),
                                                dump_json_result=False, base_dir=base_dir, beam=beam)
            imgs.append(img)
            cores.append(core)
            masks.append(mask)
            beams.append(beam)
        
        # correlate maps with the 15.4 GHz and shift if nesessary
        shift_arr =  registrate_images(imgs[0], imgs[1], masks[0], masks[1])
        print('Target shift = {} mas, {} mas; Obtained shift = {} mas, {} mas'.format(target[0], 
                        target[1], shift_arr[0]*mapsize[1], shift_arr[1]*mapsize[1]))

        beam = beams[0]
        npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
        img_toplot = imgs[0]
        std = find_image_std(img_toplot, npixels_beam)
        blc, trc = find_bbox(img_toplot, level=5*std, min_maxintensity_mjyperbeam=20*std,
                            min_area_pix=2*npixels_beam, delta=10)
        if blc[0] == 0: blc = (blc[0] + 1, blc[1])
        if blc[1] == 0: blc = (blc[0], blc[1] + 1)
        if trc[0] == img_toplot.shape: trc = (trc[0] - 1, trc[1])
        if trc[1] == img_toplot.shape: trc = (trc[0], trc[1] - 1)
        x = np.linspace(-mapsize[0]/2*mapsize[1]/206265000, mapsize[0]/2*mapsize[1]/206265000, mapsize[0])
        y = np.linspace(mapsize[0]/2*mapsize[1]/206265000, -mapsize[0]/2*mapsize[1]/206265000, mapsize[0])
        colors_mask = img_toplot < 3*std
        iplot(contours=img_toplot, colors=None, vectors=None, vectors_values=None, x=x,
                y=y, cmap='gist_rainbow', min_abs_level=3*std, colors_mask=colors_mask, beam=beam,
                blc=blc, trc=trc, colorbar_label=None, show_beam=True)
        plt.savefig(os.path.join(base_dir, 'img1.png'), bbox_inches='tight')
        plt.close()

        iplot(contours=imgs[1], colors=None, vectors=None, vectors_values=None, x=x,
                y=y, cmap='gist_rainbow', min_abs_level=3*std, colors_mask=colors_mask, beam=beam,
                blc=blc, trc=trc, colorbar_label=None, show_beam=True)
        plt.savefig(os.path.join(base_dir, 'img2.png'), bbox_inches='tight')
        plt.close()
            



if __name__ == "__main__":
    data_dir = '/home/rtodorov/maps-shifts'
    save_dir = data_dir
    mapsize = (1024, 0.1)
    freqs = [8.1, 8.4, 12.1]
    my_data = read_shift_data('un_shift_data.txt', data_dir)
    mojave_data = read_shift_data('core_shifts_mojave_proc.txt', data_dir)
    plot_shiftdata_diff(my_data, mojave_data, freqs, mapsize)
    process_mojave_shift_data(data_dir, save_dir, mapsize)

    #target = 4.820488171754302442e-01, 1.465010422536473234e-01
    #test_script('/home/rtodorov/maps-shifts/shifted_u_15.4GHz.uvf', 
    #            '/home/rtodorov/maps-shifts/shifted_x_8.1GHz.uvf', 
    #            target=target, base_dir=data_dir)

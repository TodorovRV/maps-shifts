import numpy as np
import os
import pandas as pd
import matplotlib.pyplot  as plt


def process_mojave_shift_data(data_dir, save_dir, mapsize):
    with open(os.path.join(data_dir, 'core_shifts_mojave.txt')) as f:
        lines = f.readlines()
    
    shifts = {'sourse':[],
              'shift_dec_81':[],
              'shift_ra_81':[],
              'shift_dec_84':[],
              'shift_ra_84':[],
              'shift_dec_121':[],
              'shift_ra_121':[]}
    for line in lines:
        line = line.split()
        sourse = line[1]
        date = line[2]
        pa = float(line[3])
        shifts['sourse'].append(sourse)
        if line[4] != '-':
            shifts['shift_dec_81'].append(float(line[5])*np.sin(float(line[4])+pa))
            shifts['shift_ra_81'].append(float(line[5])*np.cos(float(line[4])+pa))
        else:
            shifts['shift_dec_81'].append('-')
            shifts['shift_ra_81'].append('-')
        if line[7] != '-':
            shifts['shift_dec_84'].append(float(line[8])*np.sin(float(line[7])+pa))
            shifts['shift_ra_84'].append(float(line[8])*np.cos(float(line[7])+pa))
        else:
            shifts['shift_dec_84'].append('-')
            shifts['shift_ra_84'].append('-')
        if line[10] != '-':
            shifts['shift_dec_121'].append(float(line[11])*np.sin(float(line[10])+pa))
            shifts['shift_ra_121'].append(float(line[11])*np.cos(float(line[10])+pa))
        else:
            shifts['shift_dec_121'].append('-')
            shifts['shift_ra_121'].append('-')
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
        line = line.split()
        if len(line) < 8:
            continue
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
                if data1['shift_dec_{}'.format(freq).replace('.', '')] is not None and \
                        data2['shift_dec_{}'.format(freq).replace('.', '')] is not None:
                    shifts1[freq].append(np.hypot(data1['shift_dec_{}'.format(freq).replace('.', '')],
                                                 data1['shift_ra_{}'.format(freq).replace('.', '')]))
                    shifts2[freq].append(np.hypot(data2['shift_dec_{}'.format(freq).replace('.', '')],
                                                 data2['shift_ra_{}'.format(freq).replace('.', '')]))
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

if __name__ == "__main__":
    data_dir = '/home/rtodorov/maps-shifts'
    save_dir = data_dir
    mapsize = (1024, 0.1)
    freqs = [8.1, 8.4, 12.1]
    my_data = read_shift_data('core_shifts.txt', data_dir)
    mojave_data = read_shift_data('core_shifts_mojave_proc.txt', data_dir)
    plot_shiftdata_diff(my_data, mojave_data, freqs, mapsize)
    # process_mojave_shift_data(data_dir, save_dir, mapsize)

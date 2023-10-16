import numpy as np
import os
import pandas as pd


def process_shift_data(data_dir, save_dir, mapsize):
    with open(os.path.join(data_dir, 'core_shifts_mojave.txt')) as f:
        lines = f.readlines()
    
    shifts = {'sourse':[],
              'shift_x_81':[],
              'shift_y_81':[],
              'shift_x_84':[],
              'shift_y_84':[],
              'shift_x_121':[],
              'shift_y_121':[]}
    for line in lines:
        line = line.split()
        sourse = line[1]
        date = line[2]
        pa = float(line[3])
        shifts['sourse'].append(sourse)
        if line[4] != '-':
            shifts['shift_x_81'].append(int(-float(line[5])*np.sin(float(line[4])+pa)/mapsize[1]))
            shifts['shift_y_81'].append(int(-float(line[5])*np.cos(float(line[4])+pa)/mapsize[1]))
        else:
            shifts['shift_x_81'].append('-')
            shifts['shift_y_81'].append('-')
        if line[7] != '-':
            shifts['shift_x_84'].append(int(-float(line[8])*np.sin(float(line[7])+pa)/mapsize[1]))
            shifts['shift_y_84'].append(int(-float(line[8])*np.cos(float(line[7])+pa)/mapsize[1]))
        else:
            shifts['shift_x_84'].append('-')
            shifts['shift_y_84'].append('-')
        if line[10] != '-':
            shifts['shift_x_121'].append(int(-float(line[11])*np.sin(float(line[10])+pa)/mapsize[1]))
            shifts['shift_y_121'].append(int(-float(line[11])*np.cos(float(line[10])+pa)/mapsize[1]))
        else:
            shifts['shift_x_121'].append('-')
            shifts['shift_y_121'].append('-')
    data = pd.DataFrame(shifts)
    with open(os.path.join(save_dir, 'core_shifts_mojave_proc.txt'), 'w') as fo:
        fo.write(data.to_string())


if __name__ == "__main__":
    data_dir = '/home/rtodorov/maps-shifts'
    save_dir = data_dir
    mapsize = (1024, 0.1)
    process_shift_data(data_dir, save_dir, mapsize)

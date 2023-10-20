import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
import sys
import os
from tempfile import TemporaryDirectory
sys.path.insert(0, '/home/rtodorov/jetpol/ve/vlbi_errors')
from spydiff import (clean_difmap, find_nw_beam, create_clean_image_from_fits_file,
            create_difmap_file_from_single_component, filter_difmap_CC_model_by_r,
            join_difmap_models, modelfit_difmap, get_uvrange, convert_difmap_model_file_to_CCFITS)
from uv_data import UVData
from image import plot as iplot
from utils import find_bbox, find_image_std


def load_data(sourse, date, freq):
        freqs_dict = {8.1:'x', 8.4:'y', 12.1:'j'}
        data_dir_multi = '/mnt/jet1/yyk/VLBI/2cmVLBA/Udata/multifreq'
        if freq == 15.4:
            return '/mnt/jet1/yyk/VLBI/2cmVLBA/Udata/{}/{}/{}.u.{}.uvf'.format(sourse, date, sourse, date)
        else:
            return '{}/{}.{}.{}.uvf'.format(data_dir_multi, sourse, freqs_dict[freq], date)


def create_individual_script(sourse, date, freqs, base_dir, deep_clean=False):
    uv_min = np.inf
    uv_max = 0
    for freq in freqs:
        uv_min_, uv_max_ = get_uvrange(load_data(sourse, date, freq))
        if uv_min_ < uv_min:
            uv_min = uv_min_
        if uv_max_ > uv_max:
            uv_max = uv_max_                                

    with open(os.path.join(base_dir, "script_clean_rms.txt")) as f:
        lines = f.readlines()

    lines.insert(86, 'uvrange {}, {}'.format(uv_min, uv_max))
    if deep_clean:
        lines[90] = 'float overclean_coef; overclean_coef = 3.0\n'
    else:
        lines[90] = 'float overclean_coef; overclean_coef = 1.0\n'

    with open(os.path.join(base_dir, 'scripts/script_clean_rms_{}.txt'.format(sourse)), 'w') as f:
        f.writelines(lines)


def get_core_img_mask(uvfits, mapsize_clean, beam_fractions, path_to_script, use_elliptical=False,
                               use_brightest_pixel_as_initial_guess=True, save_dir=None,
                               dump_json_result=True, base_dir=None, beam=None):

    with TemporaryDirectory() as working_dir:
        # First CLEAN and dump difmap model file with CCs
        clean_difmap(uvfits, os.path.join(working_dir, "test_cc.fits"), "i",
                     mapsize_clean, path_to_script=path_to_script,
                     mapsize_restore=None, beam_restore=beam, shift=None,
                     show_difmap_output=True, command_file=None, clean_box=None,
                     save_dfm_model=os.path.join(working_dir, "cc.mdl"),
                     omit_residuals=False, do_smooth=True, dmap=None,
                     text_box=None, box_rms_factor=None, window_file=None,
                     super_unif_dynam=None, unif_dynam=None,
                     taper_gaussian_value=None, taper_gaussian_radius=None)

        if base_dir is not None:
            os.remove(os.path.join(base_dir, "difmap.log"))
        uvdata = UVData(uvfits)
        freq_hz = uvdata.frequency
        # Find beam
        if beam is None:
            bmin, bmaj, bpa = find_nw_beam(uvfits, stokes="i", mapsize=mapsize_clean, uv_range=None, working_dir=working_dir)
        else:
            bmin, bmaj, bpa = beam
        print("NW beam : {:.2f} mas, {:.2f} mas, {:.2f} deg".format(bmaj, bmin, bpa))
        # find resulting image
        ccimage = create_clean_image_from_fits_file(os.path.join(working_dir, "test_cc.fits"))
        # detect core and find image
        core = detect_core(ccimage, uvfits, beam_fractions, bmaj, mapsize_clean, freq_hz,
                use_brightest_pixel_as_initial_guess, use_elliptical, working_dir)
        beam = (bmaj, bmin, bpa)
        npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
        std = find_image_std(ccimage.image, npixels_beam)
        mask = create_mask_from_core(core, mapsize, working_dir, beam, uvfits, std)
        if base_dir is not None:
            os.remove(os.path.join(base_dir, "difmap.log"))

    return ccimage.image, core[1], mask, beam


def detect_core(ccimage, uvfits, beam_fractions, bmaj, mapsize_clean, freq_hz,
                use_brightest_pixel_as_initial_guess, use_elliptical, working_dir):
    core = dict()
    # Find the brightest pixel
    if use_brightest_pixel_as_initial_guess:
        im = np.unravel_index(np.argmax(ccimage.image), ccimage.image.shape)
        print("indexes of max intensity ", im)
        # - to RA cause dx_RA < 0
        r_c = (-(im[1]-mapsize_clean[0]/2)*mapsize_clean[1],
              (im[0]-mapsize_clean[0]/2)*mapsize_clean[1])
    else:
        r_c = (0, 0)
    print("Brightest pixel coordinates (RA, DEC) : {:.2f}, {:.2f}".format(r_c[0], r_c[1]))

    # Create model with a single component
    if not use_elliptical:
        comp = (1., r_c[0], r_c[1], 0.25)
    else:
        comp = (1., r_c[0], r_c[1], 0.25, 1.0, 0.0)
    create_difmap_file_from_single_component(comp, os.path.join(working_dir, "1.mdl"), freq_hz)
    
    for beam_fraction in beam_fractions:
        # Filter CCs
        filter_difmap_CC_model_by_r(os.path.join(working_dir, "cc.mdl"),
                                    os.path.join(working_dir, "filtered_cc.mdl"),
                                    r_c, bmaj*beam_fraction)
        # Add single gaussian component model to CC model
        join_difmap_models(os.path.join(working_dir, "filtered_cc.mdl"),
                           os.path.join(working_dir, "1.mdl"),
                           os.path.join(working_dir, "hybrid.mdl"))
        modelfit_difmap(uvfits, mdl_fname=os.path.join(working_dir, "hybrid.mdl"),
                        out_fname=os.path.join(working_dir, "hybrid_fitted.mdl"),
                        niter=100, stokes='i', show_difmap_output=True)

        # Extract core parameters
        with open(os.path.join(working_dir, "hybrid_fitted.mdl"), "r") as fo:
            lines = fo.readlines()
            components = list()
            for line in lines:
                if line.startswith("!"):
                    continue
                splitted = line.split()
                if len(splitted) == 3:
                    continue
                if len(splitted) == 9:
                    flux, r, theta, major, axial, phi, type_, freq, spec  = splitted
                    flux = float(flux.strip("v"))
                    r = float(r.strip("v"))
                    theta = float(theta.strip("v"))
                    major = float(major.strip("v"))
                    axial = float(axial.strip("v"))
                    phi = float(phi.strip("v"))

                    theta = np.deg2rad(theta)
                    ra = r*np.sin(theta)
                    dec = r*np.cos(theta)

                    # CG
                    if type_ == "1":
                        component = (flux, ra, dec, major)
                    elif type_ == "2":
                        component = (flux, ra, dec, major, axial, phi)
                    else:
                        raise Exception("Component must be Circualr or Elliptical Gaussian!")
                    components.append(component)
            if len(components) > 1:
                raise Exception("There should be only one core component!")
            if not components:
                raise Exception("No core component found!")
            # return components[0]

            if not use_elliptical:
                flux, ra, dec, size = components[0]
                core.update({beam_fraction: {"flux": flux, "ra": ra,
                                               "dec": dec, "size": size,
                                               "rms": np.nan}})
            else:
                flux, ra, dec, size, e, bpa = components[0]
                core.update({beam_fraction: {"flux": flux, "ra": ra,
                                               "dec": dec, "size": size,
                                               "e": e, "bpa": bpa,
                                               "rms": np.nan}})
    if base_dir is not None:
        os.remove(os.path.join(base_dir, "difmap.log"))
    return core


def create_mask_from_core(core, mapsize, working_dir, beam, uvfits, std):
    convert_difmap_model_file_to_CCFITS(os.path.join(working_dir, "1.mdl"), stokes='i', mapsize=mapsize,
                                        restore_beam=beam, uvfits_template=uvfits,
                                        out_ccfits=os.path.join(working_dir, "1.fits"), shift=None,
                                        show_difmap_output=True)
    core_image = create_clean_image_from_fits_file(os.path.join(working_dir, "1.fits"))
    mask = np.ones((mapsize[0], mapsize[0]))
    mask[core_image.image > 3*std] = 0
    return mask


def create_mask_from_round_core(core, mapsize):
    # function creates mask for map deleting sourse core
    mask = np.ones((mapsize[0], mapsize[0]))
    shift_dec_pix = int(core["dec"]/mapsize[1])
    shift_ra_pix = int(core["ra"]/mapsize[1])
    r_pix = int(core["size"]/mapsize[1])
    for dec_pix in np.arange(mapsize[0]):
        for ra_pix in np.arange(mapsize[0]):
            if (dec_pix-mapsize[0]/2-shift_dec_pix)**2 + (ra_pix-mapsize[0]/2+shift_ra_pix)**2 <= (10*r_pix)**2:
                mask[dec_pix, ra_pix] = 0
    return mask


def get_spec_ind(imgs, freq, npixels_beam):
    imgs_log = []
    freqs_log = []
    for img in imgs:
        std = find_image_std(img, npixels_beam)
        img_copy = img.copy()
        img_copy[img < std] = std
        imgs_log.append(np.log(img_copy))    
    for freq in freq:
        freqs_log.append(np.log(freq))    

    shape = imgs_log[0].shape
    imgs_log = np.array(imgs_log)
    spec_ind_map = np.polyfit(freqs_log, imgs_log.reshape((len(freqs_log), shape[0]*shape[1])), 1)[0]
    spec_ind_map = spec_ind_map.reshape(shape)
    return spec_ind_map



if __name__ == "__main__":
    # satting arguments
    base_dir = '/home/rtodorov/maps-shifts'
    mapsize = (1024, 0.1)
    deep_clean = False
    freqs = [8.1, 8.4, 12.1, 15.4]
    freqs = np.sort(np.array(freqs))
    
    with open("sourse_date_list.txt") as f:
        lines = f.readlines()
    
    core_shifts = {'sourse':[],
              'shift_dec_81':[],
              'shift_ra_81':[],
              'shift_dec_84':[],
              'shift_ra_84':[],
              'shift_dec_121':[],
              'shift_ra_121':[]}

    for line in lines:
        arr = line.split()
        if arr[0] == 'skip':
            continue
        sourse = arr[0]
        core_shifts['sourse'].append(sourse)
        date = arr[1]
        # getting images, core parameters and masks according to cores
        imgs = []
        cores = []
        masks = []
        beams = []
        beam = None
        create_individual_script(sourse, date, freqs, base_dir, deep_clean=deep_clean)
        for freq in freqs:
            img, core, mask, beam = get_core_img_mask(load_data(sourse, date, freq), mapsize, [1], 
                                                path_to_script=os.path.join(base_dir, 'scripts/script_clean_rms_{}.txt'.format(sourse)),
                                                dump_json_result=False, base_dir=base_dir, beam=beam)
            imgs.append(img)
            cores.append(core)
            masks.append(mask)
            beams.append(beam)
        
        # correlate maps with the 15.4 GHz and shift if nesessary
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            shift_arr = phase_cross_correlation(imgs[-1], img, reference_mask=masks[-1], moving_mask=mask)
            img = np.roll(img, int(shift_arr[0]), axis=0)
            img = np.roll(img, int(shift_arr[1]), axis=1)
            if freqs[i] != 15.4:
                #core_shifts['shift_dec_{}'.format(freqs[i]).replace('.', '')].append(shift_arr[0]*mapsize[1])
                #core_shifts['shift_ra_{}'.format(freqs[i]).replace('.', '')].append(shift_arr[1]*mapsize[1])
                core_shifts['shift_dec_{}'.format(freqs[i]).replace('.', '')].append(-cores[i]['dec'] 
                                                                                     -shift_arr[0]*mapsize[1]+cores[-1]['dec'])
                core_shifts['shift_ra_{}'.format(freqs[i]).replace('.', '')].append(-cores[i]['ra'] 
                                                                                     +shift_arr[1]*mapsize[1]+cores[-1]['ra'])
        
        beam = beams[-1]
        npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
        spec_ind_map = get_spec_ind(imgs, freqs, npixels_beam)

        # plot spectral index map
        img_toplot = imgs[-1]
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
        iplot(contours=img_toplot, colors=spec_ind_map, vectors=None, vectors_values=None, x=x,
                y=y, cmap='gist_rainbow', min_abs_level=3*std, colors_mask=colors_mask, beam=beam,
                blc=blc, trc=trc, colorbar_label='$\\alpha$', show_beam=True)

        plt.savefig(os.path.join(base_dir, 'index_maps/spec_ind_map_{}.png'.format(sourse)), bbox_inches='tight')

    data = pd.DataFrame(core_shifts)
    with open(os.path.join(base_dir, 'core_shifts.txt'), 'w') as fo:
        fo.write(data.to_string())

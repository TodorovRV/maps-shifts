import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
import sys
import os
from tempfile import TemporaryDirectory
sys.path.insert(0, '/home/rtodorov/jetpol/ve/vlbi_errors')
from spydiff import (clean_difmap, find_nw_beam, create_clean_image_from_fits_file,
            create_difmap_file_from_single_component, filter_difmap_CC_model_by_r,
            join_difmap_models, modelfit_difmap, convert_difmap_model_file_to_CCFITS)
from uv_data import UVData
from image import plot as iplot
from utils import find_bbox, find_image_std


def get_core_img_mask(uvfits, mapsize_clean, beam_fractions, path_to_script, use_elliptical=False,
                               use_brightest_pixel_as_initial_guess=True, save_dir=None,
                               dump_json_result=True, base_dir=None):

    with TemporaryDirectory() as working_dir:
        # First CLEAN and dump difmap model file with CCs
        clean_difmap(uvfits, os.path.join(working_dir, "test_cc.fits"), "i",
                     mapsize_clean, path_to_script=path_to_script,
                     mapsize_restore=None, beam_restore=None, shift=None,
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
        bmin, bmaj, bpa = find_nw_beam(uvfits, stokes="i", mapsize=mapsize_clean, uv_range=None, working_dir=working_dir)
        print("NW beam : {:.2f} mas, {:.2f} mas, {:.2f} deg".format(bmaj, bmin, bpa))
        # find resulting image
        ccimage = create_clean_image_from_fits_file(os.path.join(working_dir, "test_cc.fits"))
        # detect core and find image
        core = detect_core(ccimage, uvfits, beam_fractions, bmaj, mapsize_clean, freq_hz,
                use_brightest_pixel_as_initial_guess, use_elliptical, working_dir)
        mask = create_mask_from_round_core(core[1], mapsize_clean)
        beam = (bmin, bmaj, bpa)

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


def create_mask_from_round_core(core, mapsize):
    # function creates mask for map deleting sourse core
    mask = np.ones((mapsize[0], mapsize[0]))
    shift_dec_pix = int(core["dec"]/mapsize[1])
    shift_ra_pix = int(core["ra"]/mapsize[1])
    r_pix = int(core["size"]/mapsize[1])
    for dec_pix in np.arange(mapsize[0]):
        for ra_pix in np.arange(mapsize[0]):
            if (dec_pix-mapsize[0]/2-shift_dec_pix)**2 + (ra_pix-mapsize[0]/2+shift_ra_pix)**2 <= r_pix**2:
                mask[dec_pix, ra_pix] = 0
    return mask


def get_spec_ind(imgs, freqs):
    for img in imgs:
        std = np.std(img)
        img[img < std] = std
        img = np.log(img)    
    for freq in freqs:
        freq = np.log(freq)

    shape = imgs[0].shape
    imgs = np.array(imgs)
    spec_ind_map = np.polyfit(freqs, imgs.reshape((len(freqs), shape[0]*shape[1])), 1)[0]
    spec_ind_map = spec_ind_map.reshape(shape)
    return spec_ind_map


if __name__ == "__main__":
    # satting arguments
    data_dir = '/mnt/jet1/yyk/VLBI/2cmVLBA/Udata/multifreq'
    base_dir = '/home/rtodorov/maps-shifts'
    sourse = '1641+399'
    date = '2006_06_15'
    mapsize = (1024, 0.1)

    freqs_dict = {12.1:'j', 8.1:'x', 8.4:'y'}
    
    # getting images, core parameters and masks according to cores
    imgs = []
    cores = []
    masks = []
    freqs = []
    beams = []
    for freq in freqs_dict:
        print(freq)
        img, core, mask, beam = get_core_img_mask('{}/{}.{}.{}.uvf'.format(data_dir, sourse, freqs_dict[freq], date), 
                                            mapsize, [1], path_to_script='{}/script_clean_rms.txt'.format(base_dir),
                                            dump_json_result=False, base_dir=base_dir)
        imgs.append(img)
        cores.append(core)
        masks.append(mask)
        freqs.append(freq)
        beams.append(beam)

    # imgs[2][imgs[2] < 0] = 0
    # plt.imshow(np.log(imgs[2]*masks[2]+0.0001))
    # plt.savefig("img.png")

    # correlate maps with the first one and shift if nesessary
    for img, mask in zip(imgs, masks):
        shift = phase_cross_correlation(imgs[0], img, reference_mask=masks[0]*mask)
        img = np.roll(img, shift)

    spec_ind_map = get_spec_ind(imgs, freqs)

    # plot spectral index map
    beam = beams[0]
    npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
    std = np.std(imgs[0])
    blc, trc = find_bbox(imgs[0], level=3*std, min_maxintensity_mjyperbeam=4 * std,
                         min_area_pix=npixels_beam, delta=10)
    if blc[0] == 0: blc = (blc[0] + 1, blc[1])
    if blc[1] == 0: blc = (blc[0], blc[1] + 1)
    if trc[0] == imgs[0].shape: trc = (trc[0] - 1, trc[1])
    if trc[1] == imgs[0].shape: trc = (trc[0], trc[1] - 1)
    x = np.linspace(-mapsize[0]/2*mapsize[1]/206265000, mapsize[0]/2*mapsize[1]/206265000, mapsize[0])
    y = np.linspace(mapsize[0]/2*mapsize[1]/206265000, -mapsize[0]/2*mapsize[1]/206265000, mapsize[0])
    print(x)
    colors_mask = imgs[0] < 2*std
    iplot(contours=imgs[0], colors=spec_ind_map, vectors=None, vectors_values=None, x=x,
            y=y, cmap='coolwarm', min_abs_level=std, colors_mask=colors_mask, beam=beam,
            blc=blc, trc=trc, colorbar_label='$\\alpha$', show_beam=True)

    plt.savefig("spec_ind_map.png")

import warnings
# Ignore known numexpr.necompiler and table.conditions warning
warnings.filterwarnings(
    'ignore',
    message="using `oa_ndim == 0` when `op_axes` is NULL is deprecated.*",
    category=DeprecationWarning)


import numpy as np
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from Quaternion import Quat
from Ska.quatutil import radec2yagzag
import chandra_aca
import acq_char

ARC_2_PIX = 1 / acq_char.General['Pix2Arc']
PIX_2_ARC = acq_char.General['Pix2Arc']
RAD_2_PIX = 180/np.pi*3600*ARC_2_PIX


DITHER = 8
MANVR_ERROR = 60
fixedErrorPad = DITHER + MANVR_ERROR
fieldErrorPad = 0


def set_dither(dither):
    global DITHER
    DITHER = dither


def set_manvr_error(manvr_error):
    global MANVR_ERROR
    global fixedErrorPad
    MANVR_ERROR = manvr_error
    fixedErrorPad = DITHER + MANVR_ERROR


def check_off_chips(cone_stars):
    ypos = cone_stars['row']
    zpos = cone_stars['col']
    yPixLim = acq_char.Star['Body']['Pixels']['YPixLim']
    zPixLim = acq_char.Star['Body']['Pixels']['ZPixLim']
    edgeBuffer = acq_char.Star['Body']['Pixels']['EdgeBuffer']
    pad = fixedErrorPad * ARC_2_PIX
    yn = (yPixLim[0] + (pad + edgeBuffer))
    yp = (yPixLim[1] - (pad + edgeBuffer))
    zn = (zPixLim[0] + (pad + edgeBuffer))
    zp = (zPixLim[1] - (pad + edgeBuffer))
    ydist = np.min([yp - ypos, ypos - yn], axis=0)
    zdist = np.min([zp - zpos, zpos - zn], axis=0)
    chip_edge_dist = np.min([ydist, zdist], axis=0)
    offchip = chip_edge_dist < 0
    yag = cone_stars['yang']
    zag = cone_stars['zang']
    arcsec_pad = fixedErrorPad
    yArcSecLim = acq_char.Star['Body']['FOV']['YArcSecLim']
    ZArcSecLim = acq_char.Star['Body']['FOV']['ZArcSecLim']
    arcsec_yn = yArcSecLim[0] + arcsec_pad
    arcsec_yp = yArcSecLim[1] - arcsec_pad
    arcsec_zn = ZArcSecLim[0] + arcsec_pad
    arcsec_zp = ZArcSecLim[1] - arcsec_pad
    arcsec_ydist = np.min([arcsec_yp - yag, yag - arcsec_yn], axis=0)
    arcsec_zdist = np.min([arcsec_zp - zag, zag - arcsec_zn], axis=0)
    fov_edge_dist = np.min([arcsec_ydist, arcsec_zdist], axis=0)
    outofbounds = (yag < arcsec_yn) | (yag > arcsec_yp) | (zag < arcsec_zn) | (zag > arcsec_zp)
    return chip_edge_dist, fov_edge_dist, offchip, outofbounds


def check_mag(cone_stars, opt):
    magOneSigError = cone_stars['mag_one_sig_err']
    mag = cone_stars['MAG_ACA']
    magNSig = opt['Spoiler']['SigErrMultiplier'] * magOneSigError
    too_bright = ((mag - magNSig - opt['Inertial']['MagErrSyst'])
                  < np.min(opt['Inertial']['MagLimit']))
    too_dim = ((mag + magNSig + opt['Inertial']['MagErrSyst'])
               > np.max(opt['Inertial']['MagLimit']))
    nomag = mag == -9999
    cone_stars['too_bright'] = too_bright
    cone_stars['too_dim'] = too_dim
    cone_stars['nomag'] = nomag
    return ~too_bright & ~too_dim & ~nomag


def check_mag_spoilers(cone_stars, ok, opt):
    magOneSigError = cone_stars['mag_one_sig_err']
    stderr2 = cone_stars['mag_one_sig_err2']
    fidpad = fieldErrorPad * ARC_2_PIX
    minsep = opt['Spoiler']['MinSep'] + fidpad
    maxsep = opt['Spoiler']['MaxSep'] + fidpad
    intercept = opt['Spoiler']['Intercept'] + fidpad
    spoilslope = opt['Spoiler']['Slope']
    nSigma = opt['Spoiler']['SigErrMultiplier']
    magdifflim = opt['Spoiler']['MagDiffLimit']
    mag = cone_stars['MAG_ACA']
    mag_col = 'mag_spoiled_{}'.format(nSigma)
    mag_spoil_check = 'mag_spoil_check_{}'.format(nSigma)
    if mag_col in cone_stars.columns:
        # if ok for stage and not previously mag spoiler checked for this
        # nsigma
        ok = ok & ~cone_stars[mag_spoil_check]
    if not np.any(ok):
        return np.zeros_like(ok), ok
    coords = SkyCoord(cone_stars['RA_PMCORR'], cone_stars['DEC_PMCORR'],
                      unit='deg')
    maxsep_arcs = maxsep * PIX_2_ARC
    cand_idx_in_ok, cat_idx, spoil_dist, dist3d = search_around_sky(
        coords[ok],
        coords,
        seplimit=u.arcsec * maxsep_arcs)
    # index the candidates back into the full list so we have one
    # index to work with, shared betweek cand_idx and cat_idx
    cand_idx = np.flatnonzero(ok)[cand_idx_in_ok]
    # and try to find the spoilers in a vectorized way
    cand_mags = cone_stars['MAG_ACA'][cand_idx]
    spoiler_mags = cone_stars['MAG_ACA'][cat_idx]
    itself = cand_idx == cat_idx
    too_dim = (cand_mags - spoiler_mags) < magdifflim
    delmag = (cand_mags - spoiler_mags
              + nSigma * np.sqrt(stderr2[cand_idx] + stderr2[cat_idx]))
    thsep = intercept + delmag * spoilslope
    spoils = (spoil_dist < u.arcsec * thsep * PIX_2_ARC) & ~itself & ~too_dim
    # this is now indexed over the cat/cand idxs so, re-index again
    spoiled = np.zeros_like(ok)
    spoiled[np.unique(cand_idx[spoils])] = True
    # and include any previous spoilers
    spoiled = spoiled | cone_stars[mag_col]
    return spoiled, ok


def check_bad_pixels(cone_stars, not_bad, opt):
    bad_pixels = opt['Body']['Pixels']['BadPixels']
    row, col = cone_stars['row'], cone_stars['col']
    cand = cone_stars[not_bad]
    r = row[not_bad]
    c = col[not_bad]
    pad = .5 + fixedErrorPad * ARC_2_PIX
    # start with big distances
    distance = np.ones(len(cone_stars[not_bad])) * 9999
    pix_idx = np.zeros(len(cone_stars[not_bad]))
    # small number of regions to check; vectorize later
    for idx, (rmin, rmax, cmin, cmax) in enumerate(bad_pixels):
        in_reg = ((r >= (rmin - pad)) & (r <= (rmax + pad))
                  & (c >= (cmin - pad)) & (c <= (cmax + pad)))
        r_dist = np.min(np.abs(
                [r - (rmin - pad), r - (rmax + pad)]), axis=0)
        c_dist = np.min(np.abs(
                [c - (cmin - pad), c - (cmax + pad)]), axis=0)
        # for the nearest manhattan distance, we want the max
        # of the minimums
        min_dist = np.max(np.vstack([r_dist, c_dist]), axis=0)
        min_dist[in_reg] = 0
        idxmatch = np.argmin([distance, min_dist], axis=0) == 1
        pix_idx[idxmatch] = idx
        distance = np.min([distance, min_dist], axis=0)
    full_distance = np.ones(len(cone_stars)) * 9999
    full_distance[not_bad] = distance
    return full_distance


def dist_to_bright_spoiler(cone_stars, ok, nSigma):
    magOneSigError = cone_stars['mag_one_sig_err']
    row, col = cone_stars['row'], cone_stars['col']
    mag = cone_stars['MAG_ACA']
    magerr2 = cone_stars['mag_one_sig_err2']
    errorpad = (fieldErrorPad + fixedErrorPad) * ARC_2_PIX
    dist = np.ones(len(cone_stars)) * 9999
    for cand, cand_magerr, idx in zip(cone_stars[ok],
                                      magOneSigError[ok],
                                      np.flatnonzero(ok)):
        mag_diff = (cand['MAG_ACA']
                    - mag
                    + nSigma * np.sqrt(cand_magerr ** 2 + magerr2))
        brighter = mag_diff > 0
        brighter[idx] = False
        if not np.any(brighter):
            continue
        cand_row, cand_col = row[idx], col[idx]
        rdiff = np.abs(cand_row - row[brighter])
        cdiff = np.abs(cand_col - col[brighter])
        match = np.argmin(np.max([rdiff, cdiff], axis=0))
        dist[idx] = np.max([rdiff, cdiff], axis=0)[match] - errorpad
    return dist


def check_column(cone_stars, not_bad, opt, chip_pos):
    zpixlim = opt['Body']['Pixels']['ZPixLim']
    ypixlim = opt['Body']['Pixels']['YPixLim']
    center = opt['Body']['Pixels']['Center']
    Column = opt['Body']['Column']
    Column = opt['Body']['Register']
    nSigma = opt['Spoiler']['SigErrMultiplier']
    row, col = chip_pos
    starmag = cone_stars['MAG_ACA']
    magerr2 = cone_stars['mag_one_sig_err2']
    register_pad = fixedErrorPad * ARC_2_PIX
    column_pad = fieldErrorPad * ARC_2_PIX
    pass


def check_stage(cone_stars, not_bad, opt):
    mag_ok = check_mag(cone_stars, opt)
    ok = mag_ok & not_bad
    if not np.any(ok):
        return ok
    nSigma = opt['Spoiler']['SigErrMultiplier']
    mag_check_col = 'mag_spoil_check_{}'.format(nSigma)
    if mag_check_col not in cone_stars.columns:
        cone_stars[mag_check_col] = np.zeros_like(not_bad)
        cone_stars['mag_spoiled_{}'.format(nSigma)] = np.zeros_like(not_bad)
    mag_spoiled, checked= check_mag_spoilers(cone_stars, ok, opt)
    cone_stars[mag_check_col] = cone_stars[mag_check_col] | checked
    cone_stars['mag_spoiled_{}'.format(nSigma)] = (
        cone_stars['mag_spoiled_{}'.format(nSigma)] | mag_spoiled)
    bad_pix_dist = check_bad_pixels(cone_stars, ok & ~mag_spoiled, opt)
    cone_stars['bad_pix_dist'] = bad_pix_dist

    # these star distance checks are in pixels, so just do them for
    # every roll
    cone_stars['star_dist_{}'.format(nSigma)] = 9999
    star_dist = dist_to_bright_spoiler(cone_stars, ok & ~mag_spoiled, nSigma)
    cone_stars['star_dist_{}'.format(nSigma)] = np.min(
                [cone_stars['star_dist_{}'.format(nSigma)], star_dist],
                axis=0)

    minBoxArc = np.ceil(fixedErrorPad/5.0)*5
    minBoxArc = np.max([opt['Select']['MinSearchBox'], minBoxArc])
    maxBoxArc = np.array(opt['Select']['MaxSearchBox'])
    maxBoxArc = np.min(maxBoxArc[maxBoxArc >= minBoxArc])

    starBox = np.min([cone_stars['star_dist_{}'.format(nSigma)],
                      cone_stars['bad_pix_dist'],
                      cone_stars['chip_edge_dist'],
                      cone_stars['fov_edge_dist'] * ARC_2_PIX], axis=0)
    box_size_arc = ((starBox * PIX_2_ARC) // 5) * 5
    box_size_arc[box_size_arc > maxBoxArc] = maxBoxArc
    cone_stars['box_size_arc'] = box_size_arc
    bad_box = starBox < (minBoxArc * ARC_2_PIX)
    cone_stars['bad_box'] = bad_box
#    if opt['SearchSettings']['DoColumnRegisterCheck']:
#        badcolumn = check_column(cone_stars, ok & ~mag_spoiled & ~bad_dist, opt, chip_pos)
#        ok = ok & ~badcolumn
#    if opt['SearchSettings']['DoBminusVCheck']:
#        badbv = check_bv(cone_stars, ok & ~mag_spoiled & ~bad_dist)
#        ok = ok & ~badbv
    return ok & ~mag_spoiled & ~bad_box


def get_mag_errs(cone_stars):
    caterr = cone_stars['MAG_ACA_ERR'] / 100.
    caterr = np.min([np.ones(len(caterr))*acq_char.Acq['Inertial']['MaxMagError'], caterr], axis=0)
    randerr = acq_char.Acq['Inertial']['MagErrRand']
    magOneSigError = np.sqrt(randerr*randerr + caterr*caterr)
    return magOneSigError, magOneSigError**2


def select_stars(ra, dec, roll, cone_stars):

    if 'mag_one_sig_err' not in cone_stars.columns:
        cone_stars['mag_one_sig_err'], cone_stars['mag_one_sig_err2'] = get_mag_errs(cone_stars)

    q = Quat((ra, dec, roll))
    yag_deg, zag_deg = radec2yagzag(cone_stars['RA_PMCORR'], cone_stars['DEC_PMCORR'], q)
    row, col = chandra_aca.yagzag_to_pixels(yag_deg * 3600,
                                            zag_deg * 3600, allow_bad=True)
    # update these for every new roll
    cone_stars['yang'] = yag_deg * 3600
    cone_stars['zang'] = zag_deg * 3600
    cone_stars['row'] = row
    cone_stars['col'] = col

    # none of these appear stage dependent
    chip_edge_dist, fov_edge_dist, offchip, outofbounds = check_off_chips(cone_stars)
    cone_stars['offchip'] = offchip
    cone_stars['outofbounds'] = outofbounds
    cone_stars['chip_edge_dist'] = chip_edge_dist
    cone_stars['fov_edge_dist'] = fov_edge_dist

    bad_mag_error = cone_stars['MAG_ACA_ERR'] > acq_char.Acq['Inertial']['MagErrorTol']
    cone_stars['bad_mag_error'] = bad_mag_error

    bad_pos_error = cone_stars['POS_ERR'] > acq_char.Acq['Inertial']['PosErrorTol']
    cone_stars['bad_pos_error'] = bad_pos_error

    bad_aspq1 = ((cone_stars['ASPQ1'] > np.max(acq_char.Acq['Inertial']['ASPQ1Lim']))
                  | (cone_stars['ASPQ1'] < np.min(acq_char.Acq['Inertial']['ASPQ1Lim'])))
    cone_stars['bad_aspq1'] = bad_aspq1

    bad_aspq2 = ((cone_stars['ASPQ2'] > np.max(acq_char.Acq['Inertial']['ASPQ2Lim']))
                  | (cone_stars['ASPQ2'] < np.min(acq_char.Acq['Inertial']['ASPQ2Lim'])))
    cone_stars['bad_aspq2'] = bad_aspq2

    bad_aspq3 = ((cone_stars['ASPQ3'] > np.max(acq_char.Acq['Inertial']['ASPQ3Lim']))
                  | (cone_stars['ASPQ3'] < np.min(acq_char.Acq['Inertial']['ASPQ3Lim'])))
    cone_stars['bad_aspq3'] = bad_aspq3

    variable = cone_stars['VAR'] != -9999
    cone_stars['variable'] = variable

    nonstellar = cone_stars['CLASS'] != 0
    cone_stars['nonstellar'] = nonstellar



    not_bad = (~offchip & ~outofbounds & ~bad_mag_error & ~bad_pos_error
                & ~nonstellar & ~bad_aspq1 & ~bad_aspq2 & ~bad_aspq3 & ~variable)

    # Set some column defaults that will be updated in check_stage
    cone_stars['stage'] = -1

    stage1  = check_stage(cone_stars, not_bad, acq_char.Acq)
    cone_stars['stage'][stage1] = 1

    stage2 = check_stage(cone_stars, not_bad & ~stage1, acq_char.Acq2)
    cone_stars['stage'][stage2] = 2

    stage3 = check_stage(cone_stars, not_bad & ~stage1 & ~stage2, acq_char.Acq3)
    cone_stars['stage'][stage3] = 3

    stage4 = check_stage(cone_stars, not_bad & ~stage1 & ~stage2 & ~stage3,
                         acq_char.Acq4)
    cone_stars['stage'][stage4] = 4

    selected = cone_stars[stage1 | stage2 | stage3 | stage4]
    selected['box_delta'] = 240 - selected['box_size_arc']
    selected.sort(['stage', 'box_delta', 'MAG_ACA'])
    cone_stars['selected'] = False
    for agasc_id in selected[0:8]['AGASC_ID']:
        cone_stars['selected'][cone_stars['AGASC_ID'] == agasc_id] = True
    return selected[0:8], cone_stars


#return cone_stars[stage1 | stage2 | stage3 | stage4]




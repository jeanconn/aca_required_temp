#!/usr/bin/env python
import os
import warnings
import numpy as np
import agasc
import jinja2
import shutil
import hashlib
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import mpld3

from astropy.table import Table

from Ska.Matplotlib import plot_cxctime, cxctime2plotdate
from Chandra.Time import DateTime
import Ska.Sun
import chandra_aca
from chandra_aca.transform import calc_aca_from_targ
from chandra_aca.star_probs import t_ccd_warm_limit, set_acq_model_ms_filter
from chandra_aca.drift import get_aca_offsets, get_target_aimpoint
from chandra_aca import dark_model
from scipy.optimize import bisect

import mini_sausage

# Ignore known numexpr.necompiler and table.conditions warning
warnings.filterwarnings(
    'ignore',
    message="using `oa_ndim == 0` when `op_axes` is NULL is deprecated.*",
    category=DeprecationWarning)


# Expand the last stage of guide selection in SAUSAGE to get some fainter stars to use them
# to set a temperature
mini_sausage.STAR_CHAR['Guide'][-1]['Inertial']['MagLimit'][1] = 11.0
# Disable the "direct catalog search" for spoilers
for acqstage in mini_sausage.STAR_CHAR['Acq']:
    acqstage['SearchSettings']['DoSpoilerCheck'] = 0
for guistage in mini_sausage.STAR_CHAR['Guide']:
    guistage['SearchSettings']['DoSpoilerCheck'] = 0


PLANNING_LIMIT = -10.2
EDGE_DIST = 30
COLD_T_CCD = -16
WARM_T_CCD = -5
# explicitly disable MS filter
set_acq_model_ms_filter(ms_enabled=False)

TASK_DATA = os.path.join(os.environ['SKA'], 'data', 'aca_lts_eval')
ROLL_TABLE = Table.read(os.path.join(TASK_DATA, 'roll_limits.dat'), format='ascii')

# Save temperature calc a combination of stars
# indexed by hash of agasc ids
T_CCD_CACHE = {}

# Star catalog for an attitude (ignores proper motion)
CAT_CACHE = {}
G_CAT_CACHE = {}

# Roll independent stars
RI_CAT_CACHE = {}
G_RI_CAT_CACHE = {}

def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description="Make report of required ACA temperatures for a target over a time range.")
    parser.add_argument("--ra",
                        type=float,
                        required=True,
                        help="Target RA in degrees")
    parser.add_argument("--dec",
                        type=float,
                        required=True,
                        help="Target Dec in degrees")
    parser.add_argument("--cycle",
                        help="Observation proposal cycle")
    parser.add_argument("--detector",
                        help="SI one of ACIS-I, ACIS-S, HRC-I, or HRC-S")
    parser.add_argument("--too",
                        action="store_true")
    parser.add_argument("--y_offset",
                        type=float,
                        default=0.0,
                        help="Y target offset in arcmin")
    parser.add_argument("--z_offset",
                        type=float,
                        default=0.0,
                        help="Z target offset in arcmin")
    parser.add_argument("--out",
                        default="out",
                        help="Output directory.")
    parser.add_argument("--start",
                        default="2015-09-01",
                        help="Start time for evaluation of temperatures.  Default 2015-09-01")
    parser.add_argument("--stop",
                        default="2016-12-31",
                        help="Stop time for evaluation of temperatures.  Default 2016-12-31")
    parser.add_argument("--daystep",
                        default=1,
                        help="Step size in days when checking catalogs.  Default 1")
    parser.add_argument("--obsid",
                        help="Obsid for html report.  Just a label; not used to do a database lookup for parameters.")
    parser.add_argument("--debug",
                        action="store_true")
    opt = parser.parse_args()
    return opt


def max_temp(time, stars):
    id_hash = hashlib.md5(np.sort(stars['AGASC_ID'])).hexdigest()
    if id_hash not in T_CCD_CACHE:
        # Get tuple of (t_ccd, n_acq) for this star field and cache
        T_CCD_CACHE[id_hash] = t_ccd_warm_limit(
            date=time,
            mags=stars['MAG_ACA'],
            colors=stars['COLOR1'],
            min_n_acq=(2, 8e-3),
            cold_t_ccd=COLD_T_CCD,
            warm_t_ccd=WARM_T_CCD,
            model='spline')
#        print "calc temp for ", id_hash
#    else:
#        print "cached temp for ", id_hash
    return T_CCD_CACHE[id_hash]


def get_rolldev(pitch):
    idx = np.searchsorted(ROLL_TABLE['pitch'], pitch, side='right')
    return ROLL_TABLE['rolldev'][idx - 1]


def select_stars(ra, dec, roll, cone_stars):
    id_key = ("{:.3f}".format(ra),
              "{:.3f}".format(dec),
              "{:.3f}".format(roll))
    if id_key not in CAT_CACHE:
        CAT_CACHE[id_key] = mini_sausage.select_acq_stars(
            ra, dec, roll, n=8, cone_stars=cone_stars)[0]
    return CAT_CACHE[id_key]


def select_ri_stars(ra, dec, cone_stars):
    id_key = ("{:.3f}".format(ra),
              "{:.3f}".format(dec))
    if id_key not in RI_CAT_CACHE:
        RI_CAT_CACHE[id_key] = mini_sausage.select_acq_stars(
            ra, dec, None, n=8, cone_stars=cone_stars, roll_indep=True)[0]
    return RI_CAT_CACHE[id_key]


def select_guide_stars(ra, dec, roll, cone_stars):
    id_key = ("{:.3f}".format(ra),
              "{:.3f}".format(dec),
              "{:.3f}".format(roll))
    if id_key not in G_CAT_CACHE:
        stars = mini_sausage.select_guide_stars(
            ra, dec, roll, n=5, cone_stars=cone_stars)[0]
        stars.sort('MAG_ACA')
        G_CAT_CACHE[id_key] = stars
    return G_CAT_CACHE[id_key]


def select_ri_guide_stars(ra, dec, cone_stars):
    id_key = ("{:.3f}".format(ra),
              "{:.3f}".format(dec))
    if id_key not in G_RI_CAT_CACHE:
        stars = mini_sausage.select_guide_stars(
            ra, dec, None, n=5, cone_stars=cone_stars, roll_indep=True)[0]
        stars.sort('MAG_ACA')
        G_RI_CAT_CACHE[id_key] = stars
    return G_RI_CAT_CACHE[id_key]


def guide_count(mags, tccd=-10.2):
    """
    Given mags from guide stars and a temperature, calculate a guide star
    count using signal-to-noise scaled mag thresholds.
    """
    thresh1 = snr_mag_for_tccd(tccd, ref_mag=10.0)
    thresh2 = snr_mag_for_tccd(tccd, ref_mag=10.2)
    thresh3 = snr_mag_for_tccd(tccd, ref_mag=10.3)
    counts = np.zeros_like(mags)
    counts[mags <= thresh1] = 1.0
    counts[(mags <= thresh2) & (mags > thresh1)] = 0.75
    counts[(mags <= thresh3) & (mags > thresh2)] = 0.5
    return np.sum(counts)


def snr_mag_for_tccd(tccd, ref_mag=10.3, ref_tccd=-10.2, scale_4c=None):
    """
    Given a tccd, solve for the magnitude that has the same expected signal
    to noise as ref_mag / ref_tccd.
    """
    if scale_4c is None:
        scale_4c = dark_model.DARK_SCALE_4C
    return ref_mag - (tccd - ref_tccd) * np.log10(scale_4c) / 1.6


def t_ccd_for_guide(mags, min_guide_count=4, warm_t_ccd=-5, cold_t_ccd=-16):
    def n_gui_above_min(t_ccd):
        count = guide_count(mags, t_ccd)
        return count - min_guide_count
    # In the style of chandra_aca.star_probs.t_ccd_warm_limit, use an optimization
    # strategy to solve for the warmest temperature that still gets the min_guide_count
    merit_func = n_gui_above_min
    if merit_func(warm_t_ccd) >= 0:
        t_ccd = warm_t_ccd
    elif merit_func(cold_t_ccd) <= 0:
        t_ccd = cold_t_ccd
    else:
        t_ccd = bisect(merit_func, cold_t_ccd, warm_t_ccd, xtol=1e-4, rtol=1e-4)
    return t_ccd


def get_t_ccd_roll(ra, dec, cycle, detector, too, y_offset, z_offset, pitch, time, cone_stars):
    """
    Loop over possible roll range for this pitch and return best
    and nominal temperature/roll combinations
    """

    nom_roll = Ska.Sun.nominal_roll(ra, dec, time=time)
    # Round nominal roll to the nearest half degree
    nom_roll = np.round(np.round(nom_roll * 2) / 2., 1)
    chipx, chipy, chip_id = get_target_aimpoint(time, cycle, detector, too)
    aca_offset_y, aca_offset_z = get_aca_offsets(
        detector, chip_id, chipx, chipy, time, PLANNING_LIMIT - 2)
    # note that calc_aca_from_targ expects target offsets in degrees and the target table has them
    #  in arcmin
    q_pnt = calc_aca_from_targ((ra, dec, nom_roll),
                               (y_offset / 60.) + (aca_offset_y / 3600.),
                               (z_offset / 60.) + (aca_offset_z / 3600.))
    ra_pnt = q_pnt.ra
    dec_pnt = q_pnt.dec
    # if the offsets are both small, so the pointing attitude is relatively roll-independent
    # check the relatively roll independent circle
    if abs(y_offset) < .3 and abs(z_offset) < .3:
        guide_stars = select_ri_guide_stars(ra_pnt, dec_pnt, cone_stars)
        acq_stars = select_ri_stars(ra_pnt, dec_pnt, cone_stars)
        acq_tccd, nacq = max_temp(time=time, stars=acq_stars)
        guide_stars.sort('MAG_ACA')
        guide_tccd = t_ccd_for_guide(guide_stars['MAG_ACA']) if len(guide_stars) >= 4 else COLD_T_CCD
        t_ccd = np.min([acq_tccd, guide_tccd])
        if t_ccd >= WARM_T_CCD:
            nom = {'roll': nom_roll,
                   'q_pnt': q_pnt,
                   'acq_tccd': acq_tccd,
                   'guide_tccd': guide_tccd,
                   'acq_stars': acq_stars,
                   'guide_stars': guide_stars}
            return {'nomdata': nom,
                    'bestdata': nom,
                    'rolls': {nom_roll: t_ccd},
                    'cone_stars': cone_stars,
                    'roll_indep': True,
                    'comment': 'roll-independent'}
    acq_stars = select_stars(ra_pnt, dec_pnt, nom_roll, cone_stars)
    guide_stars = select_guide_stars(ra_pnt, dec_pnt, nom_roll, cone_stars)
    guide_stars.sort('MAG_ACA')
    acq_tccd, nacq = max_temp(time=time, stars=acq_stars)
    guide_tccd = t_ccd_for_guide(guide_stars['MAG_ACA']) if len(guide_stars) >= 4 else COLD_T_CCD
    t_ccd = np.min([acq_tccd, guide_tccd])
    nom = {'roll': nom_roll,
           'q_pnt': q_pnt,
           'acq_tccd': acq_tccd,
           'guide_tccd': guide_tccd,
           'acq_stars': acq_stars,
           'guide_stars': guide_stars}
    all_rolls = {nom_roll: t_ccd}
    # if nom_t_ccd is WARM_T_CCD, stop now
    if t_ccd >= WARM_T_CCD:
        best = nom
        return {'nomdata': nom,
                'bestdata': best,
                'rolls': all_rolls,
                'cone_stars': cone_stars,
                'roll_indep': False,
                'comment': 'nominal is at max'}
    # check off nominal rolls in allowed range for a better catalog / temperature
    roll_dev = get_rolldev(pitch)
    d_roll = 1.0
    if roll_dev > d_roll:
        plus_minus_rolls = np.concatenate([[-r, r] for r in
                                           np.arange(d_roll, roll_dev, d_roll)])
        off_nom_rolls = np.round(nom_roll) + plus_minus_rolls
    else:
        off_nom_rolls = [np.round(nom_roll)]
    best_t_ccd = None
    best_is_max = False
    for roll in off_nom_rolls:
        q_pnt = calc_aca_from_targ((ra, dec, roll),
                                   (y_offset / 60.) + (aca_offset_y / 3600.),
                                   (z_offset / 60.) + (aca_offset_z / 3600.))
        ra_pnt = q_pnt.ra
        dec_pnt = q_pnt.dec
        acq_stars = select_stars(ra_pnt, dec_pnt, roll, cone_stars)
        guide_stars = select_guide_stars(ra_pnt, dec_pnt, roll, cone_stars)
        guide_stars.sort('MAG_ACA')
        acq_tccd, nacq = max_temp(time=time, stars=acq_stars)
        guide_tccd = t_ccd_for_guide(guide_stars['MAG_ACA']) if len(guide_stars) >= 4 else COLD_T_CCD
        t_ccd = np.min([acq_tccd, guide_tccd])
        all_rolls[roll] = t_ccd
        if t_ccd is not None:
            if best_t_ccd is None or t_ccd > best_t_ccd:
                best_t_ccd = t_ccd
                best = {'roll': roll,
                        'q_pnt': q_pnt,
                        'acq_tccd': acq_tccd,
                        'guide_tccd': guide_tccd,
                        'acq_stars': acq_stars,
                        'guide_stars': guide_stars}
                if abs(roll - np.round(nom_roll)) > (roll_dev - d_roll):
                    best_is_max = True
            if t_ccd >= WARM_T_CCD:
                break
    comment = ''
    if best_is_max:
        comment = 'best roll at max offset'
    return {'nomdata': nom,
            'bestdata': best,
            'rolls': all_rolls,
            'cone_stars': cone_stars,
            'roll_indep': False,
            'comment': comment}


def t_ccd_for_attitude(ra, dec, cycle, detector, too, y_offset=0, z_offset=0,
                       start='2014-09-01', stop='2015-12-31', daystep=1, outdir=None):
    # reset the caches at every new attitude
    global T_CCD_CACHE
    T_CCD_CACHE.clear()
    global CAT_CACHE
    CAT_CACHE.clear()
    global RI_CAT_CACHE
    RI_CAT_CACHE.clear()
    global G_CAT_CACHE
    G_CAT_CACHE.clear()
    global G_RI_CAT_CACHE
    G_RI_CAT_CACHE.clear()


    # Load any previously obtained results into cache
    t_ccd_file = os.path.join(outdir, "t_ccd_map.dat")
    if os.path.exists(t_ccd_file):
        t_ccd = Table.read(t_ccd_file, format='ascii')
        for row in t_ccd:
            T_CCD_CACHE[row['key']] = (row['t_ccd'], row['n_acqs'])

    cat_file = os.path.join(outdir, "cat_map.dat")
    if os.path.exists(cat_file):
        cat = Table.read(cat_file, format="ascii")
        for row in cat:
            CAT_CACHE[("{:.3f}".format(row['ra']),
                       "{:.3f}".format(row['dec']),
                       "{:.3f}".format(row['roll']))] = Table.read(
                os.path.join(outdir, "{}_stars.dat".format(row['hash'])),
                format='ascii')
    ri_cat_file = os.path.join(outdir, "ri_cat_map.dat")
    if os.path.exists(ri_cat_file):
        cat = Table.read(ri_cat_file, format="ascii")
        for row in cat:
            RI_CAT_CACHE[("{:.3f}".format(row['ra']),
                          "{:.3f}".format(row['dec']))] = Table.read(
                os.path.join(outdir, "{}_stars.dat".format(row['hash'])),
                format='ascii')
    gcat_file = os.path.join(outdir, "gcat_map.dat")
    if os.path.exists(gcat_file):
        cat = Table.read(gcat_file, format="ascii")
        for row in cat:
            stars = Table.read(
                os.path.join(outdir, "{}_stars.dat".format(row['hash'])),
                format='ascii')
            stars.sort('MAG_ACA')
            G_CAT_CACHE[("{:.3f}".format(row['ra']),
                         "{:.3f}".format(row['dec']),
                         "{:.3f}".format(row['roll']))] = stars
    g_ri_cat_file = os.path.join(outdir, "g_ri_cat_map.dat")
    if os.path.exists(g_ri_cat_file):
        cat = Table.read(g_ri_cat_file, format="ascii")
        for row in cat:
            stars = Table.read(
                os.path.join(outdir, "{}_stars.dat".format(row['hash'])),
                format='ascii')
            stars.sort('MAG_ACA')
            G_RI_CAT_CACHE[("{:.3f}".format(row['ra']),
                            "{:.3f}".format(row['dec']))] = stars
    start = DateTime(start)
    stop = DateTime(stop)

    # set the agasc proper motion time to be in the middle of the
    # requested cycle
    lts_mid_time = start + (stop - start) / 2
    # Pad the agasc cone extraction radius by a chunk and extra for pointing offsets
    # and SI align
    radius_pad_arcmin = 15 + np.sqrt((3 + (y_offset / 60.))**2
                                     + (3 + (z_offset / 60.))**2)
    radius = 1.5 + radius_pad_arcmin / 60.
    # Get stars in this field
    cone_stars = agasc.get_agasc_cone(ra, dec, radius=radius, date=lts_mid_time)
    # Filter cone stars to just columns we need
    cone_stars = cone_stars[['AGASC_ID', 'RA_PMCORR', 'DEC_PMCORR',
                             'MAG_ACA', 'MAG_ACA_ERR', 'CLASS', 'POS_ERR',
                             'ASPQ1', 'ASPQ2', 'ASPQ3', 'COLOR1']]
    if len(cone_stars) == 0:
        raise ValueError("No stars found in 3 degree radius of {} {}".format(ra, dec))

    # get mag errs once for the field
    #cone_stars['mag_one_sig_err'], cone_stars['mag_one_sig_err2'] = mini_sausage.get_mag_errs(cone_stars)

    # get a list of days
    days = start + np.arange(0, stop - start, daystep)

    all_rolls = {}
    temps = {}
    # loop over them to see which need data
    last_good_pitch = None
    last_good_day = None
    for day in days.date:
        day_pitch = Ska.Sun.pitch(ra, dec, time=day)
        if day_pitch < 46.4 or day_pitch > 170:
            temps["{}".format(day[0:8])] = {
                'day': day[0:8],
                'caldate': DateTime(day).caldate[4:9],
                'pitch': day_pitch,
                'roll_range': np.nan,
                'nom_roll': np.nan,
                'nom_t_ccd': np.nan,
                'nom_acq_tccd': np.nan,
                'nom_gui_tccd': np.nan,
                'best_roll': np.nan,
                'best_t_ccd': np.nan,
                'best_acq_tccd': np.nan,
                'best_gui_tccd': np.nan,
                'nom_acq_hash': '',
                'nom_gui_hash': '',
                'best_acq_hash': '',
                'best_gui_hash': '',
                'comment': ''}
        else:
            temps["{}".format(day[0:8])] = {
                'day': day[0:8],
                'caldate': DateTime(day).caldate[4:9],
                'pitch': day_pitch,
                'roll_range': get_rolldev(day_pitch)}
            last_good_day = day
            last_good_pitch = day_pitch


    if last_good_day is not None:
        # Run the temperature thing once to see if this might be good for all rolls
        r_data_check = get_t_ccd_roll(
            ra, dec, cycle, detector, too, y_offset, z_offset,
            last_good_pitch, time=last_good_day, cone_stars=cone_stars)


    for tday in temps:
        # If this has already been defined/done for this day, continue
        if 'nom_roll' in temps[tday]:
            continue
        # If roll independent copy in the value from that solution, but get nominal roll again
        # for this day
        if r_data_check['roll_indep']:
            nom = r_data_check['nomdata']
            best = r_data_check['bestdata']

            nom_roll = Ska.Sun.nominal_roll(ra, dec, tday)
            temps[tday].update({
                'nom_roll': nom_roll,
                'nom_t_ccd': np.min([nom['guide_tccd'], nom['acq_tccd']]),
                'best_t_ccd': np.min([best['guide_tccd'], best['acq_tccd']]),
                'nom_acq_tccd': nom['acq_tccd'],
                'nom_gui_tccd': nom['guide_tccd'],
                'best_roll': best['roll'],
                'best_acq_tccd': best['acq_tccd'],
                'best_gui_tccd': best['guide_tccd'],
                'nom_acq_hash': hashlib.md5(np.sort(nom['acq_stars']['AGASC_ID'])).hexdigest(),
                'best_acq_hash': hashlib.md5(np.sort(best['acq_stars']['AGASC_ID'])).hexdigest(),
                'nom_gui_hash': hashlib.md5(np.sort(nom['guide_stars']['AGASC_ID'])).hexdigest(),
                'best_gui_hash': hashlib.md5(np.sort(best['guide_stars']['AGASC_ID'])).hexdigest(),
                'comment': r_data_check['comment'],
                })
            all_rolls[nom_roll] = np.min([nom['guide_tccd'], nom['acq_tccd']])
            continue
        t_ccd_roll_data = get_t_ccd_roll(
            ra, dec, cycle, detector, too, y_offset, z_offset,
            temps[tday]['pitch'], time=temps[tday]['day'], cone_stars=cone_stars)
        all_day_rolls = t_ccd_roll_data['rolls']
        all_rolls.update(all_day_rolls)
        cone_stars = t_ccd_roll_data['cone_stars']
        nom = t_ccd_roll_data['nomdata']
        best = t_ccd_roll_data['bestdata']

        for roll, q in zip([nom['roll'], best['roll']], [nom['q_pnt'], best['q_pnt']]):
            attfile = open(os.path.join(outdir, "roll_{:06.2f}.json".format(roll)), 'w')
            attfile.write(json.dumps({'ra': q.ra,
                                      'dec': q.dec,
                                      'roll': q.roll,
                                      'q1': q.q[0],
                                      'q2': q.q[1],
                                      'q3': q.q[2],
                                      'q4': q.q[3]},
                                     indent=4,
                                     sort_keys=True))
            attfile.close()

        temps[tday].update({
                'nom_roll': nom['roll'],
                'nom_t_ccd': np.min([nom['guide_tccd'], nom['acq_tccd']]),
                'best_t_ccd': np.min([best['guide_tccd'], best['acq_tccd']]),
                'nom_acq_tccd': nom['acq_tccd'],
                'nom_gui_tccd': nom['guide_tccd'],
                'best_roll': best['roll'],
                'best_acq_tccd': best['acq_tccd'],
                'best_gui_tccd': best['guide_tccd'],
                'nom_acq_hash': hashlib.md5(np.sort(nom['acq_stars']['AGASC_ID'])).hexdigest(),
                'best_acq_hash': hashlib.md5(np.sort(best['acq_stars']['AGASC_ID'])).hexdigest(),
                'nom_gui_hash': hashlib.md5(np.sort(nom['guide_stars']['AGASC_ID'])).hexdigest(),
                'best_gui_hash': hashlib.md5(np.sort(best['guide_stars']['AGASC_ID'])).hexdigest(),
                'comment': r_data_check['comment'],
                })
    t_ccd_table = Table(temps.values())['day', 'caldate', 'pitch', 'roll_range',
                                        'nom_roll', 'nom_t_ccd', 'nom_acq_tccd', 'nom_gui_tccd',
                                        'best_roll', 'best_t_ccd', 'best_acq_tccd', 'best_gui_tccd',
                                        'nom_acq_hash', 'best_acq_hash',
                                        'nom_gui_hash', 'best_gui_hash',
                                        'comment']

    t_ccd_table.sort('day')
    t_ccd_roll = None
    if len(all_rolls) > 0:
        t_ccd_roll = Table(rows=all_rolls.items(), names=('roll', 't_ccd'))
        t_ccd_roll.sort('roll')

    # Save anything useful and write out catalogs
    t_ccds = [{'key': key, 't_ccd': T_CCD_CACHE[key][0], 'n_acqs': T_CCD_CACHE[key][1]}
              for key in T_CCD_CACHE]
    t_ccd_file = os.path.join(outdir, "t_ccd_map.dat")
    Table(t_ccds).write(t_ccd_file, format='ascii')

    cats = []
    hashes = {}
    for key in CAT_CACHE:
        h = hashlib.md5(np.sort(CAT_CACHE[key]['AGASC_ID'])).hexdigest()
        cats.append({'ra': key[0],
                     'dec': key[1],
                     'roll': key[2],
                     'hash': h})
        hashes[h] = CAT_CACHE[key]
    if len(cats):
        Table(cats)[['ra','dec','roll', 'hash']].write(cat_file, format='ascii')
    ri_cats = []
    for key in RI_CAT_CACHE:
        h = hashlib.md5(np.sort(RI_CAT_CACHE[key]['AGASC_ID'])).hexdigest()
        ri_cats.append({'ra': key[0],
                        'dec': key[1],
                        'hash': h})
        hashes[h] = RI_CAT_CACHE[key]
    if len(ri_cats):
        Table(ri_cats)[['ra', 'dec', 'hash']].write(ri_cat_file, format='ascii')
    gcats = []
    for key in G_CAT_CACHE:
        h = hashlib.md5(np.sort(G_CAT_CACHE[key]['AGASC_ID'])).hexdigest()
        gcats.append({'ra': key[0],
                     'dec': key[1],
                     'roll': key[2],
                     'hash': h})
        hashes[h] = G_CAT_CACHE[key]
    if len(cats):
        Table(gcats)[['ra','dec','roll', 'hash']].write(gcat_file, format='ascii')

    g_ri_cats = []
    for key in G_RI_CAT_CACHE:
        h = hashlib.md5(np.sort(G_RI_CAT_CACHE[key]['AGASC_ID'])).hexdigest()
        g_ri_cats.append({'ra': key[0],
                        'dec': key[1],
                        'hash': h})
        hashes[h] = G_RI_CAT_CACHE[key]
    if len(g_ri_cats):
        Table(g_ri_cats)[['ra', 'dec', 'hash']].write(g_ri_cat_file, format='ascii')

    catcols = ['AGASC_ID', 'RA_PMCORR', 'DEC_PMCORR', 'MAG_ACA', 'MAG_ACA_ERR', 'COLOR1', 'ASPQ1']
    for h in hashes:
        starfile = os.path.join(outdir, "{}_stars.dat".format(h))
        hashes[h][catcols].write(starfile, format='ascii')
        hashes[h][catcols].write(os.path.join(outdir, "{}.html".format(h)),
                        format='jsviewer')

    return t_ccd_table, t_ccd_roll


def plot_time_table(t_ccd_table):
    fig = plt.figure(figsize=(6, 5))
    day_secs = DateTime(t_ccd_table['day']).secs
    nom_t_ccd = t_ccd_table['nom_t_ccd']
    best_t_ccd = t_ccd_table['best_t_ccd']
    plot_cxctime(day_secs,
                 nom_t_ccd,
                 'r')
    plot_cxctime(day_secs,
                 best_t_ccd,
                 'b')
    plot_cxctime(day_secs,
                 nom_t_ccd,
                 'r.',
                 label='nom roll t ccd')
    plot_cxctime(day_secs,
                 best_t_ccd,
                 'b.',
                 label='best roll t ccd')
    plt.grid()
    plt.ylim(ymin=COLD_T_CCD, ymax=WARM_T_CCD + 5.0)
    plt.xlim(xmin=cxctime2plotdate([day_secs[0]]),
             xmax=cxctime2plotdate([day_secs[-1]]))
    plt.legend(loc='upper left', title="", numpoints=1, handlelength=.5)
    plt.ylabel('Max ACA CCD Temp (degC)')
    plt.tight_layout()
    return fig


def plot_hist_table(t_ccd_table):
    fig = plt.figure(figsize=(5, 4))
    bins = np.arange(COLD_T_CCD,
                     WARM_T_CCD + 2.0,
                     1.0)
    plt.hist(t_ccd_table['nom_t_ccd'], bins=bins, color='r', alpha=.5,
             label='nom roll t ccd')
    plt.hist(t_ccd_table['best_t_ccd'], bins=bins, color='b', alpha=.5,
             label='best roll t ccd')
    plt.margins(y=.10)
    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('Max ACA CCD Temp (degC)')
    plt.tight_layout()
    return fig


def check_update_needed(target, obsdir):
    json_parfile = os.path.join(obsdir, 'obsinfo.json')
    parlist = ['ra', 'dec', 'y_offset', 'z_offset', 'report_start', 'report_stop', 'daystep']
    try:
        pars = json.load(open(json_parfile))
        for par in parlist:
            assert np.allclose(pars[par], target[par], atol=1e-10)
        assert pars['chandra_aca'] == target['chandra_aca']
    except:
        return True
    return False


def make_target_report(ra, dec, cycle, detector, too, y_offset, z_offset,
                       start, stop, daystep, obsdir, obsid=None, debug=False, redo=True):
    if not os.path.exists(obsdir):
        os.makedirs(obsdir)
    json_parfile = os.path.join(obsdir, 'obsinfo.json')
    table_file = os.path.join(obsdir, 't_ccd_vs_time.dat')
    just_roll_file = os.path.join(obsdir, 't_ccd_vs_roll.dat')
    if not redo and os.path.exists(table_file) and os.path.exists(just_roll_file):
        t_ccd_table = Table.read(table_file, format='ascii.fixed_width_two_line')
        t_ccd_roll = Table.read(just_roll_file, format='ascii.fixed_width_two_line')
    elif not redo:
        return None
    else:
        t_ccd_table, t_ccd_roll = t_ccd_for_attitude(
            ra, dec,
            cycle, detector, too,
            y_offset, z_offset,
            start=start,
            stop=stop,
            daystep=daystep,
            outdir=obsdir)
        t_ccd_table.write(table_file,
                          format='ascii.fixed_width_two_line')
        if t_ccd_roll is not None:
            t_ccd_roll.write(just_roll_file,
                             format='ascii.fixed_width_two_line')
        parfile = open(json_parfile, 'w')
        parfile.write(json.dumps({'ra': ra, 'dec': dec, 'obsid': obsid,
                                  'y_offset': y_offset, 'z_offset': z_offset,
                                  'report_start': start.secs, 'report_stop': stop.secs,
                                  'daystep': daystep, 'chandra_aca': chandra_aca.__version__},
                                 indent=4,
                                 sort_keys=True))
        parfile.close()

    tfig = plot_time_table(t_ccd_table)
    tfig_html = mpld3.fig_to_html(tfig)
    hfig = plot_hist_table(t_ccd_table)
    hfig.savefig(os.path.join(obsdir,
                              'temperature_hist.png'))
    plt.close(tfig)
    plt.close(hfig)

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.join(TASK_DATA, 'templates')))
    jinja_env.line_comment_prefix = '##'
    jinja_env.line_statement_prefix = '#'
    template = jinja_env.get_template('target.html')
    t_ccd_table['pitch'].format = '.2f'
    t_ccd_table['roll_range'].format = '.1f'
    t_ccd_table['nom_roll'].format = '.2f'
    t_ccd_table['nom_t_ccd'].format = '.2f'
    t_ccd_table['nom_acq_tccd'].format = '.2f'
    t_ccd_table['nom_gui_tccd'].format = '.2f'
    t_ccd_table['best_acq_tccd'].format = '.2f'
    t_ccd_table['best_gui_tccd'].format = '.2f'
    t_ccd_table['best_roll'].format = '.2f'
    t_ccd_table['best_t_ccd'].format = '.2f'
    formats = {'day': '%s',
               'caldate': '%s',
               'pitch': '%5.2f',
               'roll_range': '%3.1f',
               'nom_roll': '%5.2f',
               'nom_t_ccd': '%5.2f',
               'nom_acq_tccd': '%5.2f',
               'nom_gui_tccd': '%5.2f',
               'best_acq_tccd': '%5.2f',
               'best_gui_tccd': '%5.2f',
               'best_roll': '%5.2f',
               'best_t_ccd': '%5.2f',
               'nom_acq_hash': '%s',
               'best_acq_hash': '%s',
               'nom_gui_hash': '%s',
               'best_gui_hash': '%s',
               'comment': '%s'}
    masked_table = t_ccd_table[~np.isnan(t_ccd_table['nom_t_ccd'])]
    displaycols = masked_table.colnames
    if not debug:
        displaycols = ['day', 'caldate', 'pitch', 'roll_range',
                       'nom_roll', 'nom_t_ccd', 'nom_acq_tccd', 'nom_gui_tccd',
                       'best_roll', 'best_t_ccd', 'best_acq_tccd', 'best_gui_tccd', 'comment']
    page = template.render(time_plot=tfig_html,
                           hist_plot='temperature_hist.png',
                           table=masked_table,
                           formats=formats,
                           obsid=obsid,
                           ra=ra,
                           dec=dec,
                           start=start.date,
                           stop=stop.date,
                           displaycols=displaycols,
                           warm_limit=WARM_T_CCD)
    f = open(os.path.join(obsdir, 'index.html'), 'w')
    f.write(page)
    f.close()
    return t_ccd_table


def get_target_report(ra, dec, y_offset, z_offset,
                       start, stop, obsdir, obsid=None, debug=False, redo=True):
    table_file = os.path.join(obsdir, 't_ccd_vs_time.dat')
    just_roll_file = os.path.join(obsdir, 't_ccd_vs_roll.dat')
    if not redo and os.path.exists(table_file):
        t_ccd_table = Table.read(table_file, format='ascii.fixed_width_two_line')
        t_ccd_roll = Table.read(just_roll_file, format='ascii.fixed_width_two_line')
    else:
        return None
    return t_ccd_table



def main():
    """
    Determine required ACA temperature for an attitude over a time range
    """
    opt = get_options()
    t_ccd_table = make_target_report(opt.ra, opt.dec,
                                     cycle=opt.cycle,
                                     detector=opt.detector,
                                     too=opt.too,
                                     y_offset=opt.y_offset,
                                     z_offset=opt.z_offset,
                                     start=DateTime(opt.start),
                                     stop=DateTime(opt.stop),
                                     daystep=opt.daystep,
                                     obsdir=opt.out,
                                     obsid=opt.obsid,
                                     redo=True,
                                     debug=opt.debug,
                                     )


if __name__ == '__main__':
    main()






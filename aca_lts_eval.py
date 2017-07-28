#!/usr/bin/env python

import os
import warnings
# Ignore known numexpr.necompiler and table.conditions warning
warnings.filterwarnings(
    'ignore',
    message="using `oa_ndim == 0` when `op_axes` is NULL is deprecated.*",
    category=DeprecationWarning)


from itertools import count
import numpy as np
import agasc
import jinja2
import re
import shutil
import hashlib
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import mpld3

from Ska.Matplotlib import plot_cxctime, cxctime2plotdate

from Chandra.Time import DateTime
import Ska.Sun
from Ska.quatutil import radec2yagzag
from Quaternion import Quat
import chandra_aca
from chandra_aca import calc_aca_from_targ
from chandra_aca.star_probs import t_ccd_warm_limit, set_acq_model_ms_filter
from chandra_aca.drift import get_aca_offsets, get_target_aimpoint
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
import acq_char
import mini_sausage


PLANNING_LIMIT = -11.5
EDGE_DIST = 30
COLD_T_CCD = -21
WARM_T_CCD = -7
# explicitly disable MS filter
set_acq_model_ms_filter(ms_enabled=False)

TASK_DATA = os.path.join(os.environ['SKA'], 'data', 'aca_lts_eval')
ROLL_TABLE = Table.read(os.path.join(TASK_DATA, 'roll_limits.dat'), format='ascii')

# Save temperature calc a combination of stars
# indexed by hash of agasc ids
T_CCD_CACHE = {}

# Star catalog for an attitude (ignores proper motion)
CAT_CACHE = {}

# Roll independent stars
RI_CAT_CACHE = {}


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
            warm_t_ccd=WARM_T_CCD)
#        print "calc temp for ", id_hash
#    else:
#        print "cached temp for ", id_hash
    return T_CCD_CACHE[id_hash]


def get_rolldev(pitch):
    idx = np.searchsorted(ROLL_TABLE['pitch'], pitch, side='right')
    return ROLL_TABLE['rolldev'][idx - 1]


def select_stars(ra, dec, roll, cone_stars):
    id_key = (ra, dec, roll)
    updated_cone_stars = cone_stars
    if id_key not in CAT_CACHE:
        CAT_CACHE[id_key], updated_cone_stars = mini_sausage.select_acq_stars(
            ra, dec, roll, cone_stars)
    return CAT_CACHE[id_key], updated_cone_stars


def select_ri_stars(ra, dec, cone_stars):
    id_key = (ra, dec)
    updated_cone_stars = cone_stars
    if id_key not in RI_CAT_CACHE:
        RI_CAT_CACHE[id_key], updated_cone_stars = mini_sausage.select_acq_stars(
            ra, dec, None, cone_stars, roll_indep=True)
    return RI_CAT_CACHE[id_key], updated_cone_stars


def get_t_ccd_roll(ra, dec, cycle, detector, too, y_offset, z_offset, pitch, time, cone_stars):
    """
    Loop over possible roll range for this pitch and return best
    and nominal temperature/roll combinations
    """
    best_roll = None
    best_t_ccd = None
    best_stars = None
    best_n_acq = None
    nom_roll = Ska.Sun.nominal_roll(ra, dec, time=time)
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
        roll_indep_stars = select_ri_stars(ra_pnt, dec_pnt, cone_stars)[0]
        ri_t_ccd, ri_t_acq = max_temp(time=time, stars=roll_indep_stars)
        if (ri_t_ccd == WARM_T_CCD):
            nom = (WARM_T_CCD, nom_roll, ri_t_acq, roll_indep_stars)
            return {'nomdata': nom,
                    'bestdata': nom,
                    'rolls': {nom_roll: ri_t_ccd},
                    'cone_stars': cone_stars,
                    'roll_indep': True,
                    'comment': 'roll-independent'}
    nom_stars, updated_cone_stars = select_stars(ra_pnt, dec_pnt, nom_roll, cone_stars)
    cone_stars = updated_cone_stars
    nom_t_ccd, nom_n_acq = max_temp(time=time, stars=nom_stars)
    all_rolls = {nom_roll: nom_t_ccd}
    # if nom_t_ccd is WARM_T_CCD, stop now
    if (nom_t_ccd == WARM_T_CCD):
        nom =  (nom_t_ccd, nom_roll, nom_n_acq, nom_stars)
        best = nom
        return {'nomdata': nom,
                'bestdata': best,
                'rolls': all_rolls,
                'cone_stars': updated_cone_stars,
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
    best_is_max = False
    for roll in off_nom_rolls:
        q_pnt = calc_aca_from_targ((ra, dec, roll),
                                   (y_offset / 60.) + (aca_offset_y / 3600.),
                                   (z_offset / 60.) + (aca_offset_z / 3600.))
        ra_pnt = q_pnt.ra
        dec_pnt = q_pnt.dec
        roll_stars, updated_cone_stars = select_stars(ra_pnt, dec_pnt, roll, cone_stars)
        cone_stars = updated_cone_stars
        roll_t_ccd, roll_n_acq = max_temp(time=time, stars=roll_stars)
        all_rolls[roll] = roll_t_ccd
        if roll_t_ccd is not None:
            if best_t_ccd is None or roll_t_ccd > best_t_ccd:
                best_t_ccd = roll_t_ccd
                best_roll = roll
                best_stars = roll_stars
                best_n_acq = roll_n_acq
                if abs(roll - np.round(nom_roll)) > (roll_dev - d_roll):
                    best_is_max = True
            if (best_t_ccd == WARM_T_CCD):
                break
    nom =  (nom_t_ccd, nom_roll, nom_n_acq, nom_stars)
    best = (best_t_ccd, best_roll, best_n_acq, best_stars)
    comment = ''
    if best_is_max:
        comment = 'best roll at max offset'
    return {'nomdata': nom,
            'bestdata': best,
            'rolls': all_rolls,
            'cone_stars': updated_cone_stars,
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

    start = DateTime(start)
    stop = DateTime(stop)

    # set the agasc proper motion time to be in the middle of the
    # requested cycle
    lts_mid_time = start + (stop - start) / 2

    # Get stars in this field
    print "fetching stars a {} {}".format(ra, dec)
    cone_stars = agasc.get_agasc_cone(ra, dec, radius=3, date=lts_mid_time)
    if len(cone_stars) == 0:
        raise ValueError("No stars found in 3 degree radius of {} {}".format(ra, dec))

    # get mag errs once for the field
    #cone_stars['mag_one_sig_err'], cone_stars['mag_one_sig_err2'] = mini_sausage.get_mag_errs(cone_stars)

    # get a list of days
    days = start + np.arange(0, stop - start, daystep)

    all_rolls = {}
    temps = {}
    roll_indep_data = {}
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
                'nom_roll': np.nan,
                'nom_t_ccd': np.nan,
                'best_roll': np.nan,
                'best_t_ccd': np.nan,
                'nom_id_hash': '',
                'best_id_hash': '',
                'comment': ''}
        else:
            temps["{}".format(day[0:8])] = {
                'day': day[0:8],
                'caldate': DateTime(day).caldate[4:9],
                'pitch': day_pitch}
            last_good_day = day
            last_good_pitch = day_pitch


    if last_good_day is not None:
        # Run the temperature thing once to see if this might be good for all rolls
        r_data_check = get_t_ccd_roll(
            ra, dec, cycle, detector, too, y_offset, z_offset,
            last_good_pitch, time=last_good_day, cone_stars=cone_stars)

        # If it is roll independent, write out the star hashes here
        if r_data_check['roll_indep']:
            nom_t_ccd, nom_roll, nom_n_acq, nom_stars = r_data_check['nomdata']
            best_t_ccd, best_roll, best_n_acq, best_stars = r_data_check['bestdata']
            nom_id_hash = hashlib.md5(np.sort(nom_stars['AGASC_ID'])).hexdigest()
            best_id_hash = hashlib.md5(np.sort(best_stars['AGASC_ID'])).hexdigest()
            if not os.path.exists(os.path.join(outdir, "{}.html".format(nom_id_hash))):
                nom_stars.write(os.path.join(outdir, "{}.html".format(nom_id_hash)),
                                format="jsviewer")
            if not os.path.exists(os.path.join(outdir, "{}.html".format(best_id_hash))):
                best_stars.write(os.path.join(outdir, "{}.html".format(best_id_hash)),
                                format="jsviewer")

    for tday in temps:
        # If this has already been defined/done for this day, continue
        if 'nom_roll' in temps[tday]:
            continue
        # If roll independent copy in the value from that solution
        if r_data_check['roll_indep']:
            nom_t_ccd, nom_roll, nom_n_acq, nom_stars = r_data_check['nomdata']
            best_t_ccd, best_roll, best_n_acq, best_stars = r_data_check['bestdata']
            temps[tday].update({
                'nom_roll': nom_roll,
                'nom_t_ccd': nom_t_ccd,
                'best_roll': best_roll,
                'best_t_ccd': best_t_ccd,
                'nom_id_hash': nom_id_hash,
                'best_id_hash': best_id_hash,
                'comment': r_data_check['comment'],
                })
            continue
        t_ccd_roll_data = get_t_ccd_roll(
            ra, dec, cycle, detector, too, y_offset, z_offset,
            temps[tday]['pitch'], time=temps[tday]['day'], cone_stars=cone_stars)
        all_day_rolls = t_ccd_roll_data['rolls']
        all_rolls.update(all_day_rolls)
        cone_stars = t_ccd_roll_data['cone_stars']
        nom_t_ccd, nom_roll, nom_n_acq, nom_stars = t_ccd_roll_data['nomdata']
        best_t_ccd, best_roll, best_n_acq, best_stars = t_ccd_roll_data['bestdata']
        nom_id_hash = hashlib.md5(np.sort(nom_stars['AGASC_ID'])).hexdigest()
        best_id_hash = hashlib.md5(np.sort(best_stars['AGASC_ID'])).hexdigest()
        if not os.path.exists(os.path.join(outdir, "{}.html".format(nom_id_hash))):
            nom_stars.write(os.path.join(outdir, "{}.html".format(nom_id_hash)),
                            format="jsviewer")
        if not os.path.exists(os.path.join(outdir, "{}.html".format(best_id_hash))):
            best_stars.write(os.path.join(outdir, "{}.html".format(best_id_hash)),
                             format="jsviewer")
        temps[tday].update({
                'nom_roll': nom_roll,
                'nom_t_ccd': nom_t_ccd,
                'best_roll': best_roll,
                'best_t_ccd': best_t_ccd,
                'nom_id_hash': nom_id_hash,
                'best_id_hash': best_id_hash,
                'comment': t_ccd_roll_data['comment']
                })
    t_ccd_table = Table(temps.values())['day', 'caldate', 'pitch',
                                        'nom_roll', 'nom_t_ccd',
                                        'best_roll', 'best_t_ccd',
                                        'nom_id_hash', 'best_id_hash',
                                        'comment']
    t_ccd_table.sort('day')
    t_ccd_roll = None
    if len(all_rolls) > 0:
        t_ccd_roll = Table(rows=all_rolls.items(), names=('roll', 't_ccd'))
        t_ccd_roll.sort('roll')
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
    plt.ylim(ymin=COLD_T_CCD, ymax=WARM_T_CCD + 3.0)
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
                                  'daystep': daystep},
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

    #jinja_env = jinja2.Environment(
    #    loader=jinja2.FileSystemLoader(
    #        os.path.join(os.environ['SKA'], 'data', 'mica', 'templates')))

    #html_table = masked_table.pformat(html=True, max_width=-1, max_lines=-1)
    ## customize table for sorttable
    #html_table[0] = '<table class="sortable" border cellpadding=5>'
    #html_table[1] = re.sub('<th>caldate</th>',
    #                       '<th class="sorttable_nosort">caldate</th>',
    #                       html_table[1])
    if not os.path.exists(os.path.join(obsdir, 'sorttable.js')):
        shutil.copy(os.path.join(TASK_DATA, 'sorttable.js'), obsdir)

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.join(TASK_DATA, 'templates')))
    jinja_env.line_comment_prefix = '##'
    jinja_env.line_statement_prefix = '#'
    template = jinja_env.get_template('target.html')
    t_ccd_table['pitch'].format = '.2f'
    t_ccd_table['nom_roll'].format = '.2f'
    t_ccd_table['nom_t_ccd'].format = '.2f'
    t_ccd_table['best_roll'].format = '.2f'
    t_ccd_table['best_t_ccd'].format = '.2f'
    formats = {'day': '%s',
               'caldate': '%s',
               'pitch': '%5.2f',
               'nom_roll': '%5.2f',
               'nom_t_ccd': '%5.2f',
               'best_roll': '%5.2f',
               'best_t_ccd': '%5.2f',
               'nom_id_hash': '%s',
               'best_id_hash': '%s',
               'comment': '%s'}
    masked_table = t_ccd_table[~np.isnan(t_ccd_table['nom_t_ccd'])]
    displaycols = masked_table.colnames
    if not debug:
        displaycols = ['day', 'caldate', 'pitch',
                       'nom_roll', 'nom_t_ccd',
                       'best_roll', 'best_t_ccd', 'comment']
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






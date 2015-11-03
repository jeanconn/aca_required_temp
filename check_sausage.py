import numpy as np
import re
import mica.starcheck
import agasc
import mini_sausage
from kadi import events
from astropy.table import Table
import aca_required_temp

def check_obs(obsid):
    try:
        sc = mica.starcheck.get_starcheck_catalog(obsid)
    except:
        return None
    if sc['obs'] is None:
        return None
    if 'cat' not in sc:
        return None
    if len(sc['cat']) == 0:
        return None
    ra = sc['obs']['point_ra']
    if ra is None:
        return None
    dec = sc['obs']['point_dec']
    roll = sc['obs']['point_roll']
    cone_stars = agasc.get_agasc_cone(ra, dec, radius=1.2,
                                      date=sc['obs']['mp_starcat_time'],
                                      agasc_file='/proj/sot/ska/jeanproj/git/agasc/miniagasc.h5')
    acqs = sc['cat'][(sc['cat']['type'] == 'BOT') | (sc['cat']['type'] == 'ACQ')]
    acq_pass = [re.sub('g\d{1}', '', ap) for ap in acqs['pass']]
    acq_manual = np.array([re.search('X', ap) is not None for ap in acq_pass])
    acq_pass = [re.sub('gX', '', ap) for ap in acq_pass]
    acq_pass = [re.sub('aX', '', ap) for ap in acq_pass]
    acq_pass = np.array([re.sub('^$', 'a1', ap) for ap in acq_pass])

    mini_sausage.set_dither(np.max([sc['obs']['dither_y_amp'], sc['obs']['dither_z_amp'], 8.0]))
    mini_sausage.set_manvr_error(sc['manvr'][-1]['slew_err_arcsec'])
    select, all_stars = mini_sausage.select_stars(ra, dec, roll, cone_stars)
    all_stars['starcheck'] = False
    for star in acqs:
        all_stars['starcheck'][all_stars['AGASC_ID'] == star['id']] = True
    sc_t_ccd, sc_n_acq = aca_required_temp.max_temp(time=sc['obs']['mp_starcat_time'],
                                                stars=all_stars[all_stars['starcheck']])
    ms_t_ccd, ms_n_acq = aca_required_temp.max_temp(time=sc['obs']['mp_starcat_time'],
                                                stars=select)
    print obsid, "starcheck {}".format(sc_t_ccd), "manual? ", np.any(acq_manual), "mine {}".format(ms_t_ccd)
    obs = {'obsid': obsid,
           'sc_t_ccd': sc_t_ccd,
           'sc_had_manual': np.any(acq_manual),
           'ms_t_ccd': ms_t_ccd,
           'total': len(acqs),
           'new': 0}

    for star in acqs:
        if star['id'] in select['AGASC_ID']:
            obs['new'] += 1
    for p in [1, 2, 3, 4]:
        sc_acqs_for_p = acqs[acq_pass == 'a{}'.format(p)]
        obs['sc_n_{}'.format(p)] = len(sc_acqs_for_p)
        new_acqs_for_p = select[(select['stage'] == p)]
        obs['new_n_{}'.format(p)] = 0
        for star in sc_acqs_for_p:
            if star['id'] in new_acqs_for_p['AGASC_ID']:
                obs['new_n_{}'.format(p)] += 1
        obs['extra_n_{}'.format(p)] = len(new_acqs_for_p) - len(sc_acqs_for_p)
    return obs


check = []

for kadi_obs in events.obsids.filter('2013:001', '2015:100'):
#for obsid in [15057]:
    print kadi_obs
    obsid = kadi_obs.obsid

    check_table = check_obs(obsid)
    if check_table is not None:
        check.append(check_table)


check = Table(check)['obsid',
                     'sc_t_ccd', 'ms_t_ccd',
                     'total', 'new', 'sc_had_manual',
                     'sc_n_1', 'sc_n_2', 'sc_n_3', 'sc_n_4',
                     'new_n_1', 'new_n_2', 'new_n_3', 'new_n_4',
                     'extra_n_1', 'extra_n_2', 'extra_n_3', 'extra_n_4']

check.write("table.dat", format="ascii.tab")




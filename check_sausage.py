import numpy as np
import re
import mica.starcheck
import agasc
import mini_sausage
from kadi import events
from astropy.table import Table
import aca_required_temp

def check_obs_acqs(obsid):
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
                                      agasc_file='/proj/sot/ska/data/agasc/agasc1p6.h5')
    acqs = sc['cat'][(sc['cat']['type'] == 'BOT') | (sc['cat']['type'] == 'ACQ')]
    acq_pass = [re.sub('g\d{1}', '', ap) for ap in acqs['pass']]
    acq_manual = np.array([re.search('X', ap) is not None for ap in acq_pass])
    acq_pass = [re.sub('gX', '', ap) for ap in acq_pass]
    acq_pass = [re.sub('aX', '', ap) for ap in acq_pass]
    acq_pass = [re.sub('--', '', ap) for ap in acq_pass]
    acq_pass = [re.sub('^$', 'a1', ap) for ap in acq_pass]
    acq_pass = np.array([int(re.sub('a', '', ap)) for ap in acq_pass])


    guis = sc['cat'][(sc['cat']['type'] == 'BOT') | (sc['cat']['type'] == 'GUI')]
    gui_pass = [re.sub('a\d{1}', '', ap) for ap in guis['pass']]
    gui_manual = np.array([re.search('X', ap) is not None for ap in gui_pass])
    gui_pass = [re.sub('gX', '', ap) for ap in gui_pass]
    gui_pass = [re.sub('aX', '', ap) for ap in gui_pass]
    gui_pass = [re.sub('--', '', ap) for ap in gui_pass]
    gui_pass = [re.sub('^$', 'g1', ap) for ap in gui_pass]
    gui_pass = np.array([int(re.sub('g', '', ap)) for ap in gui_pass])


    mini_sausage.set_dither(np.max([sc['obs']['dither_y_amp'], sc['obs']['dither_z_amp'], 8.0]))
    mini_sausage.set_manvr_error(sc['manvr'][-1]['slew_err_arcsec'])
    select_acqs, all_stars = mini_sausage.select_acq_stars(ra, dec, roll, n=len(acqs), cone_stars=cone_stars)
    select_guis, all_stars = mini_sausage.select_guide_stars(ra, dec, roll, n=len(guis), cone_stars=all_stars)
    all_stars['starcheck_acq'] = False
    for star in acqs:
        all_stars['starcheck_acq'][all_stars['AGASC_ID'] == star['id']] = True
    all_stars['starcheck_gui'] = False
    for star in guis:
        all_stars['starcheck_gui'][all_stars['AGASC_ID'] == star['id']] = True

    sc_t_ccd, sc_n_acq = aca_required_temp.max_temp(time=sc['obs']['mp_starcat_time'],
                                                stars=all_stars[all_stars['starcheck_acq']])
    ms_t_ccd, ms_n_acq = aca_required_temp.max_temp(time=sc['obs']['mp_starcat_time'],
                                                stars=select_acqs)
    print obsid, "starcheck {}".format(sc_t_ccd), "manual? ", np.any(acq_manual), "mine {}".format(ms_t_ccd)
    obs = {'obsid': obsid,
           'sc_t_ccd': sc_t_ccd,
           'sc_acq_manual': np.any(acq_manual),
           'sc_gui_manual': np.any(gui_manual),
           'ms_t_ccd': ms_t_ccd,
           'acq_total': len(acqs),
           'gui_total': len(guis),
           'acq_match': 0,
           'gui_match': 0,
           'gui_p_match': 0,
           'gui_sc_avgmag': np.mean(guis['mag']),
           'gui_ms_avgmag': np.mean(select_guis['MAG_ACA']),
           'gui_sc_nstage': np.max(gui_pass),
           'gui_extra': 0
           'dither': np.max([sc['obs']['dither_y_amp'], sc['obs']['dither_z_amp'], 8.0]),
           }
    for star in acqs:
        if star['id'] in select_acqs['AGASC_ID']:
            obs['acq_match'] += 1
    for star in guis:
        if star['id'] in select_guis['AGASC_ID']:
            obs['gui_match'] += 1
    for star, p in zip(guis, gui_pass):
        if star['id'] in all_stars[all_stars['Guide_stage'] == int(p)]['AGASC_ID']:
            obs['gui_p_match'] +=1



#    for p in range(1, np.max([np.max(acq_pass), np.max(select_acq['Acq_stage'])]) + 1):
#        sc_acqs_for_p = acqs[acq_pass == p]
#        obs['sc_n_{}'.format(p)] = len(sc_acqs_for_p)
#        new_acqs_for_p = select[(select_acq['Acq_stage'] == p)]
#        obs['new_n_{}'.format(p)] = 0
#        for star in sc_acqs_for_p:
#            if star['id'] in new_acqs_for_p['AGASC_ID']:
#                obs['new_n_{}'.format(p)] += 1
#        obs['extra_n_{}'.format(p)] = len(new_acqs_for_p) - len(sc_acqs_for_p)
#    for p in range(1, np.max([np.max(gui_pass), np.max(select['Acq_stage'])]) + 1):
#        sc_acqs_for_p = acqs[acq_pass == p]
#        obs['sc_n_{}'.format(p)] = len(sc_acqs_for_p)
#        new_acqs_for_p = select[(select['Gui_stage'] == p)]
#        obs['new_n_{}'.format(p)] = 0
#        for star in sc_acqs_for_p:
#            if star['id'] in new_acqs_for_p['AGASC_ID']:
#                obs['new_n_{}'.format(p)] += 1
#        obs['extra_n_{}'.format(p)] = len(new_acqs_for_p) - len(sc_acqs_for_p)
#


    print(obs)
    return obs


check = []

for kadi_obs in events.obsids.filter('2017:110', '2017:200'):
#for obsid in [19864]:
    print kadi_obs
    obsid = kadi_obs.obsid

    check_table = check_obs_acqs(obsid)
    if check_table is not None:
        check.append(check_table)


#check = Table(check)['obsid',
#                     'sc_t_ccd', 'ms_t_ccd',
#                     'total', 'new', 'sc_had_manual',
#                     'sc_n_1', 'sc_n_2', 'sc_n_3', 'sc_n_4',
#                     'new_n_1', 'new_n_2', 'new_n_3', 'new_n_4',
#                     'extra_n_1', 'extra_n_2', 'extra_n_3', 'extra_n_4']
#
#check.write("ntable.dat", format="ascii.tab")
#



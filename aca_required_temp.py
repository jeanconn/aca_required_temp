
import numpy as np

import agasc
from Chandra.Time import DateTime
import Ska.Sun
from Ska.quatutil import radec2yagzag
from Quaternion import Quat
import chandra_aca
from starcheck.star_probs import t_ccd_warm_limit
from astropy.table import Table

N_ACQ_STARS = 5
EDGE_DIST = 30
COLD_T_CCD = -21
WARM_T_CCD = -5

ROLL_TABLE = Table.read('roll_limits.dat', format='ascii')

# Save temperature calc a combination of stars
# indexed by hash of agasc ids
TEMP_CACHE = {}


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description="Get required ACA temp for an attitude over a cycle")
    parser.add_argument("ra",
                        type=float)
    parser.add_argument("dec",
                        type=float)
    parser.add_argument("--out",
                        default="roll_temps.dat")
    parser.add_argument("--start",
                        default="2014-09-01")
    parser.add_argument("--stop",
                        default="2015-12-31")
    opt = parser.parse_args()
    return opt


def get_agasc_cone(ra, dec, time=None, faint_lim=10.8):
    cone_stars = agasc.get_agasc_cone(ra, dec, date=time)
    cols = ['AGASC_ID', 'MAG_ACA', 'COLOR1',
            'RA_PMCORR', 'DEC_PMCORR']
    ok_cone_stars = cone_stars[(cone_stars['MAG_ACA'] < faint_lim)
                               & (cone_stars['CLASS'] == 0)
                               & (cone_stars['ASPQ1'] == 0)
                               & (cone_stars['COLOR1'] != 0.7)][cols]
    ok_cone_stars.sort('MAG_ACA')
    return ok_cone_stars


def select_fov_stars(ra, dec, roll, cone_stars):
    edgepad = EDGE_DIST / 5.
    q = Quat((ra, dec, roll))
    yag, zag = radec2yagzag(cone_stars['RA_PMCORR'], cone_stars['DEC_PMCORR'], q)
    row, col = chandra_aca.yagzag_to_pixels(yag * 3600,
                                            zag * 3600, allow_bad=True)
    stars_in_fov = cone_stars[(row < (512 - edgepad))
                              & (row > (-512 + edgepad))
                              & (col < (512 - edgepad))
                              & (col > (-512 + edgepad))]
    return stars_in_fov


def max_temp(ra, dec, roll, time, cone_stars):
    fov_stars = select_fov_stars(ra, dec, roll, cone_stars)
    if not len(fov_stars):
        return None
    # take the 8 brightest
    stars = fov_stars[0:8]
    id_hash = tuple(stars['AGASC_ID'])
    if id_hash in TEMP_CACHE:
        t_ccd, n_acq = TEMP_CACHE[id_hash]
    else:
        t_ccd, n_acq = t_ccd_warm_limit(date=time,
                                        mags=stars['MAG_ACA'],
                                        colors=stars['COLOR1'],
                                        min_n_acq=N_ACQ_STARS,
                                        cold_t_ccd=COLD_T_CCD,
                                        warm_t_ccd=WARM_T_CCD)
        TEMP_CACHE[id_hash] = (t_ccd, n_acq)
    return t_ccd


def get_rolldev(pitch):
    if pitch < ROLL_TABLE['pitch'][0]:
        return 0
    idx = np.searchsorted(ROLL_TABLE['pitch'], pitch)
    return ROLL_TABLE['rolldev'][idx - 1]


def best_temp_roll(ra, dec, nom_roll, day_pitch, time, cone_stars):
    """
    Loop over possible roll range for this pitch and return best temp/roll
    combination
    """
    best_t_ccd = None
    best_roll = None
    rolldev = get_rolldev(day_pitch)
    # Quat doesn't care about the domain of roll, so I don't have to
    # do anything tricky with 180/360 etc
    for rolldiff in np.arange(0, rolldev, step=.5):
        # Walk out from nom_roll instead of just through the range,
        # and quit if we get one that is at the warm limit
        # That will give us the closest-to-nominal best choice
        for roll in [nom_roll - rolldiff, nom_roll + rolldiff]:
            roll_t_ccd = max_temp(ra, dec, roll, time=time, cone_stars=cone_stars)
            if roll_t_ccd is not None:
                if best_t_ccd is None or roll_t_ccd > best_t_ccd:
                    best_t_ccd = roll_t_ccd
                    best_roll = roll
                if best_t_ccd == WARM_T_CCD:
                    break
    return best_t_ccd, best_roll


def temps_for_attitude(ra, dec, start='2014-09-01', stop='2015-12-31'):
    # reset the caches at every new attitude
    global TEMP_CACHE
    TEMP_CACHE.clear()

    # set the agasc lookup time to be in the middle of the cycle for
    # proper motion correction
    agasc_mid_time = DateTime((DateTime(start).secs + DateTime(stop).secs)
                              / 2).date

    # Get stars in this field
    cone_stars = get_agasc_cone(ra, dec, time=agasc_mid_time)

    # get a list of days
    start = DateTime(start)
    stop = DateTime(stop)
    days = start + np.arange(stop - start)

    temps = {}
    # loop over them
    for day in days.date:
        nom_roll = Ska.Sun.nominal_roll(ra, dec, time=day)
        day_pitch = Ska.Sun.pitch(ra, dec, time=day)
        if day_pitch < 45 or day_pitch > 170:
            continue

        nom_roll_t_ccd = max_temp(ra, dec, nom_roll, time=day, cone_stars=cone_stars)
        # if we can get the nominal roll catalog at warmest temp, why check the rest?
        if nom_roll_t_ccd == WARM_T_CCD:
            best_t_ccd = nom_roll_t_ccd
            best_roll = nom_roll
        else:
            best_t_ccd, best_roll = best_temp_roll(ra, dec, nom_roll,
                                                   day_pitch,
                                                   time=day,
                                                   cone_stars=cone_stars)
        # should we have values or skip entries for None here?
        if best_t_ccd is None:
            continue
        temps["{}".format(day[0:8])] = {
            'day': day,
            'pitch': "{:.2f}".format(day_pitch),
            'nom_roll': "{:.2f}".format(nom_roll),
            'nom_roll_t_ccd': "{:.2f}".format(nom_roll_t_ccd),
            'best_roll': "{:.2f}".format(best_roll),
            'best_t_ccd': "{:.2f}".format(best_t_ccd)}
    table = Table(temps.values())['day', 'pitch',
                                  'nom_roll', 'nom_roll_t_ccd',
                                  'best_roll', 'best_t_ccd']

    table.sort('day')
    return table

def main():
    """
    Determine required ACA temperature for an attitude over a time range
    """
    opt = get_options()
    temp_table = temps_for_attitude(opt.ra, opt.dec,
                                    start=opt.start,
                                    stop=opt.stop)
    temp_table.write(opt.out, format='ascii.fixed_width_two_line')

if __name__ == '__main__':
    main()






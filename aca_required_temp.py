
import numpy as np

import characteristics
import agasc
from Chandra.Time import DateTime
from Ska.Sun import nominal_roll, pitch
from Ska.quatutil import radec2yagzag
from Quaternion import Quat
import chandra_aca
from starcheck.star_probs import t_ccd_warm_limit
import hashlib
from astropy.table import Table

ROLL_TABLE = Table.read('roll_limits.dat', format='ascii')

# Save a field until we move on to the next ra/dec
AGASC_CACHE = {}

# Save temperature calc a combination of stars
# indexed by hash of agasc ids
TEMP_CACHE = {}

AGASC_MID_TIME = None

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
    parser.add_argument("--start-day",
                        default="2014-09-01")
    parser.add_argument("--stop-day",
                        default="2015-12-31")
    opt = parser.parse_args()
    return opt


def get_agasc_cone(ra, dec, time=None, faint_lim=10.8):
    if time is None:
        time = AGASC_MID_TIME
    global AGASC_CACHE
    # using a cache ignores proper motion, but this should be fine
    if "{:.8f}_{:.8f}".format(ra, dec) in AGASC_CACHE:
        field = AGASC_CACHE["{:.8f}_{:.8f}".format(ra, dec)]
    else:
        field = agasc.get_agasc_cone(ra, dec, date=time)
        # just filter in place for now
        field = field[field['MAG_ACA'] < faint_lim]
        field = field[field['CLASS'] == 0]
        field = field[field['ASPQ1'] == 0]
        #field = field[(field['COLOR1'] < 0.699999) | (field['COLOR1'] > 0.700001)]
        field = field[field['COLOR1'] != 0.7]
        field = field[field['COLOR1'] != 1.5]
        field.sort('MAG_ACA')
        AGASC_CACHE["{:.8f}_{:.8f}".format(ra, dec)] = field
    return field


def select_fov_stars(ra, dec, roll, field):
    edgepad = characteristics.EDGE_DIST / 5.
    q = Quat((ra, dec, roll))
    yag, zag = radec2yagzag(field['RA'], field['DEC'], q)
    row, col = chandra_aca.yagzag_to_pixels(yag * 3600,
                                            zag * 3600, allow_bad=True)
    in_fov = field[(row < (512 - edgepad))
                   & (row > (-512 + edgepad))
                   & (col < (512 - edgepad))
                   & (col > (-512 + edgepad))]
    return in_fov


def max_temp(ra, dec, roll, time):
    global TEMP_CACHE
    field = get_agasc_cone(ra, dec)
    fov_stars = select_fov_stars(ra, dec, roll, field)
    if not len(fov_stars):
        return None
    # get brightest 8 in place, these should already be sorted
    if len(fov_stars) > 8:
        fov_stars = fov_stars[0:8]
    id_hash = hashlib.md5(fov_stars['AGASC_ID']).hexdigest()
    if id_hash in TEMP_CACHE:
        t_ccd, n_acq = TEMP_CACHE[id_hash]
    else:
        t_ccd, n_acq = t_ccd_warm_limit(date=time,
                                        mags=fov_stars['MAG_ACA'],
                                        colors=fov_stars['COLOR1'],
                                        min_n_acq=characteristics.N_ACQ_STARS,
                                        cold_t_ccd = characteristics.COLD_T_CCD,
                                        warm_t_ccd = characteristics.WARM_T_CCD)
        TEMP_CACHE[id_hash] = (t_ccd, n_acq)
    return t_ccd


def get_rolldev(pitch):
    if pitch < ROLL_TABLE['pitch'][0]:
        return 0
    idx = np.searchsorted(ROLL_TABLE['pitch'], pitch)
    return ROLL_TABLE['rolldev'][idx - 1]


def best_temp_roll(ra, dec, nom_roll, day_pitch, time):
    best_temp = None
    best_roll = None
    rolldev = get_rolldev(day_pitch)
    # Quat doesn't care about the domain of roll, so I don't have to
    # do anything tricky with 180/360 etc
    for roll in np.arange(nom_roll - rolldev, nom_roll + rolldev, step=.5):
        roll_temp = max_temp(ra, dec, roll, time=time)
        if roll_temp is not None:
            if best_temp is None or roll_temp > best_temp:
                best_temp = roll_temp
                best_roll = roll
    return best_temp, best_roll


def temps_for_attitude(ra, dec, start_day='2014-09-01', stop_day='2015-12-31'):
    # reset the caches at every new attitude
    global TEMP_CACHE
    TEMP_CACHE = {}
    global AGASC_CACHE
    AGASC_CACHE = {}

    # set the agasc lookup time to be in the middle of the cycle for
    # proper motion correction
    global AGASC_MID_TIME
    AGASC_MID_TIME = DateTime((DateTime(start_day).secs + DateTime(stop_day).secs)
                              / 2).date

    # get a list of days
    day = DateTime(start_day)
    days = []
    while day.secs < DateTime(stop_day).secs:
        days.append(day)
        day = day + 1

    temps = {}
    # loop over them
    for day in days:
        nom_roll = nominal_roll(ra, dec, time=day)
        day_pitch = pitch(ra, dec, time=day)
        if day_pitch < 45 or day_pitch > 170:
            continue
        nom_roll_temp = max_temp(ra, dec, nom_roll, time=day)
        # if we can get the nominal roll catalog at warmest temp, why check the rest?
        if nom_roll_temp == characteristics.WARM_T_CCD:
            best_temp = nom_roll_temp
            best_roll = nom_roll
        else:
            best_temp, best_roll = best_temp_roll(ra, dec, nom_roll, day_pitch, time=day)
        # should we have values or skip entries for None here?
        if best_temp is None:
            continue
        temps["{}".format(day.date[0:8])] = {
            'day': day.date,
            'pitch': "{:.2f}".format(day_pitch),
            'nom_roll': "{:.2f}".format(nom_roll),
            'nom_roll_temp': "{:.2f}".format(nom_roll_temp),
            'best_roll': "{:.2f}".format(best_roll),
            'best_temp': "{:.2f}".format(best_temp)}
    table = Table(temps.values())
    # reorder
    return table['day', 'pitch',
                 'nom_roll', 'nom_roll_temp',
                 'best_roll', 'best_temp']


def main():
    """
    Determine required ACA temperature for an attitude over a time range
    """
    opt = get_options()
    temp_table = temps_for_attitude(opt.ra, opt.dec,
                                    start_day=opt.start_day,
                                    stop_day=opt.stop_day)
    temp_table.write(opt.out, format='ascii.tab')

if __name__ == '__main__':
    main()






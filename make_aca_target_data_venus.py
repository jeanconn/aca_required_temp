"""
Convert Horizons ephemeris data for Venus to a form suitable for make_reports.py.
"""
import re
import numpy as np

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from Chandra.Time import DateTime

# ***********************************************************************************************************************
#  Date__(UT)__HR:MN     R.A._(ICRF/J2000.0)_DEC  APmag  S-brt Ang-diam            delta      deldot    S-O-T /r    S-T-O
# ***********************************************************************************************************************
# $$SOE
#  2015-Oct-23 00:00     10 55 30.16 +06 25 01.6  -4.54   1.41 24.97111 0.66830759795952  12.4345850  46.3774 /T  91.4708

lines = [x.strip() for x in open('horizons_results_296-302.txt', 'r')]
l0 = lines.index('$$SOE')
l1 = lines.index('$$EOE')
lines = lines[l0 + 1:l1]  # Just the ephemeris table

colnames = 'date hrmn rah ram ras decd decm decs apmag sbrt angdiam delta deldot sot rr sto'.split()
ephem = ascii.read(lines, format='no_header', names=colnames, guess=False)
ok = ephem['sot'] >= 46.4
ephem = ephem[ok]

caldates = [re.sub(r'-', '', r['date']) + ' at ' + r['hrmn'] + ':00.000' for r in ephem]
date = DateTime(caldates).date
ras = ['{}h{:.0f}m{}s'.format(r['rah'], r['ram'], r['ras']) for r in ephem]
decs = ['{}d{:.0f}m{}s'.format(r['decd'], r['decm'], r['decs']) for r in ephem]
c = SkyCoord(ras, decs)

ra = c.ra
dec = c.dec
n = len(c)
obsid = np.arange(n, dtype=int) + 10000
yoff = np.zeros(n, dtype=float)
zoff = np.zeros(n, dtype=float) - 0.3
simzoff = np.zeros(n, dtype=float)
appdur = np.ones(n, dtype=float) * 6.0  # 6 ksec
si = ['ACIS-I'] * n
grating = ['NONE'] * n

colnames = 'ObsID	RA	Dec	Yoff	Zoff	SIMZoff	AppDur	SI	Grating date'.split()
aca_target = Table([obsid, ra, dec, yoff, zoff, simzoff, appdur, si, grating, date],
                   names=colnames)

ascii.write(aca_target, 'aca_target_data_venus.txt', format='fixed_width_two_line')

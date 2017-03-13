from __future__ import division
from astropy.table import Table
import numpy as np

srcdir = '5year'
targets = Table.read("{}/target_table.dat".format(srcdir), format='ascii')
now_lim = -11.5
future_lim = -5
frac_now_ok = []
frac_5y_ok = []
for t in targets:
    day_data = Table.read("{}/t_ccd_vs_time.dat".format(t['obsdir']),
                          format='ascii')
    good_days = day_data[~np.isnan(day_data['nom_roll'])]
    frac_now_ok.append(np.count_nonzero(good_days['best_t_ccd'] >= now_lim)
                       / len(good_days))
    frac_5y_ok.append(np.count_nonzero(good_days['best_t_ccd'] >= future_lim)
                       / len(good_days))
frac_now_ok = np.array(frac_now_ok)
frac_5y_ok = np.array(frac_5y_ok)
hist(frac_5y_ok, bins=20, alpha=.5, log=True, color='red', label='2022')
hist(frac_now_ok, bins=20, alpha=.5, log=True, color='blue', label='2017')
ylabel("N unique targets")
xlabel("Fraction good catalog days at 'best' roll")
legend(loc='upper left')
grid()


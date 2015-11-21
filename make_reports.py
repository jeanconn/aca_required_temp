import os
import re
import shutil
import time
import numpy as np
import jinja2
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
# Ignore known numexpr.necompiler and table.conditions warning
warnings.filterwarnings(
    'ignore',
    message="using `oa_ndim == 0` when `op_axes` is NULL is deprecated.*",
    category=DeprecationWarning)

from Ska.DBI import DBI
from Ska.Matplotlib import plot_cxctime
from astropy.table import Table
from Chandra.Time import DateTime
from aca_required_temp import check_update_needed, make_target_report

cycle = 17


db = DBI(dbi='sybase', server='sqlsao', database='axafocat', user='aca_ops')
query = """SELECT t.obsid, t.ra, t.dec,
t.y_det_offset as y_offset, t.z_det_offset as z_offset,
t.approved_exposure_time, t.instrument, t.grating, t.obs_ao_str
FROM target t
WHERE
((t.status='unobserved' OR t.status='partially observed' OR t.status='untriggered')
AND NOT(t.ra = 0 AND t.dec = 0)
AND NOT(t.ra IS NULL OR t.dec IS NULL)
AND (t.obs_ao_str <= '{}'))
ORDER BY t.obsid""".format(cycle)

targets = Table(db.fetchall(query))
targets.write('target_table.txt', format='ascii.fixed_width_two_line')

LABEL = 'Outstanding Targets for AO{}'.format(cycle)
OUTDIR = 'out'

PLANNING_LIMIT = -14

stop = DateTime('{}-01-01'.format(2000 + cycle))
start = stop - (365 + 120)

targets['report_start'] = start.secs
targets['report_stop'] = stop.secs

report = []

for t in targets:
    obsdir = os.path.join(OUTDIR, 'obs{:05d}'.format(t['obsid']))
    print t['obsid']
    if not os.path.exists(obsdir):
        os.makedirs(obsdir)
    redo = check_update_needed(t, obsdir)
    t_ccd_table = make_target_report(t['ra'], t['dec'],
                                     t['y_offset'], t['z_offset'],
                                     start=start,
                                     stop=stop,
                                     obsdir=obsdir,
                                     obsid=t['obsid'],
                                     debug=False,
                                     redo=redo)

    report.append({'obsid': t['obsid'],
                   'obsdir': obsdir,
                   'ra': t['ra'],
                   'dec': t['dec'],
                   'max_nom_t_ccd': np.nanmax(t_ccd_table['nom_t_ccd']),
                   'min_nom_t_ccd': np.nanmin(t_ccd_table['nom_t_ccd']),
                   'max_best_t_ccd': np.nanmax(t_ccd_table['best_t_ccd']),
                   'min_best_t_ccd': np.nanmin(t_ccd_table['best_t_ccd']),
                   })

report = Table(report)['obsid', 'obsdir', 'ra', 'dec',
                       'max_nom_t_ccd', 'min_nom_t_ccd',
                       'max_best_t_ccd', 'min_best_t_ccd']
report.sort('min_nom_t_ccd')
formats = {
    'obsid': '%i',
    'obsdir': '%s',
    'ra': '%6.3f',
    'dec': '%6.3f',
    'max_nom_t_ccd': '%5.2f',
    'min_nom_t_ccd': '%5.2f',
    'max_best_t_ccd': '%5.2f',
    'min_best_t_ccd': '%5.2f'}


report.write(os.path.join(OUTDIR, "target_table.dat"),
             format="ascii.fixed_width_two_line")

shutil.copy('sorttable.js', OUTDIR)

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader('templates'))
jinja_env.line_comment_prefix = '##'
jinja_env.line_statement_prefix = '#'
template = jinja_env.get_template('toptable.html')
page = template.render(table=report,
                       formats=formats,
                       planning_limit=PLANNING_LIMIT,
                       start=start.fits,
                       stop=stop.fits,
                       label='ACA Evaluation of Targets')
f = open(os.path.join(OUTDIR, 'index.html'), 'w')
f.write(page)
f.close()




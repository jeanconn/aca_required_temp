#!/usr/bin/env python
import os
import shutil
import numpy as np
import jinja2
import matplotlib
import subprocess
if __name__ == '__main__':
    matplotlib.use('Agg')
import warnings
# Ignore known numexpr.necompiler and table.conditions warning
warnings.filterwarnings(
    'ignore',
    message="using `oa_ndim == 0` when `op_axes` is NULL is deprecated.*",
    category=DeprecationWarning)
from Ska.DBI import DBI
from astropy.table import Table
from Chandra.Time import DateTime
from aca_lts_eval import check_update_needed, make_target_report


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description="Make ACA LTS Eval plots for targets over a cycle")
    parser.add_argument("--out",
                       default="out")
    parser.add_argument("--cycle",
                        default=17)
    parser.add_argument("--planning-limit",
                        default=-14)
    parser.add_argument("--start",
                        help="Start time for roll/temp checks.  Defaults to ~Aug of previous cycle")
    parser.add_argument("--stop",
                        help="Stop time for roll/temp checks.  Default to January end of cycle.")
    opt = parser.parse_args()
    return opt


opt = get_options()
OUTDIR = opt.out
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
CYCLE = opt.cycle
LABEL = 'Outstanding Targets for AO{}'.format(CYCLE)
PLANNING_LIMIT = opt.planning_limit
TASK_DATA = os.path.join(os.environ['SKA'], 'data', 'aca_lts_eval')

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
ORDER BY t.obsid""".format(CYCLE)

targets = Table(db.fetchall(query))
targets.write(os.path.join(OUTDIR, 'requested_targets.txt'),
              format='ascii.fixed_width_two_line')


stop = DateTime('{}-01-01'.format(2000 + CYCLE))
start = stop - (365 + 120)
if opt.start is not None:
    start = DateTime(opt.start)
if opt.stop is not None:
    stop = DateTime(opt.stop)

targets['report_start'] = start.secs
targets['report_stop'] = stop.secs

report = []

for t in targets:
    obsdir = os.path.join(OUTDIR, 'obs{:05d}'.format(t['obsid']))
    if not os.path.exists(obsdir):
        os.makedirs(obsdir)
    redo = check_update_needed(t, obsdir)
    if redo:
        print "Processing {}".format(t['obsid'])
    else:
        print "Skipping {}; processing current".format(t['obsid'])
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

# remove obsdir from the web version of the report
del report['obsdir']


shutil.copy(os.path.join(TASK_DATA, 'sorttable.js'), OUTDIR)

try:
    gitlabel = subprocess.check_output(['git', 'describe', '--always'])
except:
    gitlabel = open(os.path.join(TASK_DATA, 'VERSION')).read().strip()

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.join(TASK_DATA, 'templates')))
jinja_env.line_comment_prefix = '##'
jinja_env.line_statement_prefix = '#'
template = jinja_env.get_template('toptable.html')
page = template.render(table=report,
                       formats=formats,
                       planning_limit=PLANNING_LIMIT,
                       start=start.fits,
                       stop=stop.fits,
                       gitlabel=gitlabel,
                       label='ACA Evaluation of Targets')
f = open(os.path.join(OUTDIR, 'index.html'), 'w')
f.write(page)
f.close()




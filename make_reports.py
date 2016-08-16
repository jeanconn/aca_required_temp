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

RELEASE_VERSION = '1.3.0'


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description="Make ACA LTS Eval plots for targets over a cycle")
    parser.add_argument("--out",
                       default="out")
    parser.add_argument("--cycle",
                        default=18)
    parser.add_argument("--planning-limit",
                        default=-14)
    parser.add_argument("--start",
                        help="Start time for roll/temp checks.  Defaults to ~Aug of previous cycle")
    parser.add_argument("--stop",
                        help="Stop time for roll/temp checks.  Default to March past end of cycle.")
    parser.add_argument("--redo",
                        action='store_true',
                        help="Redo processing even if complete and up-to-date")
    opt = parser.parse_args()
    return opt


opt = get_options()
OUTDIR = opt.out
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
CYCLE = opt.cycle
LABEL = 'Outstanding Targets'
PLANNING_LIMIT = opt.planning_limit
TASK_DATA = os.path.join(os.environ['SKA'], 'data', 'aca_lts_eval')

db = DBI(dbi='sybase', server='sqlsao', database='axafocat', user='aca_ops')
query = """SELECT t.obsid, t.ra, t.dec,
t.y_det_offset as y_offset, t.z_det_offset as z_offset,
t.approved_exposure_time, t.instrument, t.grating, t.obs_ao_str
FROM target t
WHERE
((t.status='unobserved' OR t.status='partially observed' OR t.status='untriggered' OR t.status='scheduled')
AND NOT(t.ra = 0 AND t.dec = 0)
AND NOT(t.ra IS NULL OR t.dec IS NULL))
ORDER BY t.obsid"""

targets = Table(db.fetchall(query))
targets.write(os.path.join(OUTDIR, 'requested_targets.txt'),
              format='ascii.fixed_width_two_line')


stop = DateTime('{}-03-15'.format(2000 + CYCLE))
start = stop - (365 + 210)
if opt.start is not None:
    start = DateTime(opt.start)
if opt.stop is not None:
    stop = DateTime(opt.stop)

targets['report_start'] = start.secs
targets['report_stop'] = stop.secs

last_data_file = os.path.join(OUTDIR, 'target_table.dat')
last_data = None
if os.path.exists(last_data_file):
    last_data = Table.read(last_data_file, format='ascii.fixed_width_two_line')

report = []

print "{} targets with attitudes and unobserved, partially observed, or untriggered status".format(
    len(targets))

no_update_cnt = 0
update_cnt = 0

for t in targets:
    obsdir = os.path.join(OUTDIR, 'obs{:05d}'.format(t['obsid']))
    if not os.path.exists(obsdir):
        os.makedirs(obsdir)
    redo = check_update_needed(t, obsdir) or opt.redo
    # Use "str() not in last_data.astype('str')" because it looks like last_data['obsid']
    # is sometimes an integer column and sometimes a string column.
    if redo or last_data is None or str(t['obsid']) not in last_data['obsid'].astype('str'):
        update_cnt += 1
        print "Processing {}".format(t['obsid'])
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
                       'y_offset': t['y_offset'],
                       'z_offset': t['z_offset'],
                       'max_nom_t_ccd': np.nanmax(t_ccd_table['nom_t_ccd']),
                       'min_nom_t_ccd': np.nanmin(t_ccd_table['nom_t_ccd']),
                       'max_best_t_ccd': np.nanmax(t_ccd_table['best_t_ccd']),
                       'min_best_t_ccd': np.nanmin(t_ccd_table['best_t_ccd']),
                       })

    else:
        no_update_cnt += 1
        previous_record = last_data[last_data['obsid'] == t['obsid']][0]
        report.append({'obsid': t['obsid'],
                       'obsdir': obsdir,
                       'ra': t['ra'],
                       'dec': t['dec'],
                       'y_offset': t['y_offset'],
                       'z_offset': t['z_offset'],
                       'max_nom_t_ccd': previous_record['max_nom_t_ccd'],
                       'min_nom_t_ccd': previous_record['min_nom_t_ccd'],
                       'max_best_t_ccd': previous_record['max_best_t_ccd'],
                       'min_best_t_ccd': previous_record['min_best_t_ccd'],
                       })


print "Processed {} targets".format(update_cnt)
print "Skipped {} targets already up-to-date".format(no_update_cnt)


report = Table(report)['obsid', 'obsdir', 'ra', 'dec', 'y_offset', 'z_offset',
                       'max_nom_t_ccd', 'min_nom_t_ccd',
                       'max_best_t_ccd', 'min_best_t_ccd']
report.sort('min_nom_t_ccd')
formats = {
    'obsid': '%i',
    'obsdir': '%s',
    'ra': '%6.3f',
    'dec': '%6.3f',
    'y_offset': '%5.2f',
    'z_offset': '%5.2f',
    'max_nom_t_ccd': '%5.2f',
    'min_nom_t_ccd': '%5.2f',
    'max_best_t_ccd': '%5.2f',
    'min_best_t_ccd': '%5.2f'}


report.write(os.path.join(OUTDIR, "target_table.dat"),
             format="ascii.fixed_width_two_line")

# remove obsdir from the web version of the report
del report['obsdir']


if not os.path.exists(os.path.join(OUTDIR, 'sorttable.js')):
    shutil.copy(os.path.join(TASK_DATA, 'sorttable.js'), OUTDIR)

try:
    gitlabel = subprocess.check_output(['git', 'describe', '--always'], stderr=subprocess.PIPE)
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
                       release=RELEASE_VERSION,
                       label='ACA Evaluation of Targets')
f = open(os.path.join(OUTDIR, 'index.html'), 'w')
f.write(page)
f.close()




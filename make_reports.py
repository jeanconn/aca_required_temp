#!/usr/bin/env python
import os
import shutil
import numpy as np
import jinja2
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import subprocess
from Ska.DBI import DBI
from astropy.table import Table
from Chandra.Time import DateTime
from aca_lts_eval import check_update_needed, make_target_report
import chandra_aca

import warnings
# Ignore known numexpr.necompiler and table.conditions warning
warnings.filterwarnings(
    'ignore',
    message="using `oa_ndim == 0` when `op_axes` is NULL is deprecated.*",
    category=DeprecationWarning)

RELEASE_VERSION = '2.4.0'

def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description="Make ACA LTS Eval plots for targets over a cycle")
    parser.add_argument("--out",
                       default="out")
    parser.add_argument("--cycle",
                        default=20)
    parser.add_argument("--planning-limit",
                        default=-10.2,
                        type=float)
    parser.add_argument("--start",
                        help="Start time for roll/temp checks.  Defaults to ~Aug of previous cycle")
    parser.add_argument("--stop",
                        help="Stop time for roll/temp checks.  Default to March past end of cycle.")
    parser.add_argument("--daystep",
                        default=1,
                        type=int)
    parser.add_argument("--obsid-file",
                        help="File with list of obsids to process")
    parser.add_argument("--redo",
                        action='store_true',
                        help="Redo processing even if complete and up-to-date")
    parser.add_argument("--incremental",
                       action='store_true',
                       help="Write out table as processed (good for recovery of long processing)")
    parser.add_argument("--only-existing",
                       action="store_true")
    parser.add_argument("--creep",
                        action="store_true")
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
MANVR_ERROR = 10 if opt.creep else 60



query = """SELECT t.obsid, t.ra, t.dec,
t.type, t.y_det_offset as y_offset, t.z_det_offset as z_offset, 
t.approved_exposure_time, t.instrument, t.grating, t.dither_flag, t.obs_ao_str, p.ao_str
FROM target t
right join prop_info p on t.ocat_propid = p.ocat_propid
WHERE
((t.status='unobserved' OR t.status='partially observed' OR t.status='untriggered' OR t.status='scheduled')
AND NOT(t.ra = 0 AND t.dec = 0)
AND NOT(t.ra IS NULL OR t.dec IS NULL))
ORDER BY t.obsid"""

with DBI(dbi='sybase', server='sqlsao', database='axafocat', user='aca_ops') as db:
    targets = Table(db.fetchall(query))

# Get actual dither from the database
dither_column = targets['dither_flag']
targets.remove_column('dither_flag')
dither_y = np.zeros(len(targets))
dither_z = np.zeros(len(targets))
dither_y[(targets['instrument'] == 'ACIS-S') | (targets['instrument'] == 'ACIS-I')] = 8
dither_z[(targets['instrument'] == 'ACIS-S') | (targets['instrument'] == 'ACIS-I')] = 8
dither_y[(targets['instrument'] == 'HRC-S') | (targets['instrument'] == 'HRC-I')] = 20
dither_z[(targets['instrument'] == 'HRC-S') | (targets['instrument'] == 'HRC-I')] = 20
# Mark those with non-default dither
custom_dither_obsids = targets['obsid'][dither_column == 'Y']
# Fetch custom dither from the database for the observations with custom dither
with DBI(dbi='sybase', server='sqlsao', database='axafocat', user='aca_ops') as db:
    for obs in custom_dither_obsids:
        query = "SELECT * from dither where obsid = {}".format(obs)
        dither = db.fetchone(query)
        for ax in ['y_amp', 'z_amp']:
            if dither[ax] is None:
                dither[ax] = 0
        dither_y[targets['obsid'] == obs] = dither['y_amp'] * 3600
        dither_z[targets['obsid'] == obs] = dither['z_amp'] * 3600
targets['dither_y'] = dither_y
targets['dither_z'] = dither_z

targets.write(os.path.join(OUTDIR, 'requested_targets.txt'),
              format='ascii.fixed_width_two_line')


if opt.obsid_file is not None:
    targets['requested'] = False
    obsids = Table.read(opt.obsid_file, format='ascii')['obsid']
    for obsid in obsids:
        targets['requested'][targets['obsid'] == int(obsid)] = True
    targets = targets[targets['requested'] == True]


stop = DateTime('{}-04-01'.format(2000 + CYCLE))
start = stop - (365 + 270)
if opt.start is not None:
    start = DateTime(opt.start)
if opt.stop is not None:
    stop = DateTime(opt.stop)

targets['report_start'] = start.secs
targets['report_stop'] = stop.secs
targets['daystep'] = opt.daystep
targets['chandra_aca'] = chandra_aca.__version__

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
    if not os.path.exists(obsdir) and opt.only_existing == False:
        os.makedirs(obsdir)
    redo = check_update_needed(t, obsdir) or opt.redo
    # Skip it if it really needs to be redone but we only want existing records
    if redo and opt.only_existing:
        continue
    t_dy = t['dither_y']
    t_dz = t['dither_z']
    if opt.creep:
        if t['dither_y'] == 8 and t['dither_z'] == 8:
            t_dy = 4
            t_dz = 4
        else:
            print("Using creep, but not using reduced dither on {} with dither {} {}".format(t['obsid'], t['dither_y'], t['dither_z']))
    # Use "str() not in last_data.astype('str')" because it looks like last_data['obsid']
    # is sometimes an integer column and sometimes a string column.
    if redo or last_data is None or str(t['obsid']) not in last_data['obsid'].astype('str') or opt.only_existing:
        update_cnt += 1
        print "Processing {}".format(t['obsid'])
        t_ccd_table = make_target_report(t['ra'], t['dec'],
                                         int(t['ao_str']),
                                         t['instrument'],
                                         (t['type'] == 'DDT') or (t['type'] == 'TOO'),
                                         t['y_offset'], t['z_offset'],
                                         dither_y=t_dy, dither_z=t_dz,
                                         manvr_error=MANVR_ERROR,
                                         start=start,
                                         stop=stop,
                                         daystep=opt.daystep,
                                         obsdir=obsdir,
                                         obsid=t['obsid'],
                                         debug=False,
                                         redo=redo)
        if t_ccd_table is not None:
            nom = t_ccd_table['nom_t_ccd'][~np.isnan(t_ccd_table['nom_t_ccd'])]
            best = t_ccd_table['best_t_ccd'][~np.isnan(t_ccd_table['best_t_ccd'])]
            if len(nom) == 0:
                frac_nom_ok = 0
            else:
                frac_nom_ok = np.count_nonzero(nom >= PLANNING_LIMIT) * 1.0 / len(nom)
            if len(best) == 0:
                frac_best_ok = 0
            else:
                frac_best_ok = np.count_nonzero(best >= PLANNING_LIMIT) * 1.0 / len(best)
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
                           'frac_nom_ok': frac_nom_ok,
                           'frac_best_ok': frac_best_ok,
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
                       'frac_nom_ok': previous_record['frac_nom_ok'],
                       'frac_best_ok': previous_record['frac_best_ok'],

                       })


    if opt.incremental:
        # Write out the text file on every loop/target if incremental option set
        report_table = Table(report)['obsid', 'obsdir', 'ra', 'dec', 'y_offset', 'z_offset',
                                     'max_nom_t_ccd', 'min_nom_t_ccd',
                                     'max_best_t_ccd', 'min_best_t_ccd', 'frac_nom_ok', 'frac_best_ok']
        report_table.sort('frac_best_ok')
        report_table.write(os.path.join(OUTDIR, "target_table.dat"),
                           format="ascii.fixed_width_two_line")


report_table = Table(report)['obsid', 'obsdir', 'ra', 'dec', 'y_offset', 'z_offset',
                             'max_nom_t_ccd', 'min_nom_t_ccd',
                             'max_best_t_ccd', 'min_best_t_ccd', 'frac_nom_ok', 'frac_best_ok']
report_table.sort('frac_best_ok')
report_table.write(os.path.join(OUTDIR, "target_table.dat"),
                   format="ascii.fixed_width_two_line")

print "Processed {} targets".format(update_cnt)
print "Skipped {} targets already up-to-date".format(no_update_cnt)


# remove obsdir from the web version of the report
del report_table['obsdir']


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
    'min_best_t_ccd': '%5.2f',
    'frac_nom_ok': '%7.4f',
    'frac_best_ok': '%7.4f'}

page = template.render(table=report_table,
                       formats=formats,
                       planning_limit=PLANNING_LIMIT,
                       start=start.fits,
                       stop=stop.fits,
                       gitlabel=gitlabel,
                       chandra_aca=chandra_aca.__version__,
                       release=RELEASE_VERSION,
                       label='ACA Evaluation of Targets')
f = open(os.path.join(OUTDIR, 'index.html'), 'w')
f.write(page)
f.close()




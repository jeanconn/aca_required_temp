#!/usr/bin/env python
import os
import numpy as np
from Ska.DBI import DBI
from astropy.table import Table
from Chandra.Time import DateTime
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky


N_FILES = 6


db = DBI(dbi='sybase', server='sqlsao', database='axafocat', user='aca_ops')
query = """SELECT t.obsid, t.ra, t.dec,
t.type, t.y_det_offset as y_offset, t.z_det_offset as z_offset, 
t.approved_exposure_time, t.instrument, t.grating, t.obs_ao_str, p.ao_str
FROM target t
right join prop_info p on t.ocat_propid = p.ocat_propid
WHERE
(t.soe_st_sched_date is not null
AND t.soe_st_sched_date >= '2011'
AND NOT(t.ra = 0 AND t.dec = 0)
AND NOT(t.ra IS NULL OR t.dec IS NULL))
ORDER BY t.obsid"""

targets = Table(db.fetchall(query))
db.conn.close()
del db

# Remove target attitudes within 1 arcmin
targ_coord = SkyCoord(targets['ra'], targets['dec'], unit='deg')
# Use search_around_sky to get matches from the list into itself
idx1, idx2, dist, dist3d = search_around_sky(targ_coord, targ_coord, seplimit=1 * u.arcmin)
# Exactly matching oneself is not a duplicate
itself = idx1 == idx2
# In the indices into each list, if the index in idx1 is greater than the index in idx2
# the target attitude is the 2nd or nth occurrence of the attitudes and is a duplicate
dups = idx1[~itself][idx1[~itself] > idx2[~itself]]
targets['duplicate'] = False
targets['duplicate'][dups] = True
targets = targets[targets['duplicate'] == False]

# I don't see an easy way to split a table so, split an index array
idx_split = np.array_split(np.arange(len(targets)), N_FILES)
for i, s in enumerate(idx_split):
    targets[s].write("obsids_{}.dat".format(i), format='ascii')





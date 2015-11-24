# Set the task name
TASK = aca_lts_eval

# Uncomment the correct choice indicating either SKA or TST flight environment
FLIGHT_ENV = SKA

# Set the names of all files that get installed
SHARE = make_reports.py aca_lts_eval aca_lts_eval.py acq_char.py mini_sausage.py 
TEMPLATES = templates/target.html templates/toptable.html
DATA = task_schedule.cfg roll_limits.dat sorttable.js

include /proj/sot/ska/include/Makefile.FLIGHT

install:
#  Uncomment the lines which apply for this task
	mkdir -p $(INSTALL_SHARE)
	mkdir -p $(INSTALL_DATA)
	mkdir -p $(INSTALL_DATA)/templates
	rsync --times --cvs-exclude $(SHARE) $(INSTALL_SHARE)/
	rsync --times --cvs-exclude $(DATA) $(INSTALL_DATA)/
	rsync --times --cvs-exclude $(TEMPLATES) $(INSTALL_DATA)/templates/

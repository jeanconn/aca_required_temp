ACA Required Temperature
------------------------

For any RA, Dec we introduce the concept of an "ACA Required Temperature" which is the
maximum ACA CCD temperature at which we expect the given pointing can be acquired and
successfully tracked.  This temperature is a function of the available stars and their
magnitudes for a RA, Dec, and time in the ACA frame.

Star requirements
-----------------
Class == 0
ASPQ1 == 0
Color != 0.700

Acquisition Requirements
------------------------

N Stars == 5
-15C Mag Limit == 10.0
At least 30 arcsecs from FOV chip edge


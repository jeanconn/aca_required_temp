import numpy as np
import copy

#  022  03/12/15  M.Baski    Updated Bad Stars per list from Jean for
#                            DDTS OCCcm10550.

revision = '022'

General = {}

#Pixel to arcsecond conversion
General['Pix2Arc'] = 4.96289

#Default star search area and cutoff magnitude
General['CatalogRadius'] = 1.1       #get stars over this radius for search (deg)
General['CatalogMinMag'] = 14.5     #stars dimmer than this value are ignored

#initial maneuver attitude uncertainty and search box error margin
General['ManvrErrorSigmaInit']       = np.array([10.0, 0.1, 0.1]) * np.pi / 180 / 3600
General['ManvrErrorSearchBoxMargin'] = 20.0 * np.pi / 180 / 3600

#%bad star list
General['BadStarList'] = [  '36178592',
                         '39980640',
                        '185871616',
                        '188751528',
                        '190977856',
                        '260968880',
                        '260972216',
                        '261621080',
                        '296753512',
                        '300948368',
                        '301078152',
                        '301080376',
                        '301465776',
                        '335025128',
                        '335028096',
                        '414324824',
                        '444743456',
                        '465456712',
                        '490220520',
                        '502793400',
                        '509225640',
                        '570033768',
                        '614606480',
                        '637144600',
                        '647632648',
                        '650249416',
                        '656409216',
                        '690625776',
                        '692724384',
                        '788418168',
                        '849226688',
                        '956175008',
                        '989598624',
                       '1004817824',
                       '1016736608',
                       '1044122248',
                       '1117787424',
                       '1130635848',
                       '1130649544',
                       '1161827976',
                       '1196953168',
                       '1197635184']

#Min commandable magnitude
General['ACAFaintCommandLimit'] = 13.98

#Minimum commandable error bar (command must be
#at least MAG_ACA +- this value
General['minCmdMagErrorBar'] = 1.5

#Star acquisition time used for scheduling
General['StarAcqTime'] = 270

# Image Size for ERs (defined here as attitudes with no fid lights requested)
#  choose 0 for 4x4, 1 for 6x6 and 2 for 8x8
General['ERImageSize'] = 2


#%-----------------------------------------------------------------
#%General Star Selection Settings
#%-----------------------------------------------------------------

#%First create a general set of Acq/Guide/Fid settings called
#%'Star'.  This is where the default parameters for each stage
#%are defined.  Once Star is constructed, The Acq/Guide/Fid fields
#%are created by making a copy of 'Star', and then modifying
#%or appending as required.  Don't confuse the generic search
#%settings 'Star' with the ultimate output 'Stars'

Star = {}
Star['SearchSettings'] = {}
Star['SearchSettings']['DoColumnRegisterCheck'] = 1
#%Star.SearchSettings.NMinCand              = 9;
Star['SearchSettings']['DoBminusVcheck'] = 1

#%-----------------------------------------------------------------
#%Inertial Checks
#%-----------------------------------------------------------------

#%Define some useful parameters here.
Star['Inertial'] = {}
Star['Inertial']['MagLimit'] = [5.8, 10.2] #%in magnitude units
Star['Inertial']['MagErrorTol'] = 100	 #%in 0.01 mag increments
Star['Inertial']['PosErrorTol'] = 3000	 #%in milli arc seconds
Star['Inertial']['ASPQ1Lim']    = [0, 0]	 #%ASPQ1 must be in this range (magnitude spoilers)
Star['Inertial']['ASPQ2Lim']	= [0, 0]	 #%ASPQ2 must be in this range (proper motion spoilers)
Star['Inertial']['ASPQ3Lim']	= [0, 999]	 #%ASPQ3 must be in this range (neighbor with undef pos error)
Star['Inertial']['VARIABLELim'] = -9999	 #%VARIABLE must be no bigger than this (variable magnitude marker: -9999 = not variable)
Star['Inertial']['MagErrRand'] = 0.26	 #%Random Magnitude Error in magnitude units
Star['Inertial']['MagErrSyst'] = 0	         #%Systematic Error - This value is a linear expansion of the mag error in BOTH directions, A Positive value INCREASES the magnitude error
Star['Inertial']['MaxMagError'] = 1.5	 #%magnitude errors are limited to this value


#%-----------------------------------------------------------------
#%Roll and Camera Dependent Checks
#%-----------------------------------------------------------------

Star['Body'] = {}
Star['Body']['Column'] = {}
Star['Body']['Column']['MagDiff'] = 5
Star['Body']['Column']['Separation'] = 4

Star['Body']['Register'] = {}
Star['Body']['Register']['MagDiff'] = 5
Star['Body']['Register']['Separation'] = 4
Star['Body']['Register']['Width']   = 2

#%Define some of the boundaries of the chips
Star['Body']['Pixels'] = {}
Star['Body']['Pixels']['ZPixLim'] = [-512.5, 511.5]						#%+- extent in Z direction
Star['Body']['Pixels']['YPixLim'] = [-512.5, 511.5]						#%+- extent in Y direction
Star['Body']['Pixels']['Center']  = [-0.5, -0.5]							#%Y-Z position of quadrants joint
Star['Body']['Pixels']['EdgeBuffer'] = 5

Star['Body']['Traps'] = {}
Star['Body']['Traps']['Column']  = [347]							#%trap positions in pixels (for geometric check)
Star['Body']['Traps']['Row'] = [-374]
Star['Body']['Traps']['DeltaColumn'] = [3]
Star['Body']['Traps']['ExclusionZone'] = {}
Star['Body']['Traps']['ExclusionZone']['Neg'] = [-6, -2]
Star['Body']['Traps']['ExclusionZone']['Pos'] = [3, 7]

#%Stars this many pixels from the edge cannot be candidates
#%ie, shrinks the effective search area for candidates                                                        
Star['Body']['FOV'] = {}
Star['Body']['FOV']['YArcSecLim']    = [-2410, 2473]                       #%+- limits on star positions in arcseconds.
Star['Body']['FOV']['ZArcSecLim']    = [-2504, 2450]                       #%set to [-Inf +Inf to ignore this check]



#%(1=y-pixel min, 2=y-pixel max, 3=z-pixel min, 4=z-pixel max)
#%QUAD BOUNDARIES MUST BE FIRST TWO ENTRIES

Star['Body']['Pixels']['BadPixels'] = np.array([
[ -512,  511,   -1,    0],
[   -1,    0, -512,  511],
[ -245,    0,  454,  454],
[ -512, -512, -512, -512],
[ -462, -462, -409, -409],
[ -382, -382,  328,  328],
[ -314, -314,  328,  328],
[ -302, -302,  237,  237],
[ -301, -301,  391,  391],
[ -255, -255, -251, -251],
[ -191, -191, -326, -326],
[  -44,  -44, -478, -478],
[  -20,  -20, -493, -493],
[   12,   12,  -68,  -68],
[   57,   57,  118,  118],
[  138,  138,    9,    9],
[  152,  152,  496,  496],
[  186,  186, -491, -491],
[  219,  219,   47,   47],
[  253,  253,  365,  365],
[  269,  269,  330,  330],
[  281,  281,  241,  241],
[  285,  285,   99,   99],
[  318,  318,  505,  505],
[  395,  395, -197, -197],
[  402,  402, -330, -330],
[  509,  509, -325, -325]])


#%-----------------------------------------------------------------
#%Spoiler Parameters
#%-----------------------------------------------------------------

Star['Spoiler'] = {}
Star['Spoiler']['MinSep'] = 7		      #%minimum spoiler separation (pixels)
Star['Spoiler']['MaxSep'] = 11		#  	  %maximum search box size (pixels)
Star['Spoiler']['Intercept'] = 9		#	  %0 magnitude difference intercept (pixels)
Star['Spoiler']['Slope'] = 1/2		#	  %spoiler slope boundary
Star['Spoiler']['SigErrMultiplier'] = 3    #%multiplier for 1-sigma error calculations

Star['Spoiler']['MagDiffLimit']     = 6 #%Stars this much dimmer can't ever be spoilers
                                      #%-Inf means all stars will be considered
                                      #%This value is checked against the difference
                                      #%in nominal magnitudes, regardless of errors

#%-----------------------------------------------------------------
#%Acq Selection Values
#%-----------------------------------------------------------------

#%Make a copy of the generic settings, and tack on 
#%a few Acq specific settings

Acq = copy.deepcopy(Star)
Acq['Select'] = {}
Acq['Select']['MaxSearchBox'] = [120, 140, 160, 180, 200, 220, 240] #%maximum search box size (arcsec)
                                         #%smallest valid value is chosen
                                         #%given maneuver/dither error
Acq['Select']['MinSearchBox'] = 25			 #%maximum search box size (arcsec)
Acq['Select']['NMaxSelect']   = 8			 #%maximum acquisition stars
Acq['Select']['nSurplus'] = 0

#%-----------------------------------------------------------------
#%Acq Staging Values 
#%-----------------------------------------------------------------

#%Now define the search stages.  Each stage is 
#%just a copy of the one before it, plus a few
#%further modifications.  There is no limit to
#%the number of search stages (i.e., the # of 
#%search stages is DEFINED as length(Acq)

#%Stage 1
Acq['Body']['Pixels']['BadPixels'] = Acq['Body']['Pixels']['BadPixels'][2:]      #%no quad boundaries for Acq stars
Acq['Stage'] = 1

#%Stage 2
Acq2 = copy.deepcopy(Acq)
Acq2['Inertial']['MaxMagError'] = 1
Acq2['Spoiler']['SigErrMultiplier'] = 1
Acq2['SearchSettings']['DoColumnRegisterCheck'] = 0
Acq2['Inertial']['MagLimit']      = [5.8, 10.5]
Acq2['Stage'] = 2

#%Stage 3
Acq3 = copy.deepcopy(Acq2)
Acq3['Spoiler']['SigErrMultiplier'] = 0
Acq3['Body']['Pixels']['BadPixels'] = []
Acq3['Inertial']['MagLimit']      = [5.8, 10.6]
Acq3['Stage'] = 3

#%Stage 4
Acq4 = copy.deepcopy(Acq3)
Acq4['SearchSettings']['DoBminusVcheck'] = 0
Acq4['Inertial']['MagLimit']      = [5.8, 11.0]
Acq4['Stage'] = 4

#%-----------------------------------------------------------------
#%Guide Selection values
#%-----------------------------------------------------------------
#
#%Make a copy of the generic settings, and tack on 
#%a few Guide specific settings
#
#Guide = Star;
#Guide.Select.NMaxSelect    = 8;					    %maximum guide stars
#
#Guide.Select.MaxSearchBox  = 25;		            %guide search boxes are always this size
#Guide.Select.MinSearchBox  = 25;		            %maximum search box size (arcsec)
#Guide.Select.LeverArm      = 1.*pi/180;			    %Lever arm factor for guide star optimization
#Guide.Select.NDirectSearch = nchoosek(12,5);        %maximum combinations for direct search
#Guide.Select.C_10          = 1444.0;                %ODB_CO_MAG_10
#Guide.Select.CCDIntTime    = 1.4;                   %?????
#Guide.Select.Sig_P1        = 16.2;                  %ODB_SIGMA_P1
#Guide.Select.Sig_P2        = 0.5;                   %ODB_SIGMA_P2
#
#%Guide.SearchSettings.NMinCand  = 6;                 %search staging proceeds until this many found
#Guide.Select.nSurplus = 1;
#
#%-----------------------------------------------------------------
#%Guide Staging Values 
#%-----------------------------------------------------------------
#
#%Now define the search stages.  Each stage is 
#%just a copy of the one before it, plus a few
#%further modifications.  There is no limit to
#%the number of search stages (i.e., the # of 
#%search stages is DEFINED as length(Guide)
#
#Guide(2) = Guide(1);
#Guide(2).Inertial.MagErrRand = 0.15;
#Guide(2).Spoiler.SigErrMultiplier = 2;
#Guide(2).Inertial.MaxMagError      = 1;
#
#Guide(3) = Guide(2);
#Guide(3).Inertial.MagErrRand = 0.0;
#Guide(3).Spoiler.SigErrMultiplier = 1;
#Guide(3).Inertial.MaxMagError      = 0.5;
#Guide(3).Inertial.MagLimit      = [5.8 10.5];
#
#Guide(4) = Guide(3);
#Guide(4).Spoiler.SigErrMultiplier = 0;
#Guide(4).Inertial.MagLimit      = [5.8 10.6];
#
#Guide(5) = Guide(4);
#Guide(5).SearchSettings.DoBminusVcheck = 0;
#Guide(5).Inertial.MagLimit      = [5.8 10.8]; 
#
#%-----------------------------------------------------------------
#%FID Selection values
#%-----------------------------------------------------------------
#
#%Make a copy of the generic settings, and tack on 
#%a few Fid specific settings
#
#Fid = Star;
#Fid.Select.SearchBox      = 25;     %Fid search box size in arcseconds
#Fid.Select.MagBrightLimit = 5.8;    %Max magnitude for ACA catalog
#Fid.Select.MagFaintLimit  = 8;      %Min magnitude for ACA catalog
#
#Fid.Spoiler.MinSep = 5;		        %Anything less than 5 pix, rejected
#Fid.Spoiler.MaxSep = 5;		        %anything more than 5 pix, ok
#Fid.Spoiler.Intercept = 5;			 
#Fid.Spoiler.Slope = 1/2;			 
#Fid.Spoiler.SigErrMultiplier = 3;  
#Fid.Spoiler.MagDiffLimit     = -5; 
#Fid.SearchSettings.DoGeometricTrapCheck=1;
#%Now define the search stages.  Each stage is 
#%just a copy of the one before it, plus a few
#%further modifications.  There is no limit to
#%the number of search stages (i.e., the # of 
#%search stages is DEFINED as length(Fid)
#
#%Add a second stage with a more liberal mag cuttoff
#%limit for spoilers
#Fid(2) = Fid(1);
#Fid(2).Spoiler.MagDiffLimit = -4;
#
#
#%-----------------------------------------------------------------
#
#
#%-----------------------------------------------------------------
#%Assemble it all into one structure
#%-----------------------------------------------------------------
#Stars.General = General;
#Stars.Acq     = Acq;
#Stars.Guide   = Guide;
#Stars.Fid     = Fid;


#!/usr/local/python2.7/bin/python2.7

# Multi-Echo ICA, Release 1-rc4
# See http://dx.doi.org/10.1016/j.neuroimage.2011.12.028
# Kundu, P., Inati, S.J., Evans, J.W., Luh, W.M. & Bandettini, P.A. Differentiating 
#	BOLD and non-BOLD signals in fMRI time series using multi-echo EPI. NeuroImage (2011).
#
# meica.py version 1.0-rc4 (c) 2012 Prantik Kundu
# PROCEDURE 1 : Preprocess multi-echo datasets and apply multi-echo ICA based on spatial concatenation
# -Calculation of motion parameters based on images with highest constrast
# -Calcluation of functional-anatomical coregistration using EPI gray matter + local Pearson correlation method
# -Application of motion correction and coregistration parameters
# -Misc. EPI preprocessing (temporal alignment, smoothing, etc) in appropriate order
# -Application of FastICA using FSL MELODIC and automatic dimensionality estimation with Probabilistic PCA
#

import sys
from re import split as resplit
import re
from os import system,getcwd,mkdir,chdir,popen
import os.path
from string import rstrip,split
from optparse import OptionParser,OptionGroup

#Filename parser for NIFTI and AFNI files
def dsprefix(idn):
	def prefix(datasetname):
		return split(datasetname,'+')[0]
	if len(split(idn,'.'))!=0:
		if split(idn,'.')[-1]=='HEAD' or split(idn,'.')[-1]=='BRIK' or split(idn,'.')[-2:]==['BRIK','gz']:
			return prefix(idn)
		elif split(idn,'.')[-1]=='nii' and not split(idn,'.')[-1]=='nii.gz':
			return '.'.join(split(idn,'.')[:-1])
		elif split(idn,'.')[-2:]==['nii','gz']:
			return '.'.join(split(idn,'.')[:-2])
		else:
			return prefix(idn)
	else:
		return prefix(idn)

def dssuffix(idna):
	suffix = idna.split(dsprefix(idna))[-1]
	print suffix
	spl_suffix=suffix.split('.')
	print spl_suffix
	if len(spl_suffix[0])!=0 and spl_suffix[0][0] == '+': return spl_suffix[0]
	else: return suffix


#Configure options and help dialog
parser=OptionParser()
parser.add_option('-e',"",dest='tes',help="ex: -e 14.5,38.5,62.5  Echo times (in ms)",default='')
parser.add_option('-d',"",dest='dsinputs',help="ex: -d \"PREFIX[2,1,3]ETC.nii.gz\"  TE index of base is first. Note quotes.",default='')
parser.add_option('-f',"",dest='FWHM',help="ex: -f 8mm  Target dataset smoothness (3dBlurToFWHM). Default to 5mm. ",default='5mm')
parser.add_option('-b',"",dest='basetime',help="ex: -f 10  Time to steady-state equilibration in seconds. Default 0. ",default=0)
parser.add_option('-a',"",dest='anat',help="ex: -a mprage.nii.gz  Anatomical dataset (optional)",default='')
parser.add_option('-o',"",action="store_true",dest='oblique',help="Oblique acqusition",default=False)
extopts=OptionGroup(parser,"Extended options for preprocessing and ICA")
extopts.add_option('',"--skullstrip",action="store_true",dest='skullstrip',help="Skullstrip anatomical",default=False)
extopts.add_option('',"--despike",action="store_true",dest='despike',help="Despike data. Good for datasets with spikey motion artifact",default=False)
extopts.add_option('',"--ica_include",dest='ica_include',help="Datasets to include in ICA, comma separated list, ex: -I 3,4,5 (default include all)",default='')
extopts.add_option('',"--ica_args",dest='icaargs',help="Additional options for MELODIC ICA, ex: --ica_args='--report -d=50' ",default='--no_mm')
extopts.add_option('',"--TR",dest='TR',help="The TR. Default read from input datasets",default='')
extopts.add_option('',"--tpattern",dest='tpattern',help="Slice timing (i.e. alt+z, see 3dTshift --help). Default from header. Correction skipped if not found.",default='')
extopts.add_option('',"--zeropad",dest='zeropad',help="Zeropadding options. -z N means add N slabs in all directions. Default 15 (N.B. autoboxed after coregistration)",default="15")
extopts.add_option('',"--highpass",dest='highpass',help="Highpass filter in Hz (default 0.02)",default=0.02)
extopts.add_option('',"--align_base",dest='align_base',help="Base EPI for allineation",default='')
extopts.add_option('',"--align_interp",dest='align_interp',help="Interpolation method for allineation",default='cubic')
extopts.add_option('',"--align_args",dest='align_args',help="Additional arguments for 3dAllineate EPI-anatomical alignment",default='')
parser.add_option_group(extopts)
testopts=OptionGroup(parser,"Options for testing/reusing ME-ICA preprocessed set")
testopts.add_option('',"--label",dest='label',help="Extra label to tag this ME-ICA folder",default='')
testopts.add_option('',"--test_proc",action="store_true",dest='test_proc',help="Align and preprocess 1 dataset then exit, for testing",default=False)
testopts.add_option('',"--reuse",dest='reuse',action="store_true",help="Try to reuse existing preprocessing, useful with repeated -I calls ",default=False)
testopts.add_option('',"--overwrite",dest='overwrite',action="store_true",help="If meica.xyz directory exists, overwrite. ",default=False)
testopts.add_option('',"--exit",action="store_true",dest='exit',help="Generate script and exit",default=0)
parser.add_option_group(testopts)
(options,args) = parser.parse_args()

#Parse dataset input names
if options.dsinputs=='' or options.TR==0:
	print "Need at least dataset inputs and TE. Try meica.py -h"
		sys.exit()
	if os.path.abspath(os.path.curdir).__contains__('meica.'):
		print "You are inside a ME-ICA directory! Please leave this directory and rerun."
		sys.exit()
	dsinputs=dsprefix(options.dsinputs)
	prefix=resplit(r'[\[\],]',dsinputs)[0]
	datasets=resplit(r'[\[\],]',dsinputs)[1:-1]
	trailing=resplit(r'[\]+]',dsinputs)[-1]
	isf= dssuffix(options.dsinputs)
	tes=split(options.tes,',')
	if len(options.tes.split(','))!=len(datasets):
		print len(options.tes.split(',')), len(datasets)
		print "Number of TEs and input datasets must be equal. Or try quotes around -d argument."
		sys.exit()
	
	#Parse timing arguments
	if options.TR!='':tr=float(options.TR)
	else: 
		tr=float(os.popen('3dinfo -tr %s%s%s%s' % (prefix,datasets[0],trailing,isf)).readlines()[0].strip())
		options.TR=str(tr)
	timetoclip=float(options.basetime)
	basebrik=int(timetoclip/tr)
	highpass = float(options.highpass)
	highpass_ind = 1/(highpass*tr)
	
	#Prepare script variables
	meicadir=os.path.dirname(sys.argv[0])
	sl = []					#Script command list
	sl.append('#'+" ".join(sys.argv).replace('"',r"\""))
	print '#'+" ".join(sys.argv).replace('"',r"\"")
	osf='.nii.gz'				#Using NIFTI outputs
	vrbase=prefix+datasets[0]+trailing
	if options.align_base == '': align_base = basebrik
	else: align_base = options.align_base
	setname=prefix+''.join(datasets)+trailing+options.label
	startdir=rstrip(popen('pwd').readlines()[0])
	combstr=""; allcombstr=""
	if not (options.reuse or not options.overwrite): system('rm -rf meica.%s' % (setname))
	elif not options.overwrite and not options.reuse: sl.append("if [[ -e meica.%s/stage ]]; then echo ME-ICA directory exists, exiting; exit; fi" % (setname))
	system('mkdir -p meica.%s' % (setname))
	sl.append("cp _meica_%s.sh meica.%s/" % (setname,setname))
	sl.append("cd meica.%s" % setname)
	if options.reuse: sl.append("stage=`cat stage`\nif [ \"$stage\" != \"2\" ]; then ") #Starting reuse condition
	thecwd= "%s/meica.%s" % (getcwd(),setname)
	#if options.ica_include!='': ica_datasets = options.ica_include.split(',')
	#else: ica_datasets = sorted(datasets)
	ica_datasets = sorted(datasets)
	if len(split(options.zeropad))==1 : zeropad_opts=" -I %s -S %s -A %s -P %s -L %s -R %s " % (tuple([options.zeropad]*6))
	elif options.zeropad!='': zeropad_opts=options.zeropad
	else: zeropad_opts=''
	if options.despike: despike_opt = "-despike"
	else: despike_opt = ""
	
	#Parse anatomical processing options, process anatomical
	if options.anat != '':
		nsmprage = options.anat
		anatprefix=dsprefix(nsmprage)
		pathanatprefix="%s/%s" % (startdir,anatprefix)
		if options.oblique:
			sl.append("if [ ! -e %s_do.nii.gz ]; then 3dWarp -overwrite -prefix %s_do.nii.gz -deoblique %s/%s; fi" % (pathanatprefix,pathanatprefix,startdir,nsmprage))
			nsmprage="%s_do.nii.gz" % (anatprefix)
		if options.skullstrip: 
			sl.append("if [ ! -e %s_ns.nii.gz ]; then 3dSkullStrip -overwrite -prefix %s_ns.nii.gz -input %s/%s; fi" % (pathanatprefix,pathanatprefix,startdir,nsmprage))
			nsmprage="%s_ns.nii.gz" % (anatprefix)
	
	# Calculate rigid body alignment
	sl.append("echo 1 > stage" )
	vrAinput = "%s/%s%s" % (startdir,vrbase,isf)
	if options.oblique: 
		sl.append("3dcopy %s ./%s%s" % (vrAinput,vrbase,isf))
		vrAinput = "%s/%s%s" % ('.',vrbase,isf)
		sl.append("3drefit -deoblique %s" % (vrAinput))
	sl.append("3dvolreg -Fourier -prefix ./%s_vrA%s -base %s[%s] -dfile ./%s_vrA.1D -1Dmatrix_save ./%s_vrmat.aff12.1D %s" % \
		                (vrbase,isf,vrAinput,basebrik,vrbase,prefix,vrAinput))
	sl.append("1dcat './%s_vrA.1D[1..6]{%s..$}' > motion.1D " % (vrbase,basebrik))
	sl.append("3dcalc -expr 'a' -a %s[%s] -prefix ./_eBmask%s" % (vrAinput,align_base,osf))
	sl.append("bet _eBmask%s eBmask%s" % (osf,osf))
	sl.append("fast -t 2 -n 3 -H 0.1 -I 4 -l 20.0 -b -o eBmask eBmask%s" % (osf))
	sl.append("3dcalc -a eBmask%s -b eBmask_bias%s -expr 'a/b' -prefix eBbase%s" % ( osf, osf, osf))
	e2dsin = prefix+datasets[0]+trailing
	
	# Calculate affine anatomical warp if anatomical provided, then combine motion correction and coregistration parameters 
	if options.anat!='':
		sl.append("cp %s/%s* ." % (startdir,nsmprage))
		align_args=""
		if options.align_args!="": align_args=options.align_args
		elif options.oblique: " -cmass -maxrot 30 -maxshf 30 "
		else: align_args=" -maxrot 20 -maxshf 20 -parfix 7 1  -parang 9 0.83 1.0 "
		sl.append("3dAllineate -weight_frac 1.0 -VERB -warp aff -weight eBmask_pve_0.nii.gz -lpc+ -base eBbase.nii.gz -master %s/%s -source %s/%s -prefix ./%s_al -1Dmatrix_save %s_al_mat %s" % (startdir,nsmprage,star
	tdir,nsmprage, anatprefix,anatprefix,align_args))
		sl.append("cat_matvec -ONELINE %s_al_mat.aff12.1D -I %s_vrmat.aff12.1D > %s_wmat.aff12.1D" % (anatprefix,prefix,prefix))
	else: sl.append("cp %s_vrmat.aff12.1D %s_wmat.aff12.1D" % (prefix,prefix))
	
	#Preprocess datasets
	datasets.sort()
	for echo_ii in range(len(datasets)):
		echo = datasets[echo_ii]
		dsin = prefix+echo+trailing
		if options.tpattern!='':
			tpat_opt = ' -tpattern %s ' % options.tpattern
		else:
			tpat_opt = ''
		sl.append("3dTshift %s -prefix ./%s_ts%s %s/%s%s" % (tpat_opt,dsin,osf,startdir,dsin,isf) )
		sl.append("3drefit -deoblique %s_ts%s" % (dsin,osf))
		if zeropad_opts!="" : sl.append("3dZeropad %s -overwrite -prefix %s_ts%s %s_ts%s" % (zeropad_opts,dsin,osf,dsin,osf))
		sl.append("3dAllineate -final %s -%s -float -1Dmatrix_apply %s_wmat.aff12.1D -base %s_ts%s -input  %s_ts%s -prefix ./%s_vr%s" % \
			(options.align_interp,options.align_interp,prefix,dsin,osf,dsin,osf,dsin,osf))
		if echo_ii == 0: 
			sl.append("bet %s_vr%s eBvrmask%s " % (dsin,osf,osf ))
			sl.append("3dBrickStat -mask eBvrmask.nii.gz -percentile 50 1 50  %s_vr%s[$] > gms.1D" % (dsin,osf))
			#sl.append("3dAutobox -overwrite -prefix eBvrmask%s eBvrmask%s" % (osf,osf) )
			sl.append("3dcalc -a eBvrmask.nii.gz -expr 'notzero(a)' -overwrite -prefix eBvrmask.nii.gz")
		sl.append("3dresample -overwrite -rmode NN -master eBvrmask%s -prefix ./%s_vr%s -inset ./%s_vr%s[%i..$]" % (osf,dsin,osf,dsin,osf,basebrik))
		if options.FWHM=='0mm': sl.append("3dcalc -a eBvrmask%s -b ./%s_vr%s -expr \"ispositive(a)*b\" -prefix  ./%s_sm%s" % (osf,dsin,osf,dsin,osf))	 
		#sl.append("ln -s ./%s_vr%s ./%s_sm%s" % (dsin,osf,dsin,osf))
		else: sl.append("3dBlurInMask -fwhm %s -mask eBvrmask%s -prefix ./%s_sm%s ./%s_vr%s" % (options.FWHM,osf,dsin,osf,dsin,osf))
		sl.append("gms=`cat gms.1D`; gmsa=($gms); p50=${gmsa[1]}")
		sl.append("3dcalc -overwrite -a ./%s_sm%s -expr \"a*10000/${p50}\" -prefix ./%s_sm%s" % (dsin,osf,dsin,osf))
		sl.append("3dTstat -prefix ./%s_mean%s ./%s_sm%s" % (dsin,osf,dsin,osf))
		sl.append("3dBandpass %s -prefix ./%s_in%s %f 99 ./%s_sm%s " % (despike_opt,dsin,osf,float(options.highpass),dsin,osf) )
		sl.append("3dcalc -overwrite -a ./%s_in%s -b ./%s_mean%s -expr 'a+b' -prefix ./%s_in%s" % (dsin,osf,dsin,osf,dsin,osf))
		sl.append("3dTstat -stdev -prefix ./%s_std%s ./%s_in%s" % (dsin,osf,dsin,osf))
		if options.test_proc: sl.append("exit")
	
	sl.append("echo 2 > stage")
	if options.reuse: sl.append("fi")
	
	#Concatenate for ICA
	if len(ica_datasets)==1:
		dsin = prefix+''.join(ica_datasets)+trailing
		ica_prefix = dsin
		ica_input="./%s_in%s" % (prefix,''.join(ica_datasets),trailing)
		ica_mask="eBvrmask.nii.gz"
	else:
		ica_prefix="%sC%s%s" % (prefix,''.join(ica_datasets),trailing)
		ica_input = "%s_ffd.nii.gz" % ica_prefix
		ica_mask = "%s_mask.nii.gz" % ica_prefix
		zcatstring=""
		for echo in ica_datasets: 
			dsin = prefix+echo+trailing
			zcatstring = "%s ./%s_in%s" % (zcatstring,dsin,osf)
		sl.append("3dZcat -prefix %s_ffd.nii.gz %s" % (ica_prefix,zcatstring) )
		sl.append("3dcalc -a %s[0] -expr 'notzero(a)' -prefix %s" % (ica_input,ica_mask))
	
	#Run ICA and TE-dependence analysis
	sl.append("echo Running MELODIC ICA...")
	sl.append("melodic -i %s -o %s%s.ica --sep_whiten --pbsc --Oall --Ostats --mask=%s --tr=%s %s" % (ica_input,ica_prefix,options.label,ica_mask,tr,options.icaargs))
	sl.append("cd %s%s.ica" % (ica_prefix,options.label))
	sl.append("mkdir TED; cd TED")
	if len(datasets) != len(ica_datasets): 
		#Placeholder. Need more logic for tedana.py multi dataset handling
		sl.append("%s %s -e %s -m ../melodic_mix -d ../../%s_ffd.nii.gz --bpdata=../../%s_ffd.nii.gz" % (sys.executable, '/'.join([meicadir,'tedana.py']),options.tes,ica_prefix,ica_prefix))
	else: sl.append("%s %s -e %s -m ../melodic_mix -d ../../%s_ffd.nii.gz --bpdata=../../%s_ffd.nii.gz" % (sys.executable, '/'.join([meicadir,'tedana.py']),options.tes,ica_prefix,ica_prefix))
	sl.append("1dplot -ps -title \"\\kappa and \\rho spectra \" -ynames \"\\kappa\" \"\\rho\" - -one -box comp_table.txt[1] comp_table.txt[2] > krho_spectra.ps")
	sl.append("cd ../..")
	sl.append("rm TED; ln -s %s%s.ica/TED ." % (ica_prefix,options.label))
	
	#Write the preproc script and execute it
	ofh = open('_meica_%s.sh' % setname ,'w')
	print "\n".join(sl)+"\n"
	ofh.write("\n".join(sl)+"\n")
	ofh.close()
	if not options.exit: system('bash _meica_%s.sh' % setname)	
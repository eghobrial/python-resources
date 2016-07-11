#!/usr/bin/env python

# Multi-Echo ICA, Release 1-rc4
# See http://dx.doi.org/10.1016/j.neuroimage.2011.12.028
# Kundu, P., Inati, S.J., Evans, J.W., Luh, W.M. & Bandettini, P.A. Differentiating 
#	BOLD and non-BOLD signals in fMRI time series using multi-echo EPI. NeuroImage (2011).
#
# tedana.py version 1.0		(c) 2012 Noah Brenowitz, Prantik Kundu, Souheil Inati
# tedana.py version 0.5		(c) 2011 Prantik Kundu, Souheil Inati
#
# PROCEDURE 2 : Process components and perform TE-dependence analysis
# -Computes T2* map using least-squares approach 
# -Computes TE-dependence for each component resulting from ICA 
# -or- Computes TE-dependence of each component of a general linear model
# 

import os
from optparse import OptionParser
import numpy as np
import nibabel as nib
# from ipdb import set_trace as st
# from pylab import *
from sys import stdout
#import ipdb



def _interpolate(a, b, fraction):
    """Returns the point at the given fraction between a and b, where
    'fraction' must be between 0 and 1.
    """
    return a + (b - a)*fraction;

def scoreatpercentile(a, per, limit=(), interpolation_method='fraction'):
    """
    This function is grabbed from scipy

    """
    values = np.sort(a, axis=0)
    if limit:
        values = values[(limit[0] <= values) & (values <= limit[1])]

    idx = per /100. * (values.shape[0] - 1)
    if (idx % 1 == 0):
        score = values[idx]
    else:
        if interpolation_method == 'fraction':
            score = _interpolate(values[int(idx)], values[int(idx) + 1],
                                 idx % 1)
        elif interpolation_method == 'lower':
            score = values[np.floor(idx)]
        elif interpolation_method == 'higher':
            score = values[np.ceil(idx)]
        else:
            raise ValueError("interpolation_method can only be 'fraction', " \
                             "'lower' or 'higher'")

    return score

def niwrite(data,aff, name , head=None):
	stdout.write("Writing file: %s ...." % name) 
	if head is not None:
		nx,ny,nz,nt = data.shape
		head.set_data_shape((nx,ny,nz,nt))

	outni = nib.Nifti1Image(data,aff,header=head)
	outni.to_filename(name)
	print "done."



def cat2echos(data,Ne):
	"""
	cat2echos(data,Ne)

	Input:
	data shape is (nx,ny,Ne*nz,nt)
	"""
	nx,ny = data.shape[0:2]
	nz = data.shape[2]/Ne
	if len(data.shape) >3:
		nt = data.shape[3]
	else:
		nt = 1

	return np.reshape(data,(nx,ny,nz,Ne,nt),order='F')

def makemask(cdat):

	nx,ny,nz,Ne,nt = cdat.shape

	mask = np.ones((nx,ny,nz),dtype=np.bool)

	for i in range(Ne):
		tmpmask = (cdat[:,:,:,i,:] != 0).prod(axis=-1,dtype=np.bool)
		mask = mask & tmpmask

	return mask

def fmask(data,mask):
	"""
	fmask(data,mask)

	Input:
	data shape is (nx,ny,nz,...)
	mask shape is (nx,ny,nz)

	Output:
	out shape is (Nm,...)
	"""

	s = data.shape
	sm = mask.shape

	N = s[0]*s[1]*s[2]
	news = []
	news.append(N)

	if len(s) >3:
		news.extend(s[3:])

	tmp1 = np.reshape(data,news)
	fdata = tmp1.compress((mask > 0 ).ravel(),axis=0)

	return fdata.squeeze()

def unmask (data,mask):
	"""
	unmask (data,mask)

	Input:

	data has shape (Nm,nt)
	mask has shape (nx,ny,nz)

	"""
	M = (mask != 0).ravel()
	Nm = M.sum()

	nx,ny,nz = mask.shape

	if len(data.shape) > 1:
		nt = data.shape[1]
	else:
		nt = 1

	out = np.zeros((nx*ny*nz,nt),dtype=data.dtype)
	out[M,:] = np.reshape(data,(Nm,nt))

	return np.reshape(out,(nx,ny,nz,nt))

def t2smap(catd,mask,tes):
	"""
	t2smap(catd,mask,tes)

	Input:

	catd  has shape (nx,ny,nz,Ne,nt)
	mask  has shape (nx,ny,nz)
	tes   is a 1d numpy array
	"""
	nx,ny,nz,Ne,nt = catd.shape
	N = nx*ny*nz

	echodata = fmask(catd,mask)
	Nm = echodata.shape[0]

	#Do Log Linear fit
	B = np.reshape(np.abs(echodata), (Nm,Ne*nt)).transpose()
	B = np.log(B)
	x = np.array([np.ones(Ne),-tes])
	X = np.tile(x,(1,nt))
	X = np.sort(X)[:,::-1].transpose()

	beta,res,rank,sing = np.linalg.lstsq(X,B)
	t2s = 1/beta[1,:].transpose()
	s0  = np.exp(beta[0,:]).transpose()

	out = unmask(t2s,mask)

	return out

	head.set_data_shape((nx,ny,nz,2))
	vecwrite(out,'t2s.nii.gz',aff,head=head)

def get_coeffs(data,mask,X):
	"""
	get_coeffs(data,X)

	Input:

	data has shape (nx,ny,nz,nt)
	mask has shape (nx,ny,nz)
	X    has shape (nt,nc)

	Output:

	out  has shape (nx,ny,nz,nc)
	""" 
	mdata = fmask(data,mask).transpose()
	tmpbetas = np.linalg.lstsq(X,mdata)[0].transpose()
	out = unmask(tmpbetas,mask)

	return out

def fitmodels(betas,t2s,mu,mask,tes,cc=-1,sig=None,fout=None,pow=2.):
	"""
	Usage:

	fitmodels(betas,t2s,mu,mask,tes)

	Input:
	betas,mu,mask are all (nx,ny,nz,Ne,nc) ndarrays
	t2s is a (nx,ny,nz) ndarray
	tes is a 1d array
	"""
	fouts  = []
	nx,ny,nz,Ne,nc = betas.shape
	Nm = mask.sum()

	tes = np.reshape(tes,(Ne,1))
	mumask   = fmask(mu,mask)
	t2smask  = fmask(t2s,mask)
	betamask = fmask(betas,mask)

	comptab = np.zeros((nc,3))

	#Setup Xmats

	#Model 1
	X1 = mumask.transpose()

	#Model 2
	X2 = np.tile(tes,(1,Nm))*mumask.transpose()/t2smask.transpose()

	if cc!=-1: comps=[cc]
	else: comps=range(nc)
	
	if sig!=None: sigmask=fmask(sig,mask).transpose()
	
	for i in comps:

		#size of B is (nc, nx*ny*nz)
		B = betamask[:,:,i].transpose()
		alpha = (np.abs(B)**pow).sum(axis=0)

		#S0 Model
		coeffs_S0 = (B*X1).sum(axis=0)/(X1**2).sum(axis=0)
		SSE_S0 = (B - X1*np.tile(coeffs_S0,(Ne,1)))**2
		SSE_S0 = SSE_S0.sum(axis=0)

		F_S0 = (alpha - SSE_S0)*2/(SSE_S0)
		#F_S0 = (SSTR_S0)*2/(alpha - SSTR_S0)

		#R2 Model
		coeffs_R2 = (B*X2).sum(axis=0)/(X2**2).sum(axis=0)
		SSE_R2 = (B - X2*np.tile(coeffs_R2,(Ne,1)))**2
		SSE_R2 = SSE_R2.sum(axis=0)
		F_R2 = (alpha - SSE_R2)*2/(SSE_R2)

		# Output Voxelwise F-Stats. "fout" is the affine transformation
		if fout is not None:
			
			out = np.zeros((nx,ny,nz,2))
			name = "cc%.3d.nii.gz"%i

			out[:,:,:,0] = np.squeeze(unmask(F_S0,mask))
			out[:,:,:,1] = np.squeeze(unmask(F_R2,mask))

			niwrite(out,fout,name)
			os.system('3drefit -sublabel 0 F_SO -sublabel 1 F_R2 %s 2> /dev/null > /dev/null'%name)


		Bn = B/sigmask
		Bz=(Bn-np.tile(Bn.mean(axis=-1),(Bn.shape[1],1)).T)/np.tile(Bn.std(axis=-1),(Bn.shape[1],1)).T
		wts = Bz.mean(axis=0)
		# When computing K and Rho, flip sign of components with negative weighted average
		if np.average(wts,weights=np.abs(wts))<0: wts*=-1
		wts[wts<0]=0
		wts[wts>4]=4
		wts = np.power(wts,2)
		F_R2[F_R2>500]=500
		F_S0[F_S0>500]=500
		
		kappa = np.average(F_R2,weights = wts)
		rho   = np.average(F_S0,weights = wts)		

		print "Comp %d Kappa: %f Rho %f" %(i,kappa,rho)
		comptab[i,0] = i
		comptab[i,1] = kappa
		comptab[i,2] = rho

		#debug
		if cc!=-1: return [F_R2,alpha]
			

	return comptab

def selcomps(comptable):
	"""
	selcomps(comptable)

	Input:
	comptable = [component kappa rho]

	Uses the method for finding an elbow proposed here:
	http://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
	"""
	nc = comptable.shape[0]
	comps = comptable[:,0]
	kappa = comptable[:,1]
	rho   = comptable[:,2]

	baseind  = comps.min()

	order  = kappa.argsort()
	ks     = kappa[order]
	rs    = rho[order]
	cs    = comps[order]

	# Calculate K threshold
	hi_k_min = 100.0
	
	coords = np.array([np.arange(nc),ks])
	p  = coords - np.tile(np.reshape(coords[:,0],(2,1)),(1,nc))
	b  = p[:,-1] 

	b_hat = np.reshape(b/np.sqrt((b**2).sum()),(2,1))
	proj_p_b = p - np.dot(b_hat.T,p)*np.tile(b_hat,(1,nc))
	d = np.sqrt((proj_p_b**2).sum(axis=0))

	k_min_ind = d.argmax()
	k_min  = ks[k_min_ind]


	# Calculate Rho Threshold
	# r_max the 35th percentile of the low kappa components 
	selR = rs[1:k_min_ind]
	r_max = scoreatpercentile(selR,35.0)

	
	# might want to tweak this
	accepted      = (ks > hi_k_min) | ((ks > k_min) & (rs<r_max))
	midk_rejected = (ks > k_min) & (rs > r_max)
	rejected      = True - (accepted|midk_rejected)

	# plot(ks)
	# plot(rs,'r')
	# plot(accepted.nonzero()[0],ks[accepted],'bo')
	# plot(rejected.nonzero()[0],ks[rejected],'ro')
	# plot(midk_rejected.nonzero()[0],ks[midk_rejected],'go')
	# show()

	acc_comps = (cs[accepted] - baseind).astype(np.int)
	rej_comps = (cs[rejected] - baseind).astype(np.int)
	midk_comps = (cs[midk_rejected] - baseind).astype(np.int)

	# saving files
	np.savetxt('accepted.txt',acc_comps,fmt='%d',delimiter=',')
	np.savetxt('rejected.txt',rej_comps,fmt='%d',delimiter=',')
	np.savetxt('midk_rejected.txt',midk_comps,fmt='%d',delimiter=',')

	return acc_comps,rej_comps,midk_comps

def split_ts(data,betas,mu,mix,segdic,aff,head,echo,split_mode):

	nx,ny,nz,ne,nt = data.shape
	nc = mix.shape[1]

	data = np.squeeze(data[:,:,:,echo,:])
	betas = np.squeeze(betas[:,:,:,echo,:]).reshape([nx*ny*nz,nc],order='F')
	mu = mu[:,:,:,echo]

	if split_mode=='denoise':
		print "Denoising data"
		seg='lowk'
		segdata = betas[:,segdic[seg]].dot(mix[:,segdic[seg]].T)
		segdata = segdata.reshape([nx,ny,nz,nt],order='F')
		niwrite(segdata,aff,'%s_e%i.nii.gz' % (seg,echo+1))
		data -= segdata
		niwrite(data,aff,'e%i_dn.nii.gz' % (echo+1))
		
	else:
		print "Splitting data"
		for seg in segdic.keys():
			segdata = np.dot(betas[:,segdic[seg]],(mix[:,segdic[seg]].T))
			segdata = segdata.reshape([nx,ny,nz,nt],order='F')
			data -= segdata
			niwrite(segdata,aff,'%s_e%i.nii.gz' % (seg,echo+1))
		niwrite(data,aff,'resid_e%i.nii.gz' % (echo+1))
	

def optcom(data,t2s,tes,mask):
	"""
	out = optcom(data,t2s)


	Input:

	data.shape = (nx,ny,nz,Ne,Nt)
	t2s.shape  = (nx,ny,nz)
	tes.shape  = (Ne,)

	Output:

	out.shape = (nx,ny,nz,Nt)
	"""
	nx,ny,nz,Ne,Nt = data.shape 

	fdat = fmask(data,mask)
	ft2s = fmask(t2s,mask)
	
	tes = tes[np.newaxis,:]
	ft2s = ft2s[:,np.newaxis]
	
	alpha = tes * np.exp(-tes /ft2s)
	alpha = np.tile(alpha[:,:,np.newaxis],(1,1,Nt))

	fout  = np.average(fdat,axis = 1,weights=alpha)
	out = unmask(fout,mask)
	print 'Out shape is ', out.shape
	return out

def getelbow(ks):
	nc = ks.shape[0]
	coords = np.array([np.arange(nc),ks])
	p  = coords - np.tile(np.reshape(coords[:,0],(2,1)),(1,nc))
	b  = p[:,-1] 
	b_hat = np.reshape(b/np.sqrt((b**2).sum()),(2,1))
	proj_p_b = p - np.dot(b_hat.T,p)*np.tile(b_hat,(1,nc))
	d = np.sqrt((proj_p_b**2).sum(axis=0))
	k_min_ind = d.argmax()
	k_min  = ks[k_min_ind]
	return k_min_ind

def mdpica2():
	import mdp
	OCcatd = optcom(catd,t2s,tes,mask)
	nx,ny,nz,ne,nt = catd.shape
	d = np.float64(fmask(OCcatd,mask))
	dv = d.shape[0]
	dm = d.mean(axis=1)
	dz = ((d.T-d.T.mean(axis=0))/d.T.std(axis=0)).T #Variance normalize in time

	##Do PC dimension selection
	#Get eigenvalue cutoff
	u,s,v = np.linalg.svd(dz,full_matrices=0)
	sp = s/s.sum()
	eigelb = sp[getelbow(sp)]
	#Compute K and Rho for PCA comps
	betasv = get_coeffs(bpd,np.tile(mask,(1,1,Ne)),v.T)
	betasv = cat2echos(betasv,Ne)
	ctb = fitmodels(betasv,t2s,mu,mask,tes,sig=sig,fout=options.fout,pow=2)
	ctb = np.vstack([ctb.T,sp] ).T
	#Pick components on minimum K/Rho, making sure no comps with tiny % var explained get in
	pcsel = (np.array(ctb[:,1]>30,dtype=np.int)+np.array(ctb[:,2]>30,dtype=np.int)+np.array(ctb[:,3]>eigelb,dtype=np.int))*np.array(ctb[:,3]>eigelb/2,dtype=np.int) > 0
	dd = u.dot(np.diag(s*np.array(pcsel,dtype=np.int))).dot(v)
	nc = s[pcsel].shape[0]
	print nc
	#ipdb.set_trace()
	
	#Do ICA
	icanode = mdp.nodes.FastICANode(white_comp=nc,white_parm={'svd':True},approach='symm',g='tanh',fine_g='pow3',limit=0.00001,verbose=True)
	icanode.train(dd)
	smaps = icanode.execute(dd)
	del dd
	mmix = icanode.get_recmatrix().T
	mmix = mmix/mmix.std(axis=0)
	
	#Compute K and Rho for ICA compos
	betas = get_coeffs(bpd,np.tile(mask,(1,1,Ne)),mmix)
	betas = cat2echos(betas,ne)
	comptable = fitmodels(betas,t2s,mu,mask,tes,sig=sig,fout=options.fout,pow=2)
	acc,rej,midk = selcomps(comptable)
	
	feats = smaps[:,acc]
	feats = unmask(feats,mask)
	niwrite(feats,aff,'feats.nii.gz',head=head)
	os.system('3dcopy feats.nii.gz feats+orig  > /dev/null 2> /dev/null')
	
	return smaps,mmix,comptable
	
	
###################################################################################################
# 						Begin Main
###################################################################################################

MELOIC = 0
BPDATA = 1

if __name__=='__main__':

	parser=OptionParser()
	parser.add_option('-b',"--betas",dest='betas',help="Dataset with Bandpassed data (stacked)",default=None)
	parser.add_option('-d',"--orig_data",dest='data',help="Z-catted Dataset for T2S Estimation",default=None)
	#parser.add_option('-u',"--mu",dest='mu',help="Means (stacked)",default=None)
	parser.add_option('-s',"--sig",dest='sig',help="Scales (stacked)",default=None)
	parser.add_option('',"--bpdata",dest='bpdata',help="Dataset with Bandpassed dataset (stacked)",default=None)
	parser.add_option('-m',"--melodic_mix",dest='mmix',help="Melodic Mix",default=None)
	parser.add_option('-e',"--TEs",dest='tes',help="Echo times (in ms) ex: 15,39,63",default='15,39,63')
	parser.add_option('',"--split",dest='split',action="store_true",help="Split time series data into segments instead of denoising",default=False)
	parser.add_option('','--tsdata',dest='tsdata',help="Concattenated Time Series Data for Optcom",default=None)
	parser.add_option('','--fout',dest='fout',help="Output Voxelwise Kappa/Rho Maps",action="store_true",default=False)

	(options,args) = parser.parse_args()

	dataname  = options.data
	betaname  = options.betas
	#muname    = options.mu
	signame   = options.sig

	bpdataname = options.bpdata
	melname = options.mmix

	tes = np.fromstring(options.tes,sep=',',dtype=np.float32)
	Ne = tes.shape[0]

	if betaname is not None:
		mode = MELOIC
	else:
		mode = BPDATA


	#load data
	catim  = nib.load(dataname)	
	head   = catim.get_header()
	aff    = catim.get_affine()

	catd = cat2echos(catim.get_data(),Ne)
	nx,ny,nz,Ne,nt = catd.shape

	mu  = catd.mean(axis=-1)

	if mode == MELOIC:
		#load datasets
		betaim = nib.load(betaname)
		#muim =nib.load(muname)
		sigim =nib.load(signame)

		betad = betaim.get_data()
		sig  = np.reshape(sigim.get_data(),(nx,ny,nz*Ne,1))
		#mu  = cat2echos(muim.get_data(),Ne)

		nc = betad.shape[-1]

	elif mode == BPDATA:
		bpim = nib.load(bpdataname)
		mmix = np.genfromtxt(melname)
		nc = mmix.shape[1]
		bpd = bpim.get_data()


	print "Making Mask"
	mask  = makemask(catd)

	print "Finding T2S map"
	t2s   = t2smap(catd,mask,tes) 


	niwrite(t2s,aff,'t2sv.nii',head=head)

	#Compute ICA
	#cmaps,mmix = mdpica(catd,60)

	print "Calculating Betas"
	if mode == MELOIC:
		betas = betad*np.tile(sig,(1,1,1,nc))

	elif mode == BPDATA:
		betas = get_coeffs(bpd,np.tile(mask,(1,1,Ne)),mmix)
		niwrite(betas,aff,'betas.nii.gz',head=head)

	betas = cat2echos(betas,Ne)


	print "Fitting T2S and S0 Models\n"
	if options.fout:
		options.fout = aff
	else:
		options.fout = None

	sig = catd.std(axis=-1)
	comptable = fitmodels(betas,t2s,mu,mask,tes,sig=sig,fout=options.fout,pow=2)
	#comptable = fitmodels(betas,t2s,mu,mask,tes)

	print "Writing to file"
	sortab = comptable[comptable[:,1].argsort()[::-1],:]
	with open('comp_table.txt','w') as f:
		f.write("#	comp	Kappa	Rho\n")
		for i in range(nc):
			f.write('%d\t%f\t%f\n'%(sortab[i,0]+1,sortab[i,1],sortab[i,2]))
	
	print "Selecting BOLD-like Components"
	acc,rej,midk = selcomps(comptable)
	
	print "Optimally Combining Betas"
	dataOC = optcom(betas,t2s,tes,mask)
	niwrite(dataOC,aff,'melodic_OC.nii.gz',head=head)
	
	print "Creating Feature Dataset"
	dataOCz = fmask(dataOC,mask)
	dataOCz = ((dataOCz.T-dataOCz.T.mean(axis=0))/dataOCz.T.std(axis=0)).T
	dataOCz = unmask(dataOCz,mask)
	niwrite(dataOCz,aff,'feats_OC.nii.gz',head=head)

	if options.tsdata:
		print "Optimally Combining Time Series"
		tsim  = nib.load(options.tsdata)
		tsdat = cat2echos(tsim.get_data(),Ne)
		tsdatOC = optcom(tsdat,t2s,tes,mask)
		print tsdatOC.shape
		niwrite(tsdatOC,aff,'tsOC.nii.gz')

	print "Writing Kappa-filtered timeseries datasets"
	segdic={'hik': acc, 'midk': midk, 'lowk':rej}
	if options.split: split_mode='split'
	else: split_mode='denoise'
	split_ts(catd,betas,mu,mmix,segdic,aff,head,int(np.ceil((Ne-1)/2.)),split_mode=split_mode)

	#os.system('3dTcat -overwrite -prefix feats feats_OC.nii.gz[`cat accepted.txt`] > /dev/null 2> /dev/null')







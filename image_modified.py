# -*- coding: utf-8 -*-
"""
Created on Tue Feb 04 10:36:35 2020

@author: Dong Li
"""

from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from scipy import asarray as ar,exp
from scipy.optimize import curve_fit
import scipy.stats as stats
import random as rd




## drop the long tail of strongly bright data
def drop_long_tail(data,cell_area_max):
    fity=powerlaw.Fit(data)
    fit_xmin=fity.power_law.xmin
            
    xyst=list(data)
    xyst.sort(reverse=True)
    cell_area_thre=xyst[cell_area_max]
    xmin0=np.maximum(fit_xmin,cell_area_thre)
    data_select=[ixy for ixy in data if ixy<xmin0]    
    return data_select

## drop parts of the data with too small or too large values 
def drop_data_valeus(data,proportion_select):
    data.sort()
    xyst_length=len(data)
    xyst_length_select=int(xyst_length*proportion_select)
    xyst_start=int((xyst_length-xyst_length_select)*0.5)
    data_select=data[xyst_start:xyst_start+xyst_length_select]
    return data_select
    
## drop ranges of the data with too small or too large values 
def drop_data_ranges(data,range_drop):
    if len(data)>0:
        xyst_min=min(data)
        xyst_max=max(data)
        xyst_min0=xyst_min+range_drop*(xyst_max-xyst_min)
        xyst_max0=xyst_max-range_drop*(xyst_max-xyst_min)
        data_select=[x for x in data if x>xyst_min0 and x<xyst_max0]
    elif len(data)<=0:
        data_select=[]
    return data_select
    
## Gaussian fitting
def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
    
## 2D Gaussian fitting   
def twoDgaussian(x, x0, y0, xalpha, yalpha, A):
    return A*exp(-((x[0]-x0)/xalpha)**2-((x[1]-y0)/yalpha)**2)

## Perform the Gaussian fitting
def Perform_G_fitting(data,n_hist):
    xhist=np.histogram(data,n_hist)
    xhist_x_b=xhist[1]
    xhist_x=[]
    for ixhist in range(1,len(xhist_x_b)):
        xhist_x.append((xhist_x_b[ixhist]+xhist_x_b[ixhist-1])/2)
    xhist_y=list(xhist[0])
    xhist_x0=ar(xhist_x)
    xhist_y0=ar(xhist_y)
    binsize=xhist_x0[1]-xhist_x0[0]
    # Guuassian fit
    x=xhist_x0.copy()
    y=xhist_y0/(sum(xhist_y0)*binsize)
    mean = sum(x*y)*binsize              
    sigma = sum(y*(x-mean)**2)/binsize  
    popt,pcov = curve_fit(gaus,x,y,p0=[max(y),mean,sigma])
    return popt,pcov,binsize,x,y


Path_rawimage='./image/'
Path_modified='./modifiedimage/'
file_note= open("note.txt","w+") 
file_list=os.listdir(Path_rawimage)
#file_name='M248L1_052215_ref_modified.tif'
#parameters
xysize=512 # x-y size of the image
proportion_select=0.955 # 2 error range used to drop the data basing on the values
range_drop=0.05 # used to drop the data basing on the range
lattice_len=64 # length of the moving window
lattice_step=32 # step of the moving window
n_hist=64 # number of the bins used for histogram
fitting_thre=0.98 # S_W_test: how similar to a Gausssian distribution
cell_area_max=int(lattice_len*lattice_len*0.04) # assume the area od neurons are not bigger than 4%
target_brightness=2000 # target if the brightness
target_contrast=200 # target of the contrast





for file_name in file_list:   #
    im = io.imread(Path_rawimage+file_name)
#    im = io.imread(Path_modified+'modified'+file_name)
    im_modified=im.copy()
    
    
    
    ## creat folders if the are not there   
    #path_brightness='./brightness/'
    #path_contrast='./contrast/'
    
    #####################
    for j in range(0,np.size(im,0)):#
    #    j=60
        xy2D=[]
        Mblist=[]
        Mclist=[]
        # orginal image
        xys=im[j,:,:]
        # histogram of the original image
        xysf=np.reshape(xys,(np.size(xys),1))[:,0]
        #brightness and contrast matrices
        M_brightness=np.full([xysize,xysize],np.nan)
        M_contrast=np.full([xysize,xysize],np.nan)
        
        for ix0 in range(0,xysize-lattice_len+lattice_step,lattice_step):   
            for iy0 in range(0,xysize-lattice_len+lattice_step,lattice_step):   
                print(ix0,iy0)
                # select a range of the orginal image
                ixl=lattice_len
                iyl=ixl
                xys_select=xys[ix0:ix0+ixl,iy0:iy0+iyl]
                # histogram of the selcted image
                xysf_select=np.reshape(xys_select,(np.size(xys_select),1))[:,0]
                ## drop long tail
                xyst_select=drop_long_tail(xysf_select,cell_area_max)
                ## drop parts of the data with too small or too large values 
                xyst_select_0=drop_data_valeus(xyst_select,proportion_select)
                ## drop ranges of the data with too small or too large values
                xyst_select_1=drop_data_ranges(xyst_select_0,range_drop)
                
                if len(xyst_select_1)>=3:
                ## Shapiro-Wilk test
                    S_W_test=stats.shapiro(xyst_select_1)
                        
                    if S_W_test[0]>fitting_thre:                    
                        print("S_W_test",S_W_test,str(j))
                        ## Perform the Gaussian fitting
                        try:
                            popt,pcov,binsize,x,y=Perform_G_fitting(xyst_select_1,n_hist)
            
            ## check                
            #                plt.plot(x,y,'b+:',label='data')
            #                plt.plot(x,gaus(x,*popt),'ro:',label='fit')
            #                plt.legend()
            #                plt.show()
            #                print(popt)
                          
                            if popt[1]<np.max(x) and popt[1]>np.min(x):    
                                M_brightness[int(ix0+ixl/2),int(iy0+iyl/2)]=popt[1]
                                M_contrast[int(ix0+ixl/2),int(iy0+iyl/2)]=np.abs(popt[2])
                                xy2D.append([int(ix0+ixl/2),int(iy0+iyl/2)])
                                Mblist.append(popt[1])
                                Mclist.append(np.abs(popt[2]))
                        except RuntimeError:   
                            print('Number of calls to function has reached maxfev.')
                    elif S_W_test[0]<=fitting_thre:
        #                xhist=plt.hist(xyst_select_1,n_hist)
        #                plt.show()
                        print("S_W_test",S_W_test, "NO FITTING!",str(j))
                elif len(xyst_select_1)<3:
                    print('Not enough data!')
                 
    #  2 D Gaussion fitting
        if len(Mblist)>=9:
            xy2Dnp=np.array(xy2D).transpose()
                  
            Mblistnp=np.array(Mblist).transpose()    
            p0b=[xysize/2,xysize/2,xysize/2,xysize/2,np.max(Mblistnp)]
            try:
                popt2Db,pcov2Db = curve_fit(twoDgaussian,xy2Dnp,Mblistnp,p0=p0b)
            except RuntimeError: 
                LX=rd.sample(range(len(Mblistnp)),int(len(Mblistnp)*0.9))
                xy2Dnpr=xy2Dnp[:,LX]
                Mblistnpr=Mblistnp[LX]
                try:
                    popt2Db,pcov2Db = curve_fit(twoDgaussian,xy2Dnpr,Mblistnpr,p0=p0b)
                    file_note.write(file_name+':_slice_'+str(j)+' Brightness_0.9 data\n')
                except RuntimeError: 
                    LX=rd.sample(range(len(Mblistnp)),int(len(Mblistnp)*0.8))
                    xy2Dnpr=xy2Dnp[:,LX]
                    Mblistnpr=Mblistnp[LX]
                    try:
                        popt2Db,pcov2Db = curve_fit(twoDgaussian,xy2Dnpr,Mblistnpr,p0=p0b)
                        file_note.write(file_name+':_slice_'+str(j)+' Brightness_0.8 data\n')
                    except RuntimeError: 
                        LX=rd.sample(range(len(Mblistnp)),int(len(Mblistnp)*0.7))
                        xy2Dnpr=xy2Dnp[:,LX]
                        Mblistnpr=Mblistnp[LX]
                        try:
                            popt2Db,pcov2Db = curve_fit(twoDgaussian,xy2Dnpr,Mblistnpr,p0=p0b)
                            file_note.write(file_name+':_slice_'+str(j)+' Brightness_0.7 data\n')
                        except RuntimeError: 
                            LX=rd.sample(range(len(Mblistnp)),int(len(Mblistnp)*0.6))
                            xy2Dnpr=xy2Dnp[:,LX]
                            Mblistnpr=Mblistnp[LX]
                            try:
                                popt2Db,pcov2Db = curve_fit(twoDgaussian,xy2Dnpr,Mblistnpr,p0=p0b)
                                file_note.write(file_name+':_slice_'+str(j)+' Brightness_0.6 data\n')
                            except RuntimeError: 
                                popt2Db=np.array(p0b)
                                file_note.write(file_name+':_slice_'+str(j)+' Brightness_initial_guess\n')
           
            Mclistnp=np.array(Mclist).transpose() 
            p0c=[popt2Db[0],popt2Db[1],popt2Db[2],popt2Db[3],np.max(Mclistnp)]    
            try:
                popt2Dc,pcov2Dc = curve_fit(twoDgaussian,xy2Dnp,Mclistnp,p0=p0c)
            except RuntimeError: 
                LX=rd.sample(range(len(Mclistnp)),int(len(Mclistnp)*0.9))
                xy2Dnpr=xy2Dnp[:,LX]
                Mclistnpr=Mclistnp[LX]
                try:
                    popt2Dc,pcov2Dc = curve_fit(twoDgaussian,xy2Dnpr,Mclistnpr,p0=p0c)
                    file_note.write(file_name+':_slice_'+str(j)+' Contrast_0.9 data\n')    
                except RuntimeError: 
                    LX=rd.sample(range(len(Mclistnp)),int(len(Mclistnp)*0.8))
                    xy2Dnpr=xy2Dnp[:,LX]
                    Mclistnpr=Mclistnp[LX]
                    try:
                        popt2Dc,pcov2Dc = curve_fit(twoDgaussian,xy2Dnpr,Mclistnpr,p0=p0c)
                        file_note.write(file_name+':_slice_'+str(j)+' Contrast_0.8 data\n')    
                    except RuntimeError: 
                        LX=rd.sample(range(len(Mclistnp)),int(len(Mclistnp)*0.7))
                        xy2Dnpr=xy2Dnp[:,LX]
                        Mclistnpr=Mclistnp[LX]
                        try:
                            popt2Dc,pcov2Dc = curve_fit(twoDgaussian,xy2Dnpr,Mclistnpr,p0=p0c)
                            file_note.write(file_name+':_slice_'+str(j)+' Contrast_0.7 data\n')    
                        except RuntimeError: 
                            LX=rd.sample(range(len(Mclistnp)),int(len(Mclistnp)*0.6))
                            xy2Dnpr=xy2Dnp[:,LX]
                            Mclistnpr=Mclistnp[LX]
                            try:
                                popt2Dc,pcov2Dc = curve_fit(twoDgaussian,xy2Dnpr,Mclistnpr,p0=p0c)
                                file_note.write(file_name+':_slice_'+str(j)+' Contrast_0.6 data\n')    
                            except RuntimeError: 
                                popt2Dc=np.array(p0c) 
                                file_note.write(file_name+':_slice_'+str(j)+' Contrast_initial_guess\n')      
        
            xylattice=[]
            for ix in range(xysize):
                for iy in range(xysize):
                    xylattice.append([ix,iy])
            latticedata=np.array(xylattice).transpose()
            
            brightness=twoDgaussian(latticedata,*popt2Db)
            brightnessM=np.reshape(brightness,(xysize,xysize))
        #    io.imshow(brightnessM)
        #    plt.show()
        #    io.imsave(path_brightness+'brightness'+str(j)+'.tiff',brightnessM)
        #    io.imshow(xys)
        #    plt.show()
           
            contrast=twoDgaussian(latticedata,*popt2Dc)
            contrastM=np.reshape(contrast,(xysize,xysize))
        #    io.imshow(contrastM)
        #    plt.show()
        #    io.imsave(path_contrast+'contrast'+str(j)+'.tiff',contrastM)
        #    io.imshow(xys)
        #    plt.show()
            
            xys_modified=target_contrast*(xys-brightnessM)/contrastM+target_brightness
        elif len(Mblist)<9:
            xys_mean=np.mean(xys)
            xys_std=np.std(xys)
            xys_modified=target_contrast*(xys-xys_mean)/np.std(xys)+target_brightness
            file_note.write('!!!'+file_name+':_slice_'+str(j)+'Not modified!\n')              
            print('######################################################')         
            print('!!!'+file_name+':_slice_'+str(j)+'____Not modified!')         
            print('######################################################')
                  
        io.imshow(xys_modified)
        plt.show()    
        io.imshow(xys)
        plt.show()
        
        im_modified[j,:,:]=xys_modified
        io.imsave(Path_modified+'modified'+file_name,im_modified)
        print('...............Modified until'+' SLICE__'+str(j))

file_note.close()

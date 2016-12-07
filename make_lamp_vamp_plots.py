#!/usr/bin/python
import numpy as np
import glob
import re

print """
% line styles and colors
LISTA = {'-+','blue'};
LAMPl1 = {'-s','black'};
LAMPbg = {'-o',[.2,.5,.7]};
LAMPbguntied = {'-.o',[.1,.3,.1]};
LAMPexpo = {'-d',[.7,.5,.2]};
LAMPpwlin = {'-x',[.1,.7,.1]};
LAMPpwlinuntied = {'-.x',[.1,.3,.1]};
LAMPspline = {'->',[.9,.1,.8]};
LVAMPbg = {':o',[.7,.1,.1]};
LVAMPpwlin = {':x',[.5,.5,.1]};
matchedVAMP = {':p',[.1,.1,.1]};
SupportOracle = {'--','red'};
"""

Tmax=15

def extract( filepat, keyname='nmse_val'):
    'pull out the nmse_val= fields from a sequence of files that match filepat%(t) for t in 1...Tmax'
    vals = np.ones(Tmax)*float('nan')
    for T in range(Tmax):
        filename = filepat % (T+1)
        files = glob.glob( filename)
        if len(files) > 1:
            raise RuntimeError("several files match '%s' " % filename)
        if len(files)==1:
            for l in [ l.strip() for l in open( files[0] ).readlines() ]:
                (k,v) = l.split('=',1)
                if k == keyname:
                    vals[T] = float(v)
    return vals

def plot_traces(fig_name,traces):
    first=True;
    print """
    legs={};
    figure_named('%s');
    """  % (fig_name)

    for legend_entry,xvals,yvals in traces:
        lckey = re.sub(r'[\s -]','',legend_entry)

        yvals= '...\n[' + (','.join([repr(x) for x in yvals ])) +']...\n'
        print """
        legs={legs{:} '%s'};
        hfig=plot(%s,%s,%s{1}); set(hfig,'Color',%s{2});""" % (legend_entry,xvals,yvals,lckey,lckey )
        if first:
            first =False
            print 'hold all'

    epsname = re.sub(r'[^0-9a-z]','',fig_name.lower())
    print """
    hold off
    legend(legs{:} )
    grid minor
    xlabel('Layers')
    ylabel('NMSE (dB)')
    width = 6; set(gcf,'PaperPosition',[(8.5-width)/2,(11-0.75*width)/2,width,0.75*width])
    axis tight;axe=axis; axis([axe(1:2), floor(axe(3)),ceil(axe(4))])
    saveas(gcf,'../figures/%s.eps','epsc2')
    title(%s)
    """ % (epsname,repr(fig_name))

nmse_val={}
vamp_matched={}
lista={}
#lista_ut={}

for midname in ('Giid','k15'):
    lista[midname] = extract('data/LISTA_jun13_%s_snr40_T%s.sum' %(midname,'%d') )
    #lista_ut[midname] = extract('data/LISTA_jun13_%s_untied*_T%s.sum' %(midname,'%d') )
    vamp_matched[midname] = extract('data/VAMP_%s_matched_T%sf.sum' % ( midname,'%d'),'nmse_val')
    for firstname in ('LAMP','LAMPut','LVAMP'):
        for shrink in  'soft bg expo pwlin spline'.split():
            base='%s_%s_%s' % ( firstname,midname,shrink)
            nmse_val[base] = extract('data/%s_T%sf.sum' % ( base,'%d'),'nmse_val')

layers='1:%d'%Tmax
oracle_nmse={'Giid':-45.97,'k15':-44.7} # see amp_baseline.m for details
for midname,MidName in (('Giid','Gaussian'),('k15','kappa=15') ):
    plot_traces(
            MidName + " LAMP", (
            ('LISTA',layers,lista[midname]),
            ('LAMP-l1',layers,nmse_val['LAMP_%s_soft'%midname]),
            ('LAMP-bg',layers,nmse_val['LAMP_%s_bg'%midname]),
            ('LAMP-expo',layers,nmse_val['LAMP_%s_expo'%midname]),
            ('LAMP-pwlin',layers,nmse_val['LAMP_%s_pwlin'%midname]),
            ('LAMP-spline',layers,nmse_val['LAMP_%s_spline'%midname]),
            ('LAMP-bg-untied',layers,nmse_val['LAMPut_%s_bg'%midname]),
            ('LAMP-pwlin-untied',layers,nmse_val['LAMPut_%s_pwlin'%midname]),
            ('Support Oracle',layers,np.ones(Tmax)*oracle_nmse[midname] ),
            ) )
    plot_traces(
            MidName + " LVAMP", (
            ('LAMP-bg',layers,nmse_val['LAMP_%s_bg'%midname]),
            ('LAMP-pwlin',layers,nmse_val['LAMP_%s_pwlin'%midname]),
            ('LAMP-bg-untied',layers,nmse_val['LAMPut_%s_bg'%midname]),
            ('LAMP-pwlin-untied',layers,nmse_val['LAMPut_%s_pwlin'%midname]),
            ('LVAMP-bg',layers,nmse_val['LVAMP_%s_bg'%midname]),
            ('LVAMP-pwlin',layers,nmse_val['LVAMP_%s_pwlin'%midname]),
            ('matched VAMP',layers,vamp_matched[midname]),
            ('Support Oracle',layers,np.ones(Tmax)*oracle_nmse[midname]),
            ) )

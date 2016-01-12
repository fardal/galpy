from __future__ import print_function, division
import numpy
from galpy.util import bovy_coords
from test_streamdf import expected_failure

def test_radec_to_lb_ngp():
    # Test that the NGP is at b=90
    ra, dec= 192.25, 27.4
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[1]-90.) < 10.**-8., 'Galactic latitude of the NGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[1]-numpy.pi/2.) < 10.**-8., 'Galactic latitude of the NGP given in ra,dec is not pi/2'
    return None

def test_radec_to_lb_sgp():
    # Test that the SGP is at b=90
    ra, dec= 12.25, -27.4
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[1]+90.) < 10.**-8., 'Galactic latitude of the SGP given in ra,dec is not 90'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[1]+numpy.pi/2.) < 10.**-8., 'Galactic latitude of the SGP given in ra,dec is not pi/2'
    return None

# Test the longitude of the north celestial pole
def test_radec_to_lb_ncp():
    ra, dec= 180., 90.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(lb[0]-123.) < 10.**-8., 'Galactic longitude of the NCP given in ra,dec is not 123'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.fabs(lb[0]-123./180.*numpy.pi) < 10.**-8., 'Galactic longitude of the NCP given in ra,dec is not 123'
    # Also test the latter for vector inputs
    os= numpy.ones(2)
    lb= bovy_coords.radec_to_lb(os*ra/180.*numpy.pi,os*dec/180.*numpy.pi,
                                degree=False,epoch=1950.)
    assert numpy.all(numpy.fabs(lb[:,0]-123./180.*numpy.pi) < 10.**-8.), 'Galactic longitude of the NCP given in ra,dec is not 123'
    return None

# Test that other epochs do not work
def test_radec_to_lb_otherepochs():
    ra, dec= 180., 90.
    try:
        lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                    degree=False,epoch=1975.)   
    except IOError:
        pass
    else:
        raise AssertionError('radec functions with epoch not equal to 1950 or 2000 did not raise IOError')

# Test that radec_to_lb and lb_to_radec are each other's inverse
def test_lb_to_radec():
    ra, dec= 120, 60.
    lb= bovy_coords.radec_to_lb(ra,dec,degree=True,epoch=2000.)
    rat, dect= bovy_coords.lb_to_radec(lb[0],lb[1],degree=True,epoch=2000.)
    assert numpy.fabs(ra-rat) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.fabs(dec-dect) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    # Also test this for degree=False
    lb= bovy_coords.radec_to_lb(ra/180.*numpy.pi,dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    rat, dect= bovy_coords.lb_to_radec(lb[0],lb[1],degree=False,epoch=2000.)
    assert numpy.fabs(ra/180.*numpy.pi-rat) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.fabs(dec/180.*numpy.pi-dect) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'
    # And also test this for arrays
    os= numpy.ones(2)
    lb= bovy_coords.radec_to_lb(os*ra/180.*numpy.pi,os*dec/180.*numpy.pi,
                                degree=False,epoch=2000.)
    ratdect= bovy_coords.lb_to_radec(lb[:,0],lb[:,1],degree=False,epoch=2000.)
    rat= ratdect[:,0]
    dect= ratdect[:,1]
    assert numpy.all(numpy.fabs(ra/180.*numpy.pi-rat) < 10.**-10.), 'lb_to_radec is not the inverse of radec_to_lb'
    assert numpy.all(numpy.fabs(dec/180.*numpy.pi-dect) < 10.**-10.), 'lb_to_radec is not the inverse of radec_to_lb'   
    #Also test for a negative l
    l,b= 240., 60.
    ra,dec= bovy_coords.lb_to_radec(l,b,degree=True)
    lt,bt= bovy_coords.radec_to_lb(ra,dec,degree=True)
    assert numpy.fabs(lt-l) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'   
    assert numpy.fabs(bt-b) < 10.**-10., 'lb_to_radec is not the inverse of radec_to_lb'   
    return None

# Test lb_to_XYZ
def test_lbd_to_XYZ():
    l,b,d= 90., 30.,1.
    XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    assert numpy.fabs(XYZ[0]) <10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.fabs(XYZ[1]-numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.fabs(XYZ[2]-0.5) < 10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    # Also test for degree=False
    XYZ= bovy_coords.lbd_to_XYZ(l/180.*numpy.pi,b/180.*numpy.pi,d,degree=False)
    assert numpy.fabs(XYZ[0]) <10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.fabs(XYZ[1]-numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.fabs(XYZ[2]-0.5) < 10.**-10., 'lbd_to_XYZ conversion does not work as expected'
    # Also test for arrays
    os= numpy.ones(2)
    XYZ= bovy_coords.lbd_to_XYZ(os*l/180.*numpy.pi,os*b/180.*numpy.pi,
                                os*d,degree=False)
    assert numpy.all(numpy.fabs(XYZ[:,0]) <10.**-10.), 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.all(numpy.fabs(XYZ[:,1]-numpy.sqrt(3.)/2.) < 10.**-10.), 'lbd_to_XYZ conversion does not work as expected'
    assert numpy.all(numpy.fabs(XYZ[:,2]-0.5) < 10.**-10.), 'lbd_to_XYZ conversion does not work as expected'
    return None

# Test that XYZ_to_lbd is the inverse of lbd_to_XYZ
def test_XYZ_to_lbd():
    l,b,d= 90., 30.,1.
    XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    lt,bt,dt= bovy_coords.XYZ_to_lbd(XYZ[0],XYZ[1],XYZ[2],degree=True)
    assert numpy.fabs(lt-l) <10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.fabs(bt-b) < 10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.fabs(dt-d) < 10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    # Also test for degree=False
    XYZ= bovy_coords.lbd_to_XYZ(l/180.*numpy.pi,b/180.*numpy.pi,d,degree=False)
    lt,bt,dt= bovy_coords.XYZ_to_lbd(XYZ[0],XYZ[1],XYZ[2],degree=False)
    assert numpy.fabs(lt-l/180.*numpy.pi) <10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.fabs(bt-b/180.*numpy.pi) < 10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.fabs(dt-d) < 10.**-10., 'XYZ_to_lbd conversion does not work as expected'
    # Also test for arrays
    os= numpy.ones(2)
    XYZ= bovy_coords.lbd_to_XYZ(os*l/180.*numpy.pi,os*b/180.*numpy.pi,
                                os*d,degree=False)
    lbdt= bovy_coords.XYZ_to_lbd(XYZ[:,0],XYZ[:,1],XYZ[:,2],degree=False)
    assert numpy.all(numpy.fabs(lbdt[:,0]-l/180.*numpy.pi) <10.**-10.), 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.all(numpy.fabs(lbdt[:,1]-b/180.*numpy.pi) < 10.**-10.), 'XYZ_to_lbd conversion does not work as expected'
    assert numpy.all(numpy.fabs(lbdt[:,2]-d) < 10.**-10.), 'XYZ_to_lbd conversion does not work as expected'
    return None

def test_vrpmllpmbb_to_vxvyvz():
    l,b,d= 90., 0.,1.
    vr,pmll,pmbb= 10.,20./4.74047,-10./4.74047
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l,b,d,
                                             degree=True,XYZ=False)
    assert numpy.fabs(vxvyvz[0]+20.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[1]-10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[2]+10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l/180.*numpy.pi,
                                             b/180.*numpy.pi,d,
                                             degree=False,XYZ=False)
    assert numpy.fabs(vxvyvz[0]+20.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[1]-10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[2]+10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,0.,1,0.,
                                             XYZ=True)
    assert numpy.fabs(vxvyvz[0]+20.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[1]-10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[2]+10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,0.,1,0.,
                                             XYZ=True,degree=True)
    assert numpy.fabs(vxvyvz[0]+20.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[1]-10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxvyvz[2]+10.) < 10.**-10., 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(os*vr,os*pmll,os*pmbb,os*l,os*b,
                                             os*d,degree=True,XYZ=False)
    assert numpy.all(numpy.fabs(vxvyvz[:,0]+20.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxvyvz[:,1]-10.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxvyvz[:,2]+10.) < 10.**-10.), 'vrpmllpmbb_to_vxvyvz conversion did not work as expected'
    return None

def test_vxvyvz_to_vrpmllpmbb():
    vx,vy,vz= -20.*4.74047,10.,-10.*4.74047
    X,Y,Z= 0.,1.,0.
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,X,Y,Z,
                                                 XYZ=True)
    assert numpy.fabs(vrpmllpmbb[0]-10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[1]-20.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[2]+10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    # also try with degree=True (that shouldn't fail!)
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,X,Y,Z,
                                                 XYZ=True,
                                                 degree=True)
    assert numpy.fabs(vrpmllpmbb[0]-10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[1]-20.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[2]+10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    # also for lbd
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,90.,0.,1.,
                                                 XYZ=False,degree=True)
    assert numpy.fabs(vrpmllpmbb[0]-10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[1]-20.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[2]+10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    # also for lbd, not in degree
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,numpy.pi/2.,0.,1.,
                                                 XYZ=False,degree=False)
    assert numpy.fabs(vrpmllpmbb[0]-10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[1]-20.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.fabs(vrpmllpmbb[2]+10.) < 10.**-10., 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    # and for arrays
    os= numpy.ones(2)
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(os*vx,os*vy,os*vz,
                                                 os*numpy.pi/2.,os*0.,os,
                                                 XYZ=False,degree=False)
    assert numpy.all(numpy.fabs(vrpmllpmbb[:,0]-10.) < 10.**-10.), 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.all(numpy.fabs(vrpmllpmbb[:,1]-20.) < 10.**-10.), 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    assert numpy.all(numpy.fabs(vrpmllpmbb[:,2]+10.) < 10.**-10.), 'vxvyvz_to_vrpmllpmbb conversion did not work as expected'
    return None

def test_XYZ_to_galcenrect():
    X,Y,Z= 1.,3.,-2.
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=1.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(gcXYZ[0]) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[1]-3.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[2]+2.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    #Another test
    X,Y,Z= -1.,3.,-2.
    gcXYZ= bovy_coords.XYZ_to_galcenrect(X,Y,Z,Xsun=1.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(gcXYZ[0]-2.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[1]-3.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(gcXYZ[2]+2.) < 10.**-10., 'XYZ_to_galcenrect conversion did not work as expected'
    return None

def test_galcenrect_to_XYZ():
    gcX, gcY, gcZ= -1.,4.,2.
    XYZ= bovy_coords.galcenrect_to_XYZ(gcX,gcY,gcZ,Xsun=1.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(XYZ[0]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1]-4.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[2]-2.) < 10.**-10., 'galcenrect_to_XYZ conversion did not work as expected'
    return None

def test_XYZ_to_galcencyl():
    X,Y,Z= 5.,4.,-2.
    gcRpZ= bovy_coords.XYZ_to_galcencyl(X,Y,Z,Xsun=8.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(gcRpZ[0]-5.) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[1]-numpy.arctan(4./3.)) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[2]+2.) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    #Another X
    X,Y,Z= 11.,4.,-2.
    gcRpZ= bovy_coords.XYZ_to_galcencyl(X,Y,Z,Xsun=8.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(gcRpZ[0]-5.) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[1]-numpy.pi+numpy.arctan(4./3.)) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    assert numpy.fabs(gcRpZ[2]+2.) < 10.**-10., 'XYZ_to_galcencyl conversion did not work as expected'
    return None

def test_galcencyl_to_XYZ():
    gcR, gcp, gcZ= 5.,numpy.arctan(4./3.),2.
    XYZ= bovy_coords.galcencyl_to_XYZ(gcR,gcp,gcZ,Xsun=8.,Ysun=0.,Zsun=0.)
    assert numpy.fabs(XYZ[0]-5.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[1]-4.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    assert numpy.fabs(XYZ[2]-2.) < 10.**-10., 'galcencyl_to_XYZ conversion did not work as expected'
    return None

def test_vxvyvz_to_galcenrect():
    vx,vy,vz= 10.,-20.,30
    vgc= bovy_coords.vxvyvz_to_galcenrect(vx,vy,vz,vsun=[-5.,10.,5.])
    assert numpy.fabs(vgc[0]+15.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[1]+10.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[2]-35.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    return None

def test_vxvyvz_to_galcencyl():
    X,Y,Z= 3.,4.,2.
    vx,vy,vz= 10.,-20.,30
    vgc= bovy_coords.vxvyvz_to_galcencyl(vx,vy,vz,X,Y,Z,vsun=[-5.,10.,5.])
    assert numpy.fabs(vgc[0]+17.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[1]-6.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[2]-35.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    #with galcen=True
    vgc= bovy_coords.vxvyvz_to_galcencyl(vx,vy,vz,5.,numpy.arctan(4./3.),Z,
                                         vsun=[-5.,10.,5.],galcen=True)
    assert numpy.fabs(vgc[0]+17.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[1]-6.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    assert numpy.fabs(vgc[2]-35.) < 10.**-10., 'vxvyvz_to_galcenrect conversion did not work as expected'
    return None
    
def test_galcenrect_to_vxvyvz():
    vxg,vyg,vzg= -15.,-10.,35.
    vxyz= bovy_coords.galcenrect_to_vxvyvz(vxg,vyg,vzg,vsun=[-5.,10.,5.])
    assert numpy.fabs(vxyz[0]-10.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxyz[1]+20.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxyz[2]-30.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    #Also for arrays
    os= numpy.ones(2)
    vxyz= bovy_coords.galcenrect_to_vxvyvz(os*vxg,os*vyg,os*vzg,
                                           vsun=[-5.,10.,5.])
    assert numpy.all(numpy.fabs(vxyz[0]-10.) < 10.**-10.), 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxyz[1]+20.) < 10.**-10.), 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vxyz[2]-30.) < 10.**-10.), 'galcenrect_to_vxvyvz conversion did not work as expected'
    return None

def test_galcencyl_to_vxvyvz():
    vr,vp,vz= -17.,6.,35.
    phi= numpy.arctan(4./3.)
    vxyz= bovy_coords.galcencyl_to_vxvyvz(vr,vp,vz,phi,vsun=[-5.,10.,5.])
    assert numpy.fabs(vxyz[0]-10.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxyz[1]+20.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    assert numpy.fabs(vxyz[2]-30.) < 10.**-10., 'galcenrect_to_vxvyvz conversion did not work as expected'
    return None   

def test_sphergal_to_rectgal():
    l,b,d= 90.,0.,1.
    vr,pmll,pmbb= 10.,-20./4.74047,30./4.74047
    X,Y,Z,vx,vy,vz= bovy_coords.sphergal_to_rectgal(l,b,d,vr,pmll,pmbb,
                                                    degree=True)
    assert numpy.fabs(X-0.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(Y-1.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(Z-0.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vx-20.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vy-10.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vz-30.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    #Also test for degree=False
    X,Y,Z,vx,vy,vz= bovy_coords.sphergal_to_rectgal(l/180.*numpy.pi,
                                                    b/180.*numpy.pi,
                                                    d,vr,pmll,pmbb,
                                                    degree=False)
    assert numpy.fabs(X-0.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(Y-1.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(Z-0.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vx-20.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vy-10.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.fabs(vz-30.) < 10.**-10., 'sphergal_to_rectgal conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    XYZvxvyvz= bovy_coords.sphergal_to_rectgal(os*l,os*b,os*d,
                                                    os*vr,os*pmll,os*pmbb,
                                                    degree=True)
    X= XYZvxvyvz[:,0]
    Y= XYZvxvyvz[:,1]
    Z= XYZvxvyvz[:,2]
    vx= XYZvxvyvz[:,3]
    vy= XYZvxvyvz[:,4]
    vz= XYZvxvyvz[:,5]
    assert numpy.all(numpy.fabs(X-0.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(Y-1.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(Z-0.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(vx-20.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(vy-10.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    assert numpy.all(numpy.fabs(vz-30.) < 10.**-10.), 'sphergal_to_rectgal conversion did not work as expected'
    return None

def test_rectgal_to_sphergal():
    #Test that this is the inverse of sphergal_to_rectgal
    l,b,d= 90.,30.,1.
    vr,pmll,pmbb= 10.,-20.,30.
    X,Y,Z,vx,vy,vz= bovy_coords.sphergal_to_rectgal(l,b,d,vr,pmll,pmbb,
                                                    degree=True)
    lt,bt,dt,vrt,pmllt,pmbbt= bovy_coords.rectgal_to_sphergal(X,Y,Z,
                                                              vx,vy,vz,
                                                              degree=True)
    assert numpy.fabs(lt-l) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(bt-b) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(dt-d) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(vrt-vr) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(pmllt-pmll) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(pmbbt-pmbb) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    #Also test for degree=False
    lt,bt,dt,vrt,pmllt,pmbbt= bovy_coords.rectgal_to_sphergal(X,Y,Z,
                                                              vx,vy,vz,
                                                              degree=False)
    assert numpy.fabs(lt-l/180.*numpy.pi) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(bt-b/180.*numpy.pi) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(dt-d) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(vrt-vr) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(pmllt-pmll) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.fabs(pmbbt-pmbb) < 10.**-10., 'rectgal_to_sphergal conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    lbdvrpmllpmbbt= bovy_coords.rectgal_to_sphergal(os*X,os*Y,os*Z,
                                                              os*vx,os*vy,
                                                              os*vz,
                                                              degree=True)
    lt= lbdvrpmllpmbbt[:,0]
    bt= lbdvrpmllpmbbt[:,1]
    dt= lbdvrpmllpmbbt[:,2]
    vrt= lbdvrpmllpmbbt[:,3]
    pmllt= lbdvrpmllpmbbt[:,4]
    pmbbt= lbdvrpmllpmbbt[:,5]
    assert numpy.all(numpy.fabs(lt-l) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(bt-b) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(dt-d) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(vrt-vr) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(pmllt-pmll) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'
    assert numpy.all(numpy.fabs(pmbbt-pmbb) < 10.**-10.), 'rectgal_to_sphergal conversion did not work as expected'    
    return None

def test_pmrapmdec_to_pmllpmbb():
    #This is a random ra,dec
    ra, dec= 132., -20.4
    pmra, pmdec= 10., 20.
    pmll, pmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,
                                              ra,dec,degree=True,epoch=1950.)
    assert numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10., 'pmrapmdec_to_pmllpmbb conversion did not work as expected'
    # This is close to the NGP at 1950.
    ra, dec= 192.24, 27.39
    pmra, pmdec= 10., 20.
    os= numpy.ones(2)
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(os*pmra,os*pmdec,
                                                  os*ra,os*dec,
                                                  degree=True,epoch=1950.)
    
    pmll= pmllpmbb[:,0]
    pmbb= pmllpmbb[:,1]
    assert numpy.all(numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10.), 'pmrapmdec_to_pmllpmbb conversion did not work as expected close to the NGP'
    # This is the NGP at 1950.
    ra, dec= 192.25, 27.4
    pmra, pmdec= 10., 20.
    os= numpy.ones(2)
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(os*pmra,os*pmdec,
                                                  os*ra,os*dec,
                                                  degree=True,epoch=1950.)
    
    pmll= pmllpmbb[:,0]
    pmbb= pmllpmbb[:,1]
    assert numpy.all(numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10.), 'pmrapmdec_to_pmllpmbb conversion did not work as expected for the NGP'
    # This is the NCP
    ra, dec= numpy.pi, numpy.pi/2.
    pmra, pmdec= 10., 20.
    pmll, pmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,
                                                  ra,dec,degree=False,
                                                  epoch=1950.)
    assert numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10., 'pmrapmdec_to_pmllpmbb conversion did not work as expected for the NCP'
    return None

def test_pmllpmbb_to_pmrapmdec():
    #This is a random l,b
    ll, bb= 132., -20.4
    pmll, pmbb= 10., 20.
    pmra, pmdec= bovy_coords.pmllpmbb_to_pmrapmdec(pmll,pmbb,
                                                   ll,bb,
                                                   degree=True,epoch=1950.)
    assert numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10., 'pmllpmbb_to_pmrapmdec conversion did not work as expected for a random l,b'
    # This is close to the NGP
    ll,bb= numpy.pi-0.001, numpy.pi/2.-0.001
    pmll, pmbb= 10., 20.
    os= numpy.ones(2)
    pmrapmdec= bovy_coords.pmllpmbb_to_pmrapmdec(os*pmll,os*pmbb,
                                                 os*ll,os*bb,
                                                 degree=False,epoch=1950.)
    pmra= pmrapmdec[:,0]
    pmdec= pmrapmdec[:,1]
    assert numpy.all(numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10.), 'pmllpmbb_to_pmrapmdec conversion did not work as expected close to the NGP'
    # This is the NGP
    ll,bb= numpy.pi, numpy.pi/2.
    pmll, pmbb= 10., 20.
    os= numpy.ones(2)
    pmrapmdec= bovy_coords.pmllpmbb_to_pmrapmdec(os*pmll,os*pmbb,
                                                 os*ll,os*bb,
                                                 degree=False,epoch=1950.)
    pmra= pmrapmdec[:,0]
    pmdec= pmrapmdec[:,1]
    assert numpy.all(numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10.), 'pmllpmbb_to_pmrapmdec conversion did not work as expected at the NGP'
    # This is the NCP
    ra, dec= numpy.pi, numpy.pi/2.
    ll, bb= bovy_coords.radec_to_lb(ra,dec,degree=False,epoch=1950.)
    pmll, pmbb= 10., 20.
    pmra, pmdec= bovy_coords.pmllpmbb_to_pmrapmdec(pmll,pmbb,
                                                   ll,bb,
                                                   degree=False,epoch=1950.)
    assert numpy.fabs(numpy.sqrt(pmll**2.+pmbb**2.)-numpy.sqrt(pmra**2.+pmdec**2.)) < 10.**-10., 'pmllpmbb_to_pmrapmdec conversion did not work as expected at the NCP'
    return None

def test_cov_pmradec_to_pmllbb():
    # This is the NGP at 1950., for this the parallactic angle is 180
    ra, dec= 192.25, 27.4
    cov_pmrapmdec= numpy.array([[100.,100.],[100.,400.]])
    cov_pmllpmbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmrapmdec,
                                                       ra,dec,
                                                       degree=True,
                                                       epoch=1950.)
    
    assert numpy.fabs(cov_pmllpmbb[0,0]-100.) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    assert numpy.fabs(cov_pmllpmbb[0,1]-100.) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    assert numpy.fabs(cov_pmllpmbb[1,0]-100.) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    assert numpy.fabs(cov_pmllpmbb[1,1]-400.) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    # This is a random position, check that the conversion makes sense
    ra, dec= 132.25, -23.4
    cov_pmrapmdec= numpy.array([[100.,100.],[100.,400.]])
    cov_pmllpmbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmrapmdec,
                                                        ra/180.*numpy.pi,
                                                        dec/180.*numpy.pi,
                                                        degree=False,
                                                        epoch=1950.)
    assert numpy.fabs(numpy.linalg.det(cov_pmllpmbb)-numpy.linalg.det(cov_pmrapmdec)) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    assert numpy.fabs(numpy.trace(cov_pmllpmbb)-numpy.trace(cov_pmrapmdec)) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    # This is a random position, check that the conversion makes sense, arrays
    ra, dec= 132.25, -23.4
    icov_pmrapmdec= numpy.array([[100.,100.],[100.,400.]])
    cov_pmrapmdec= numpy.empty((3,2,2))
    for ii in range(3): cov_pmrapmdec[ii,:,:]= icov_pmrapmdec
    os= numpy.ones(3)
    cov_pmllpmbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmrapmdec,
                                                        os*ra,
                                                        os*dec,
                                                        degree=True,
                                                        epoch=1950.)
    for ii in range(3):
        assert numpy.fabs(numpy.linalg.det(cov_pmllpmbb[ii,:,:])-numpy.linalg.det(cov_pmrapmdec[ii,:,:])) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
        assert numpy.fabs(numpy.trace(cov_pmllpmbb[ii,:,:])-numpy.trace(cov_pmrapmdec[ii,:,:])) < 10.**-10., 'cov_pmradec_to_pmllbb conversion did not work as expected'
    return None

def test_cov_dvrpmllbb_to_vxyz():
    l,b,d= 90., 0., 2.
    e_d, e_vr= 0.2, 2.
    cov_pmllpmbb= numpy.array([[100.,0.],[0.,400.]])
    pmll,pmbb= 20.,30.
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(d,e_d,e_vr,
                                                  pmll,pmbb,
                                                  cov_pmllpmbb,
                                                  l,b,
                                                  degree=True,
                                                  plx=False)
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[0,0])
                      -d*4.74047*pmll*numpy.sqrt((e_d/d)**2.+(10./pmll)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[1,1])-e_vr) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[2,2])
                      -d*4.74047*pmbb*numpy.sqrt((e_d/d)**2.+(20./pmbb)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    #Another one
    l,b,d= 180., 0., 1./2.
    e_d, e_vr= 0.05, 2.
    cov_pmllpmbb= numpy.array([[100.,0.],[0.,400.]])
    pmll,pmbb= 20.,30.
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(d,e_d,e_vr,
                                                  pmll,pmbb,
                                                  cov_pmllpmbb,
                                                  l/180.*numpy.pi,
                                                  b/180.*numpy.pi,
                                                  degree=False,
                                                  plx=True)
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[0,0])-e_vr) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[1,1])
                      -1./d*4.74047*pmll*numpy.sqrt((e_d/d)**2.+(10./pmll)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    assert numpy.fabs(numpy.sqrt(cov_vxvyvz[2,2])
                      -1./d*4.74047*pmbb*numpy.sqrt((e_d/d)**2.+(20./pmbb)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    #Another one, w/ arrays
    l,b,d= 90., 90., 2.
    e_d, e_vr= 0.2, 2.
    tcov_pmllpmbb= numpy.array([[100.,0.],[0.,400.]])
    cov_pmllpmbb= numpy.empty((3,2,2))
    for ii in range(3): cov_pmllpmbb[ii,:,:]= tcov_pmllpmbb
    pmll,pmbb= 20.,30.
    os= numpy.ones(3)
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(os*d,os*e_d,os*e_vr,
                                                  os*pmll,os*pmbb,
                                                  cov_pmllpmbb,
                                                  os*l,os*b,
                                                  degree=True,
                                                  plx=False)
    for ii in range(3):
        assert numpy.fabs(numpy.sqrt(cov_vxvyvz[ii,0,0])
                          -d*4.74047*pmll*numpy.sqrt((e_d/d)**2.+(10./pmll)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
        assert numpy.fabs(numpy.sqrt(cov_vxvyvz[ii,1,1])
                          -d*4.74047*pmbb*numpy.sqrt((e_d/d)**2.+(20./pmbb)**2.)) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
        assert numpy.fabs(numpy.sqrt(cov_vxvyvz[ii,2,2])-e_vr) < 10.**-10., 'cov_dvrpmllbb_to_vxyz coversion did not work as expected'
    return None

def test_dl_to_rphi_2d():
    #This is a tangent point
    l= numpy.arcsin(0.75)
    d= 6./numpy.tan(l)
    r,phi= bovy_coords.dl_to_rphi_2d(d,l,degree=False,ro=8.,phio=0.)
    assert numpy.fabs(r-6.) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.fabs(phi-numpy.arccos(0.75)) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    #This is a different point
    d,l= 2., 45.
    r,phi= bovy_coords.dl_to_rphi_2d(d,l,degree=True,ro=2.*numpy.sqrt(2.),
                                     phio=10.)
    assert numpy.fabs(r-2.) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.fabs(phi-55.) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    #This is a different point, for array
    d,l= 2., 45.
    os= numpy.ones(2)
    r,phi= bovy_coords.dl_to_rphi_2d(os*d,os*l,degree=True,
                                     ro=2.*numpy.sqrt(2.),
                                     phio=0.)
    assert numpy.all(numpy.fabs(r-2.) < 10.**-10.), 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.all(numpy.fabs(phi-45.) < 10.**-10.), 'dl_to_rphi_2d conversion did not work as expected'
    #This is a different point, for list (which I support for some reason)
    d,l= 2., 45.
    r,phi= bovy_coords.dl_to_rphi_2d([d,d],[l,l],degree=True,
                                     ro=2.*numpy.sqrt(2.),
                                     phio=0.)
    r= numpy.array(r)
    phi= numpy.array(phi)
    assert numpy.all(numpy.fabs(r-2.) < 10.**-10.), 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.all(numpy.fabs(phi-45.) < 10.**-10.), 'dl_to_rphi_2d conversion did not work as expected'
    return None

def test_rphi_to_dl_2d():
    #This is a tangent point
    r,phi= 6., numpy.arccos(0.75)
    d,l= bovy_coords.rphi_to_dl_2d(r,phi,degree=False,ro=8.,phio=0.)
    l= numpy.arcsin(0.75)
    d= 6./numpy.tan(l)
    assert numpy.fabs(d-6./numpy.tan(numpy.arcsin(0.75))) < 10.**-10., 'dl_to_rphi_2d conversion did not work as expected'
    assert numpy.fabs(l-numpy.arcsin(0.75)) < 10.**-10., 'rphi_to_dl_2d conversion did not work as expected'
    #This is another point
    r,phi= 2., 55.
    d,l= bovy_coords.rphi_to_dl_2d(r,phi,degree=True,ro=2.*numpy.sqrt(2.),
                                   phio=10.)
    assert numpy.fabs(d-2.) < 10.**-10., 'rphi_to_dl_2d conversion did not work as expected'
    assert numpy.fabs(l-45.) < 10.**-10., 'rphi_to_dl_2d conversion did not work as expected'
    #This is another point, for arrays
    r,phi= 2., 45.
    os= numpy.ones(2)
    d,l= bovy_coords.rphi_to_dl_2d(os*r,os*phi,
                                   degree=True,ro=2.*numpy.sqrt(2.),
                                   phio=0.)
    assert numpy.all(numpy.fabs(d-2.) < 10.**-10.), 'rphi_to_dl_2d conversion did not work as expected'
    assert numpy.all(numpy.fabs(l-45.) < 10.**-10.), 'rphi_to_dl_2d conversion did not work as expected'
    #This is another point, for lists, which for some reason I support
    r,phi= 2., 45.
    d,l= bovy_coords.rphi_to_dl_2d([r,r],[phi,phi],
                                   degree=True,ro=2.*numpy.sqrt(2.),
                                   phio=0.)
    d= numpy.array(d)
    l= numpy.array(l)
    assert numpy.all(numpy.fabs(d-2.) < 10.**-10.), 'rphi_to_dl_2d conversion did not work as expected'
    assert numpy.all(numpy.fabs(l-45.) < 10.**-10.), 'rphi_to_dl_2d conversion did not work as expected'
    return None

def test_uv_to_Rz():
    u, v= numpy.arccosh(5./3.), numpy.pi/6.
    R,z= bovy_coords.uv_to_Rz(u,v,delta=3.)
    assert numpy.fabs(R-2.) < 10.**-10., 'uv_to_Rz conversion did not work as expected'
    assert numpy.fabs(z-2.5*numpy.sqrt(3.)) < 10.**-10., 'uv_to_Rz conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    R,z= bovy_coords.uv_to_Rz(os*u,os*v,delta=3.)
    assert numpy.all(numpy.fabs(R-2.) < 10.**-10.), 'uv_to_Rz conversion did not work as expected'
    assert numpy.all(numpy.fabs(z-2.5*numpy.sqrt(3.)) < 10.**-10.), 'uv_to_Rz conversion did not work as expected'
    return None

def test_Rz_to_uv():
    u, v= numpy.arccosh(5./3.), numpy.pi/6.
    ut,vt= bovy_coords.Rz_to_uv(*bovy_coords.uv_to_Rz(u,v,delta=3.),delta=3.)
    assert numpy.fabs(ut-u) < 10.**-10., 'Rz_to_uvz conversion did not work as expected'
    assert numpy.fabs(vt-v) < 10.**-10., 'Rz_to_uv conversion did not work as expected'
    #Also test for arrays
    os= numpy.ones(2)
    ut,vt= bovy_coords.Rz_to_uv(*bovy_coords.uv_to_Rz(u*os,v*os,delta=3.),delta=3.)
    assert numpy.all(numpy.fabs(ut-u) < 10.**-10.), 'Rz_to_uvz conversion did not work as expected'
    assert numpy.all(numpy.fabs(vt-v) < 10.**-10.), 'Rz_to_uv conversion did not work as expected'
    return None

def test_Rz_to_coshucosv():
    u, v= numpy.arccosh(5./3.), numpy.pi/3.
    R,z= bovy_coords.uv_to_Rz(u,v,delta=3.)
    coshu,cosv= bovy_coords.Rz_to_coshucosv(R,z,delta=3.)
    assert numpy.fabs(coshu-5./3.) < 10.**-10., 'Rz_to_coshucosv conversion did notwork as expected'
    assert numpy.fabs(cosv-0.5) < 10.**-10., 'Rz_to_coshucosv conversion did notwork as expected'
    #Also test for arrays
    os= numpy.ones(2)
    coshu,cosv= bovy_coords.Rz_to_coshucosv(R*os,z*os,delta=3.)
    assert numpy.all(numpy.fabs(coshu-5./3.) < 10.**-10.), 'Rz_to_coshucosv conversion did notwork as expected'
    assert numpy.all(numpy.fabs(cosv-0.5) < 10.**-10.), 'Rz_to_coshucosv conversion did notwork as expected'
    return None

def test_lbd_to_XYZ_jac():
    #Just position
    l,b,d= 180.,30.,2.
    jac= bovy_coords.lbd_to_XYZ_jac(l,b,d,degree=True)
    assert numpy.fabs(jac[0,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,1]-numpy.pi/180.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,2]+numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,0]+numpy.sqrt(3.)*numpy.pi/180.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,1]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,2]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,1]-numpy.sqrt(3.)*numpy.pi/180.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,2]-0.5) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    #6D
    l,b,d= 3.*numpy.pi/2.,numpy.pi/6.,2.
    vr,pmll,pmbb= 10.,20.,-30.
    jac= bovy_coords.lbd_to_XYZ_jac(l,b,d,vr,pmll,pmbb,degree=False)
    assert numpy.fabs(jac[0,0]-numpy.sqrt(3.)) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,1]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,2]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,1]-1.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,2]+numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,1]-numpy.sqrt(3.)) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,2]-0.5) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[:3,3:]) < 10.**-10.), 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,0]-numpy.sqrt(3.)/2.*vr+0.5*pmbb*d*4.74047) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,1]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,2]-pmll*4.74047) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,3]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,4]-d*4.74047) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,5]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,0]-pmll*d*4.74047) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,1]-vr/2.-numpy.sqrt(3.)/2.*d*pmbb*4.74047) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,2]-0.5*4.74047*pmbb) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,3]+numpy.sqrt(3.)/2.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,4]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,5]-4.74047) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,0]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,1]+0.5*d*4.74047*pmbb-numpy.sqrt(3.)/2.*vr) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,2]-numpy.sqrt(3.)/2.*4.74047*pmbb) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,3]-0.5) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,4]-0.) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,5]-numpy.sqrt(3.)/2.*d*4.74047) < 10.**-10., 'lbd_to_XYZ_jac calculation did not work as expected'
    return None

def test_cyl_to_rect_jac():
    #Just position
    R,phi,Z= 2., numpy.pi, 1.
    jac= bovy_coords.cyl_to_rect_jac(R,phi,Z)
    assert numpy.fabs(numpy.linalg.det(jac)-R) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,0]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,1]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,2]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,1]+2.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,2]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,1]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,2]-1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    #6D
    R,phi,Z= 2., numpy.pi, 1.
    vR,vT,vZ= 1.,2.,3.
    jac= bovy_coords.cyl_to_rect_jac(R,vR,vT,Z,vZ,phi)
    vindx= numpy.array([False,True,True,False,True,False],dtype='bool')
    assert numpy.fabs(numpy.linalg.det(jac)-R) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,0]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,5]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[0,3]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[0,vindx]) < 10.**-10.), 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,5]+2.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[1,3]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[1,vindx]) < 10.**-10.), 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,5]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[2,3]-1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[2,vindx]) < 10.**-10.), 'cyl_to_rect_jac calculation did not work as expected'
    #Velocities
    assert numpy.fabs(jac[3,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,1]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,2]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,3]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,4]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[3,5]-2.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,0]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,1]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,2]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,3]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,4]-0.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[4,5]+1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.all(numpy.fabs(jac[5,numpy.array([True,True,True,True,False,True],dtype='bool')]-0.) < 10.**-10.), 'cyl_to_rect_jac calculation did not work as expected'
    assert numpy.fabs(jac[5,4]-1.) < 10.**-10., 'cyl_to_rect_jac calculation did not work as expected'
    return None

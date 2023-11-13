
# Neng Lu
# nengl@anu.edu.au
# nenglu.geo@gmail.com
# ANU, Canberra, Australia
#
# First version 05 Aug, 2023
# Last modified 01 Sep, 2023
#
# Determine topographic relaxation for mountain loads on viscoelastic mantle and lithosphere (multiple layers). 
# The formulation is in 2D Cartesian geometry (_2D_C) and a spectrum method is used [ref. Zhong, 1997, JGR].
# The source codes are written with Fortran by Shijie Zhong. I convert the codes to Python with the package mpmath. 

import numpy as np
import mpmath as mp
mp.mp.dps = 100

nii = int(20) # for aa(4,4,nii) amd caa(nii), nii=60 in fortran codes # or just simplied to 3*nlayer?

# generate propagator matrix  
def prop1(visc, tau, ak, xx0, xx1):
    a = mp.matrix(4, 4, nii)
    ia = np.ones((4,4),dtype=int)

    tau1 = tau
    tt = ak * (xx1 - xx0)

    a[0, 0, 0] = cos1(tt) - tt * sin1(tt)
    a[0, 0, 1] = (cos1(tt) - tt * sin1(tt)) * tau
    if tau1 != 0.0:
        ia[0, 0] = 2

    a[0, 1, 0] = -tt * cos1(tt)
    a[0, 1, 1] = -tt * cos1(tt) * tau
    if tau1 != 0.0:
        ia[0, 1] = 2

    a[0, 2, 0] = (sin1(tt) - tt * cos1(tt)) / visc
    a[0, 2, 1] = 2.0 * tau * (sin1(tt) - tt * cos1(tt)) / visc
    a[0, 2, 2] = tau * tau * (sin1(tt) - tt * cos1(tt)) / visc
    if tau1 != 0.0:
        ia[0, 2] = 3

    a[0, 3, 0] = -tt * sin1(tt) / visc
    a[0, 3, 1] = -2.0 * tau * tt * sin1(tt) / visc
    a[0, 3, 2] = -tau * tau * tt * sin1(tt) / visc
    if tau1 != 0.0:
        ia[0, 3] = 3

    a[1, 0, 0] = tt * cos1(tt)
    a[1, 0, 1] = tau * tt * cos1(tt)
    if tau1 != 0.0:
        ia[1, 0] = 2

    a[1, 1, 0] = cos1(tt) + tt * sin1(tt)
    a[1, 1, 1] = (cos1(tt) + tt * sin1(tt)) * tau
    if tau1 != 0.0:
        ia[1, 1] = 2

    a[1, 2, 0] = tt * sin1(tt) / visc
    a[1, 2, 1] = 2 * tau * tt * sin1(tt) / visc
    a[1, 2, 2] = tau * tau * tt * sin1(tt) / visc
    if tau1 != 0.0:
        ia[1, 2] = 3

    a[1, 3, 0] = (sin1(tt) + tt * cos1(tt)) / visc
    a[1, 3, 1] = 2 * tau * (sin1(tt) + tt * cos1(tt)) / visc
    a[1, 3, 2] = tau * tau * (sin1(tt) + tt * cos1(tt)) / visc
    if tau1 != 0.0:
        ia[1, 3] = 3

    a[2, 0, 0] = (sin1(tt) - tt * cos1(tt)) * visc
    ia[2, 0] = 1

    a[2, 1, 0] = -tt * sin1(tt) * visc
    ia[2, 1] = 1

    a[2, 2, 0] = cos1(tt) - tt * sin1(tt)
    a[2, 2, 1] = (cos1(tt) - tt * sin1(tt)) * tau
    if tau1 != 0.0:
        ia[2, 2] = 2

    a[2, 3, 0] = -tt * cos1(tt)
    a[2, 3, 1] = -tt * cos1(tt) * tau
    if tau1 != 0.0:
        ia[2, 3] = 2

    a[3, 0, 0] = tt * sin1(tt) * visc
    ia[3, 0] = 1

    a[3, 1, 0] = (sin1(tt) + tt *  cos1(tt)) * visc
    ia[3, 1] = 1

    a[3, 2, 0] = tt * cos1(tt)
    a[3, 2, 1] = tt * cos1(tt) * tau
    if tau1 != 0.0:
        ia[3, 2] = 2

    a[3, 3, 0] = cos1(tt) + tt * sin1(tt)
    a[3, 3, 1] = (cos1(tt) + tt * sin1(tt)) * tau
    if tau1 != 0.0:
        ia[3, 3] = 2
    return a,ia

def cos1(x):
    one = 1.0
    two = 2.0
    return (one + mp.exp(-two * x)) / two

def sin1(x):
    one = 1.0
    two = 2.0
    return (one - mp.exp(-two * x)) / two
    
# functions for matrix calculation    
def get_unit_matrix():
    aa = mp.matrix(4, 4, nii)
    iaa = np.ones((4,4),dtype=int)
    for i in range(4):
        for j in range(4):
            if i == j:
                aa[i, j, 0] = 1
    return aa, iaa

# (p2, ip2, aa, iaa)
def matrix_multi_mat(ain1, iain1, ain2, iain2):
    out1 = mp.matrix(4, 4, nii)
    iout1 = np.ones((4,4),dtype=int)
    out = mp.matrix(4, 4, nii)
    iout = np.ones((4,4),dtype=int)
    
    for i in range(4):
        for j in range(4):
            iout1[i,j]=1
            for k in range(4):
                iout1[i,j] = max(iout1[i,j], iain1[i, k] + iain2[k, j] - 1)
                for ii in range(iain1[i, k]):
                    for jj in range(iain2[k, j]):
                        kk = ii + jj   
                        out1[i, j, kk] += ain1[i, k, ii] * ain2[k, j, jj]
                        
    for i in range(4):
        for j in range(4):
            outtt = out1[i, j, iout1[i, j]-1]
            if outtt == 0.0:
                iout1[i, j] -= 1
            iout[i, j] = int(iout1[i, j])
            for k in range(iout1[i, j]):
                out[i, j, k] = out1[i, j, k]
    iout[iout< 1] = 1
    return out,iout    

def multip_s(temp, itemp,c, ic):
    out1 = mp.matrix(nii,1)
    out = mp.matrix(nii,1)
    iout1 = ic + itemp - 1
    for i in range(ic):
        for j in range(itemp):
            k = i + j
            out1[k] += c[i] * temp[j]
            
    iout = iout1
    for k in range(iout):
        out[k] = out1[k]
    return out, iout

def add_s(c, ic, temp, itemp):
    out = mp.matrix(nii,1)
    for k in range(nii):
        out[k] = 0.0
        if k >= ic:
            c[k] = 0.0
        if k >= itemp:
            temp[k] = 0.0
    iout = max(ic, itemp)
    for i in range(iout):
        out[i] = c[i] + temp[i]
    return out, iout

def add_ss(out, iout,oin, ioin, i):
    for k in range(nii):
        if k >= iout:
            out[k] = 0.0
        if k >= ioin:
            oin[k] = 0.0
    iout = max(iout, ioin)

    for k in range(iout):
        if i == 1:
            out[k] = out[k] + oin[k]
        else:
            out[k] = out[k] - oin[k]
    return out, iout
        
# system equations
def form_alphas(nlayers,tau,ak,imark,am):
    # for the caa=alfa1*alfa2*alfa3.  cbb=alfa5*alfa4
    icaa = 1
    icbb = 1
    caa = mp.matrix(nii,1)
    cbb = mp.matrix(nii,1)
    temp = mp.matrix(nii,1)
    caa[0] = 1.0
    cbb[0] = 1.0

    for i in range(nlayers):
        if imark[i] == 0:
            temp[0] = 1.0
            temp[1] = tau[i]
            itemp = 2
            caa, icaa = multip_s(temp, itemp,caa, icaa)
        elif imark[i] == 1:
            temp[0] = 1.0
            temp[1] = tau[i]
            itemp = 2
            cbb, icbb = multip_s(temp, itemp,cbb, icbb)
            
    temp[0] = 1/mp.exp(ak*am)
    for k in range(icaa):
        caa[k] = temp[0]*caa[k]
    temp[0] = 1/mp.exp(ak*(1-am))
    for k in range(icbb):
        cbb[k] = temp[0]*cbb[k]
    return caa,icaa,cbb,icbb
    
# form the 4x4 matrix of the eigen-equations
def form_dett(aa,iaa,bb,ibb,cc,icc,ak,r0,delta_rho,qqq,qqq1):
    dett = mp.matrix(4,4,nii)
    idett = np.ones((4,4),dtype=int)
    
    # for the first two columns
    for k in range(nii):
        if k == 0:
            down = 0.0
            j = k
        else:
            down = 1.0
            j = k - 1
        dett[0,0,k] = down * aa[0,0,j] + aa[0,2,k] * qqq / (2.0 * ak)
        dett[1,0,k] = down * bb[0,0,j] + bb[0,2,k] * qqq / (2.0 * ak)
        dett[2,0,k] = down * bb[2,0,j] + bb[2,2,k] * qqq / (2.0 * ak)
        dett[3,0,k] = down * bb[3,0,j] + bb[3,2,k] * qqq / (2.0 * ak)

        dett[0,1,k] = aa[0,1,k]
        dett[1,1,k] = bb[0,1,k]
        dett[2,1,k] = bb[2,1,k]
        dett[3,1,k] = bb[3,1,k]
    idett[0,0] = max(iaa[0,0] + 1, iaa[0,2])
    idett[1,0] = max(ibb[0,0] + 1, ibb[0,2])
    idett[2,0] = max(ibb[2,0] + 1, ibb[2,2])
    idett[3,0] = max(ibb[3,0] + 1, ibb[3,2])
    idett[0,1] = iaa[0,1]
    idett[1,1] = ibb[0,1]
    idett[2,1] = ibb[2,1]
    idett[3,1] = ibb[3,1]

    # for the third column
    idett[0,2] = 2
    idett[1,2] = icc[0,2]
    idett[2,2] = icc[2,2]
    idett[3,2] = icc[3,2]

    dett[0,2,0] = 0.0
    dett[0,2,1] = -1.0
    for k in range(nii):
        dett[1,2,k] = cc[0,2,k] * r0 * delta_rho / (2.0 * ak)
        dett[2,2,k] = cc[2,2,k] * r0 * delta_rho / (2.0 * ak)
        dett[3,2,k] = cc[3,2,k] * r0 * delta_rho / (2.0 * ak)

    # For the last column
    idett[1,3] = 2
    dett[1,3,0] = 0.0
    dett[1,3,1] = -1.0

    idett[2,3] = 1
    dett[2,3,0] = -qqq1 / (2.0 * ak)

    return dett, idett
 
# form constants of the eigen-equations
def form_coeff(dett,idett):
    temp = mp.matrix(3,3,nii)
    itemp = np.ones((3,3),dtype=int)
    temp1 = mp.matrix(nii,1)
    # Co-determinant of (2,4)
    for j in range(3):
        for k in range(idett[0,j]):
            temp[0,j,k] = dett[0,j,k]
        itemp[0,j] = idett[0,j]
        for k in range(idett[2,j]):
            temp[1,j,k] = dett[2,j,k]
        itemp[1,j] = idett[2,j]

        for k in range(idett[3,j]):
            temp[2,j,k] = dett[3,j,k]
        itemp[2,j] = idett[3,j]
    out,iout = determinant3_s(temp,itemp)
    for k in range(idett[1,3]):
        temp1[k] = dett[1,3,k]
    itemp1 = idett[1,3]
    out1,iout1 = multip_s(out,iout,temp1,itemp1)

    # Co-determinant of (3,4)
    for j in range(3):
        for k in range(idett[0,j]):
            temp[0,j,k] = dett[0,j,k]
        itemp[0,j] = idett[0,j]

        for k in range(idett[1,j]):
            temp[1,j,k] = dett[1,j,k]
        itemp[1,j] = idett[1,j]

        for k in range(idett[3,j]):
            temp[2,j,k] = dett[3,j,k]
        itemp[2,j] = idett[3,j]

    out,iout = determinant3_s(temp,itemp)
    for k in range(idett[2,3]):
        temp1[k] = -dett[2,3,k]
    itemp1 = idett[2,3]
    out2,iout2 = multip_s(out,iout,temp1,itemp1)
    co,ico = add_s(out1,iout1,out2,iout2)
    return co,ico
 
def determinant3_s(tin,itin):
    a11 = mp.matrix(nii,1)
    a12 = mp.matrix(nii,1)
    a13 = mp.matrix(nii,1)
    a21 = mp.matrix(nii,1)
    a22 = mp.matrix(nii,1)
    a23 = mp.matrix(nii,1)
    a31 = mp.matrix(nii,1)
    a32 = mp.matrix(nii,1)
    a33 = mp.matrix(nii,1)
    for k in range(itin[0, 0]):
        a11[k]=tin[0, 0, k]
    for k in range(itin[0, 1]):
        a12[k]=tin[0, 1, k]
    for k in range(itin[0, 2]):
        a13[k]=tin[0, 2, k]
    for k in range(itin[1, 0]):
        a21[k]=tin[1, 0, k]
    for k in range(itin[1, 1]):
        a22[k]=tin[1, 1, k]
    for k in range(itin[1, 2]):
        a23[k]=tin[1, 2, k]
    for k in range(itin[2, 0]):
        a31[k]=tin[2, 0, k]
    for k in range(itin[2, 1]):
        a32[k]=tin[2, 1, k]
    for k in range(itin[2, 2]):
        a33[k]=tin[2, 2, k]

    temp, itemp = multip_s(a11, itin[0,0], a22, itin[1,1])
    out1, iout1 = multip_s(temp, itemp, a33, itin[2,2])

    temp, itemp = multip_s(a21, itin[1,0], a32, itin[2,1])
    out2, iout2 = multip_s(temp, itemp, a13, itin[0,2])

    temp, itemp = multip_s(a12, itin[0,1], a23, itin[1,2])
    out3, iout3 = multip_s(temp, itemp, a31, itin[2,0])

    temp, itemp = multip_s(a13, itin[0,2], a22, itin[1,1])
    out4, iout4 = multip_s(temp, itemp, a31, itin[2,0])

    temp, itemp = multip_s(a11, itin[0,0], a32, itin[2,1])
    out5, iout5 = multip_s(temp, itemp, a23, itin[1,2])

    temp, itemp = multip_s(a33, itin[2,2], a12, itin[0,1])
    out6, iout6 = multip_s(temp, itemp, a21, itin[1,0])
    
    out, iout = add_s(out1, iout1, out2, iout2)
    out, iout = add_ss(out, iout,out3, iout3, 1)
    out, iout = add_ss(out, iout,out4, iout4, 0)
    out, iout = add_ss(out, iout,out5, iout5, 0)
    out, iout = add_ss(out, iout,out6, iout6, 0)

    return out, iout
    
def eigenvalues(co,ico,dett,idett):
    EPS = 1e-7
    #its = 10000
    zero = 0.0
    
    coi = co[:ico]
    
    ac = mp.matrix(int(ico),1)
    ad = mp.matrix(int(ico),1)
    for i in range(ico):
        ac[i] = mp.mpc(co[i], zero)
        ad[i] = mp.mpc(co[i], zero)
    m = int(ico - 1)
    
    xxc = mp.matrix(m,1)
    xx = mp.matrix(m,1)

    for j in range(m-1, -1, -1):
        x = mp.mpc(zero,zero)
        x = laguer(ad, j+1,x)
        ximag = abs(x.imag)
        xreal = abs(x.real)
        if ximag <= xreal * EPS:
            x = mp.mpc(x.real, zero)
        xxc[j] = x
        b = ad[j+1]
        for jj in range(j, -1, -1):
            c = ad[jj]
            ad[jj] = b
            b = x * b + c
        
    for j in range(m):
        xxc[j] = laguer(ac, m, xxc[j]) 
        
#     dettf = mp.matrix(4,4,m)
#     for ie in range(m):
#         xx[ie] = xxc[ie].real
#         dettf = get_dettf(xx[ie], ie, dett,idett)   
#         #dettf[:,:,ie] = tempdettf
    xx,dettf = get_dettf(xxc, m, dett,idett) 
    return xx,dettf

def laguer(a, m,x):
    maxit = 8 * 100
    EPSS = 1e-26 
    zero = 0.0

    for iter_ in range(maxit):
        its = iter_
        b = a[m]
        err = abs(b)
        d = mp.mpc(zero,zero)
        f = mp.mpc(zero,zero)
        abx = abs(x)
        for j in range(m-1, -1, -1):
            f = x * f + d
            d = x * d + b
            b = x * b + a[j]
            err = abs(b) + abx * err
        err *= EPSS
        errt = abs(b)
        if errt > err:
            g = d / b
            g2 = g * g
            h = g2 - 2.0 * f / b
            sq = mp.sqrt((m - 1.0) * (m * h - g2))
            gp = g + sq
            gm = g - sq
            abp = abs(gp)
            abm = abs(gm)
            if abp < abm:
                gp = gm
            if max(abp, abm) > 0.0:
                dx = m / gp
            else:
                temp = iter_+1.0
                dx = mp.exp(mp.log(1.0 + abx)) * mp.mpc(mp.cos(temp),mp.sin(temp))
            x1 = x - dx
            x = x1
        else:
            break
    return x 

def get_dettf(xxc,m,dett,idett):
    dettf = mp.matrix(4,4,m)
    x = mp.matrix(m,1)
    for ie in range(m):
        x[ie] = xxc[ie].real
        temp = mp.matrix(nii,1)
        for i in range(4):
            for j in range(4):
                for k in range(idett[i,j]):
                    temp[k] = dett[i,j,k]
                dettf[i,j,ie] = evalue(temp, idett[i,j], x[ie])
    return x,dettf
                           
def evalue(caa,icaa,x):
    value = caa[0]
    for i in range(2,icaa+1):
        value += caa[i-1]*(x**(i-1))
    return value
        
def eigenfunctions(xx,ico,n,dettf,caa,icaa,cbb,icbb,f,co):
    eigenf = mp.matrix(int(ico-1),1) 
    if n == 1:
        for i in range(ico-1):
            eigenf[i] = residue1(xx[i], i,dettf,caa,icaa,cbb,icbb,f,co,ico)
    elif n == 2:
        for i in range(ico-1):
            eigenf[i] = residue2(xx[i], i,dettf,caa,icaa,cbb,icbb,f,co,ico)
    elif n == 3:
        for i in range(ico-1):
            eigenf[i] = residue3(xx[i], i,dettf,caa,icaa,cbb,icbb,f,co,ico)
    elif n == 4:
        for i in range(ico-1):
            eigenf[i] = residue4(xx[i], i,dettf,caa,icaa,cbb,icbb,f,co,ico)
    return eigenf  
    
# eigenfunction for surface topography 
def residue1(x, ie,dettf,caa,icaa,cbb,icbb,f,co,ico):
    upm = mp.matrix(4,4)

    upm[0,0] = dettf[0,0,ie]
    upm[1,0] = dettf[1,0,ie]
    upm[2,0] = dettf[2,0,ie]
    upm[3,0] = dettf[3,0,ie]
    upm[0,1] = dettf[0,1,ie]
    upm[1,1] = dettf[1,1,ie]
    upm[2,1] = dettf[2,1,ie]
    upm[3,1] = dettf[3,1,ie]
    upm[0,2] = dettf[0,2,ie]
    upm[1,2] = dettf[1,2,ie]
    upm[2,2] = dettf[2,2,ie]
    upm[3,2] = dettf[3,2,ie]

    const1 = evalue(cbb, icbb, x)

    #if mantle_s == 0:
    upm[0,3] = -f[0] / const1
    upm[1,3] = -f[1]
    upm[2,3] = 0.0
    upm[3,3] = 0.0

    up = determinant(upm)
    down = co[1]
    for i in range(2, ico):
        down += i * co[i] * (x ** (i - 1))
    residue1 = up / down
    return residue1

# moho topography 
def residue2(x, ie,dettf,caa,icaa,cbb,icbb,f,co,ico):
    upm = mp.matrix(4,4)

    upm[0,0] = dettf[0,0,ie]
    upm[1,0] = dettf[1,0,ie]
    upm[2,0] = dettf[2,0,ie]
    upm[3,0] = dettf[3,0,ie]
    upm[0,1] = dettf[0,1,ie]
    upm[1,1] = dettf[1,1,ie]
    upm[2,1] = dettf[2,1,ie]
    upm[3,1] = dettf[3,1,ie]
    upm[0,3] = dettf[0,3,ie]
    upm[1,3] = dettf[1,3,ie]
    upm[2,3] = dettf[2,3,ie]
    upm[3,3] = dettf[3,3,ie]

    const1 = evalue(cbb, icbb, x)
    
    #if mantle_s == 0:
    upm[0,2] = -f[0]  
    upm[1,2] = -f[1]*const1
    upm[2,2] = 0.0
    upm[3,2] = 0.0

    up = determinant(upm)
    down = co[1]
    for i in range(2, ico):
        down += i * co[i] * (x ** (i - 1))

    residue2 = up / down
    return residue2

# bottom topography
def residue3(x, ie,dettf,caa,icaa,cbb,icbb,f,co,ico):
    upm = mp.matrix(4,4)

    upm[0,1] = dettf[0,1,ie]
    upm[1,1] = dettf[1,1,ie]
    upm[2,1] = dettf[2,1,ie]
    upm[3,1] = dettf[3,1,ie]
    upm[0,2] = dettf[0,2,ie]
    upm[1,2] = dettf[1,2,ie]
    upm[2,2] = dettf[2,2,ie]
    upm[3,2] = dettf[3,2,ie]
    upm[0,3] = dettf[0,3,ie]
    upm[1,3] = dettf[1,3,ie]
    upm[2,3] = dettf[2,3,ie]
    upm[3,3] = dettf[3,3,ie]

    const1 = evalue(caa, icaa, x)
    const2 = evalue(cbb, icbb, x)*const1
    
    #if mantle_s == 0:
    upm[0,0] = -f[0]*const1  
    upm[1,0] = -f[1]*const2
    upm[2,0] = 0.0
    upm[3,0] = 0.0

    up = determinant(upm)
    down = co[1]
    for i in range(2, ico):
        down += i * co[i] * (x ** (i - 1))

    residue3 = up / down
    return residue3

# bottom horizontal velocsity
def residue4(x, ie, dettf,caa,icaa,cbb,icbb,f,co,ico):
    upm = mp.matrix(4,4)

    upm[0,0] = dettf[0,0,ie]
    upm[1,0] = dettf[1,0,ie]
    upm[2,0] = dettf[2,0,ie]
    upm[3,0] = dettf[3,0,ie]
    upm[0,2] = dettf[0,2,ie]
    upm[1,2] = dettf[1,2,ie]
    upm[2,2] = dettf[2,2,ie]
    upm[3,2] = dettf[3,2,ie]
    upm[0,3] = dettf[0,3,ie]
    upm[1,3] = dettf[1,3,ie]
    upm[2,3] = dettf[2,3,ie]
    upm[3,3] = dettf[3,3,ie]

    const1 = evalue(caa, icaa, x)
    const2 = evalue(cbb, icbb, x)*const1
    #if mantle_s == 0:
    
    upm[0,1] = -f[0]*const1  
    upm[1,1] = -f[1]*const1
    upm[2,1] = 0.0
    upm[3,1] = 0.0

    up = determinant(upm)
    down = co[1]
    for i in range(2, ico):
        down += i * co[i] * (x ** (i - 1))

    residue4 = up / down
    return residue4    

def determinant(a):
    return a[0, 0] * sub3(a[1, 1], a[1, 2], a[1, 3], a[2, 1], a[2, 2], a[2, 3], a[3, 1], a[3, 2], a[3, 3]) \
         - a[1, 0] * sub3(a[0, 1], a[0, 2], a[0, 3], a[2, 1], a[2, 2], a[2, 3], a[3, 1], a[3, 2], a[3, 3]) \
         + a[2, 0] * sub3(a[0, 1], a[0, 2], a[0, 3], a[1, 1], a[1, 2], a[1, 3], a[3, 1], a[3, 2], a[3, 3]) \
         - a[3, 0] * sub3(a[0, 1], a[0, 2], a[0, 3], a[1, 1], a[1, 2], a[1, 3], a[2, 1], a[2, 2], a[2, 3])
def sub3(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    return a11 * a22 * a33 + a21 * a32 * a13 + a12 * a23 * a31 \
           - a13 * a22 * a31 - a11 * a32 * a23 - a33 * a12 * a21

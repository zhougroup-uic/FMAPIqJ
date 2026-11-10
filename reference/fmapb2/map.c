#include "rdf.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

/* One dimension mapping into the PBC box */
int PBC(int x, int mx) {
    int px;
    if (x >= 0 && x < mx) {
        return x;
    } else {
        px = x % mx;
        return px >= 0 ? px : px + mx;
    }
}

int signI(double x) {
    return (x < 0.0) ? -1 : 1;
}


void Map2GrdR(int l, fftw_real* grid, double xyz[3], const double coeff, const double sys[4], double (*rdf)(double,double)) {
    fftw_real (*grd)[l][2*(l/2+1)] = (void*)grid;
    int xyzmin[3], xyzmax[3];
    int i, j, k, px, py, pz;
    double x, y, z, x2, y2, z2, r2;
    const double rlow = sys[0];
    const double rup = sys[1];
    const double dx = sys[2];
    const double rup2 = rup * rup;
    const double rlow2 = rlow * rlow;
    const double scl = sys[3];
    
    // Calculate the bounding box
    for (i = 0; i < 3; i++) {
        xyzmin[i] = (int)floor(xyz[i] - rup/dx);
        xyzmax[i] = (int)ceil(xyz[i] + rup/dx);
    }
    
    // Spherical iteration with early exits
    for (i = xyzmin[0]; i <= xyzmax[0]; i++) {
        x = (i - xyz[0]) * dx;  // Distance from atom to grid point
        x2 = x * x;
        if (x2 > rup2) continue;  // Early exit for x
        px = PBC(i, l);
        
        for (j = xyzmin[1]; j <= xyzmax[1]; j++) {
            y = (j - xyz[1]) * dx;  // Distance from atom to grid point
            y2 = y * y;
            double xy2 = x2 + y2;
            if (xy2 > rup2) continue;  // Early exit for x+y
            py = PBC(j, l);
            
            for (k = xyzmin[2]; k <= xyzmax[2]; k++) {
                z = (k - xyz[2]) * dx;  // Distance from atom to grid point
                z2 = z * z;
                r2 = xy2 + z2;
                if (r2 > rup2) continue;  // Final distance check
                
                pz = PBC(k, l);
                double var;
                if (r2 > rlow2) {
                    var = coeff * rdf(r2, scl);
                } else {
                    var = coeff * rdf(rlow2, scl);
                }
                #pragma omp atomic
                grd[px][py][pz] += var;
            }
        }
    }
}

// Last two parameters are useless; make Map2GrdL same as Map2GrdR
void Map2GrdL(int l, fftw_real* grid, double xyz[3], const double coeff, const double sys[4], double (*rdf)(double,double)) {
    fftw_real (*grd)[l][2*(l/2+1)] = (void*)grid;
    int px, py, pz;
    int pxn, pyn, pzn;
    double a, b, c;
    double av, bv, cv;
    double xyzi[3], xyzf[3];
    int i;
    
    for (i = 0; i < 3; i++) {
        xyzf[i] = modf(xyz[i], &xyzi[i]);
    }
    
    px = PBC((int)xyzi[0], l);
    py = PBC((int)xyzi[1], l);
    pz = PBC((int)xyzi[2], l);
    a = (double)xyzf[0];
    b = (double)xyzf[1];
    c = (double)xyzf[2];
    
    pxn = PBC(signI(a) + px, l);
    pyn = PBC(signI(b) + py, l);
    pzn = PBC(signI(c) + pz, l);
    
    av = fabs(a);
    bv = fabs(b);
    cv = fabs(c);
    
    #pragma omp atomic
    grd[px][py][pz] += coeff * (1.0 - av) * (1.0 - bv) * (1.0 - cv);
    #pragma omp atomic
    grd[px][py][pzn] += coeff * (1.0 - av) * (1.0 - bv) * cv;
    #pragma omp atomic
    grd[px][pyn][pz] += coeff * (1.0 - av) * bv * (1.0 - cv);
    #pragma omp atomic
    grd[px][pyn][pzn] += coeff * (1.0 - av) * bv * cv;
    #pragma omp atomic
    grd[pxn][py][pz] += coeff * av * (1.0 - bv) * (1.0 - cv);
    #pragma omp atomic
    grd[pxn][py][pzn] += coeff * av * (1.0 - bv) * cv;
    #pragma omp atomic
    grd[pxn][pyn][pz] += coeff * av * bv * (1.0 - cv);
    #pragma omp atomic
    grd[pxn][pyn][pzn] += coeff * av * bv * cv;
}

/* 10-point second-order charge spreading (Qin & Zhou, JCTC 2014, 10, 2824), closed form.
   The weights reproduce the charge, its first and second moments exactly. Per axis: c = nearest grid point, s = +1/-1 toward the
   charge, u,v,w in [0,1/2] = distances to c along x,y,z (y and z by permuting):
     c: 1-u^2-v^2-w^2+uv+uw+vw    c-s_x ex: u(u-1)/2    c+s_x ex: u(u+1)/2-u(v+w)    c+s_y ey+s_z ez: vw (and the other two diagonals)
   (replaces the lattice.txt table of 1000 sub-grid positions) */
void Map2GrdL2nd(int l, fftw_real* grid, double xyz[3], const double coeff, const double sys[4], double (*rdf)(double,double)) {
    fftw_real (*grd)[l][2*(l/2+1)] = (void*)grid;
    int p0[3], pn[3], pf[3];   /* nearest grid point, its neighbour toward the charge, its neighbour away from it */
    double d[3];
    for (int i = 0; i < 3; i++) {
        double lo = floor(xyz[i]), f = xyz[i] - lo;
        int near = f < 0.5;
        d[i] = near ? f : 1.0 - f;
        /* wrap the nearest grid point once; its two neighbours follow by a compare */
        p0[i] = PBC((int)lo + (near ? 0 : 1), l);
        int up = p0[i] + 1 == l ? 0 : p0[i] + 1;
        int dn = p0[i] == 0 ? l - 1 : p0[i] - 1;
        pn[i] = near ? up : dn;
        pf[i] = near ? dn : up;
    }
    const double u = d[0], v = d[1], w = d[2];
    const double uv = u*v, uw = u*w, vw = v*w;
    const int x0 = p0[0], y0 = p0[1], z0 = p0[2];
    const int xn = pn[0], yn = pn[1], zn = pn[2];
    const int xf = pf[0], yf = pf[1], zf = pf[2];
    #pragma omp atomic
    grd[x0][y0][z0] += coeff * (1.0 - u*u - v*v - w*w + uv + uw + vw);
    #pragma omp atomic
    grd[xn][y0][z0] += coeff * (u*(u+1.0)/2 - uv - uw);
    #pragma omp atomic
    grd[xf][y0][z0] += coeff * (u*(u-1.0)/2);
    #pragma omp atomic
    grd[x0][yn][z0] += coeff * (v*(v+1.0)/2 - uv - vw);
    #pragma omp atomic
    grd[x0][yf][z0] += coeff * (v*(v-1.0)/2);
    #pragma omp atomic
    grd[x0][y0][zn] += coeff * (w*(w+1.0)/2 - uw - vw);
    #pragma omp atomic
    grd[x0][y0][zf] += coeff * (w*(w-1.0)/2);
    #pragma omp atomic
    grd[x0][yn][zn] += coeff * vw;
    #pragma omp atomic
    grd[xn][y0][zn] += coeff * uw;
    #pragma omp atomic
    grd[xn][yn][z0] += coeff * uv;
}

// Volume grid functions
void volGrd(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4],
        void (*map)(int,fftw_real*,double[3],const double, const double[4],double(*)(double,double)),
        double (*rdf)(double,double)) {
    int i;

    #pragma omp parallel for 
    for (i = 0; i < nAtom; i++) { 
        double rc2 = atoms[i].r * atoms[i].r;
        double sys_local[4] = {sys[0], atoms[i].r, sys[2], rc2};
        map(l, grd, atoms[i].xyz, 1.0, sys_local, rdf);
    }
}

void volGrdR(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4]) {
    volGrd(l, grd, nAtom, atoms, sys, Map2GrdR, rdf_vol);
}

void debyeGrd(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4],
        void (*map)(int,fftw_real*,double[3],const double, const double[4],double(*)(double,double)),
        double (*rdf)(double,double)) {
    int i;

    #pragma omp parallel for 
    for (i = 0; i < nAtom; i++) {
        map(l, grd, atoms[i].xyz, atoms[i].q, sys, rdf);
    }
}

void debyeGrdR(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4]) {
    debyeGrd(l, grd, nAtom, atoms, sys, Map2GrdR, rdf_debye);
}

void debyeGrdL2nd(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4]) {
    debyeGrd(l, grd, nAtom, atoms, sys, Map2GrdL2nd, NULL);
}

// Van der Waals grid functions
void vdwGrd(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4],
        void (*map)(int,fftw_real*,double[3],const double, const double[4],double(*)(double,double)),
        double (*rdf)(double,double), double n, double sign) {
    int i;

    #pragma omp parallel for 
    for (i = 0; i < nAtom; i++) {
        double sys_local[4] = {sys[0], sys[1], sys[2], n};
        double coeff = (n < 4.0) ? atoms[i].Bsq : atoms[i].Asq;
        map(l, grd, atoms[i].xyz, sign * coeff, sys_local, rdf);
    }
}

void r6GrdR(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4]) {
    vdwGrd(l, grd, nAtom, atoms, sys, Map2GrdR, rdf_vdw, 3.0, -1.0);
}

void r12GrdR(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4]) {
    vdwGrd(l, grd, nAtom, atoms, sys, Map2GrdR, rdf_vdw, 6.0, 1.0);
}

void r6GrdL(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4]) {
    vdwGrd(l, grd, nAtom, atoms, sys, Map2GrdL, NULL, 3.0, 1.0);
}

void r12GrdL(int l, fftw_real *grd, int nAtom, ATOM atoms[], double sys[4]) {
    vdwGrd(l, grd, nAtom, atoms, sys, Map2GrdL, NULL, 6.0, 1.0);
}

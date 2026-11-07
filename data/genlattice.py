#!/usr/bin/env python3
# Generates data/lattice.txt: weights of the 10-point second-order charge spreading (Qin & Zhou, JCTC 2014, 10, 2824).
# Usage: python3 genlattice.py > lattice.txt      (1000 lines, 10 "index weight" pairs each; index=(i+1)*16+(j+1)*4+(k+1))
# The weights reproduce the moments up to second order (rows of A: 1, dx, dy, dz, dx^2, dy^2, dz^2, dx*dy, dx*dz, dy*dz).
# Note: an earlier version used dx*dz twice (no dy*dz row), which left the y*z moment unfixed (errors up to 0.34).

import numpy as np
from sys import float_info
import sys

def xy2A(x,y):
        lst=[]
        A=np.zeros((6,6),'d')
        assert(x>=0 and x<1.0)
        assert(y>=0 and y<1.0)
        n=0
        for xi in range(-1,3):
                dx=xi-x
                ax=abs(dx)
                if ax>=1.5:
                        continue
                for yi in range(-1,3):
                        dy=yi-y
                        ay=abs(dy)
                        if ay>=1.5:
                                continue
                        if (ax<1.0 and ay<1.0) or (ax<=0.5) or (ay<=0.5) or (xi==1 and yi==1):
                                A[0][n]=1.0
                                A[1][n]=dx
                                A[2][n]=dy
                                A[4][n]=dx*dx
                                A[5][n]=dy*dy
                                A[3][n]=dx*dy
                                lst.append([xi,yi])
                                n=n+1
        assert(n<=6)
        #print n
        return A,lst

def absmin(x):
        xabs=list(map(abs,x))
        xmin=min(xabs)
        return xabs.index(xmin)
         
def xyz2A(x,y,z):
        lst=[]
        A=np.zeros((10,10),'d')
        assert(x>=0 and x<1.0)
        assert(y>=0 and y<1.0)
        assert(z>=0 and z<1.0)
        n=0
        for xi in range(-1,3):
                dx=xi-x
                ax=abs(dx)
                if ax>=1.5:
                        continue
                for yi in range(-1,3):
                        dy=yi-y
                        ay=abs(dy)
                        if ay>=1.5:
                                continue
                        for zi in range(-1,3):
                                dz=zi-z
                                az=abs(dz)
                                if az>=1.5:
                                        continue
                                if (ax>0.5 and ay > 0.5 and az >0.5):
                                        continue
                                if (ax<1.0 and ay<1.0 and az<1.0) or (ax<=0.5 and ay<0.5 ) or (ax<=0.5 and az<0.5) or (ay<=0.5 and az<0.5):
                                        A[0][n]=1.0
                                        A[1][n]=dx
                                        A[2][n]=dy
                                        A[3][n]=dz
                                        A[4][n]=dx*dx
                                        A[5][n]=dy*dy
                                        A[6][n]=dz*dz
                                        A[7][n]=dx*dy
                                        A[8][n]=dx*dz
                                        A[9][n]=dy*dz
                                        lst.append([xi,yi,zi])
                                        n=n+1
        assert (n<=10)
        #print n
        return A,lst

def GetQ(crd):
        if len(crd)==2:
                x,y=crd
                A,lst=xy2A(x,y)
        elif len(crd)==3:
                x,y,z=crd
                A,lst=xyz2A(x,y,z)
        else:
                A=np.zeros((6,6),'d')
                A[0][0]=1.0
                lst=[[0,0]]
        #print lst
        n=A.shape[0]
        q=np.zeros((n,1),'d')
        q[0]=1.0
        x,res,rank,s=np.linalg.lstsq(A,q)
        #print np.dot(A,x)
        print(np.dot(A,x)[0][0],file=sys.stderr)
        return lst,x


def pr2(lst,x):
        ar=np.zeros((4,4),'d')
        for xi in range(-1,3):
                for yi in range(-1,3):
                        key=[xi,yi]
                        if key in lst:
                                q=x[lst.index(key)]
                                ar[xi+1][yi+1]=q
        #return ar
        print(ar)

def pr3(lst,x):
        ar=np.zeros((4,4,4),'d')
        for xi in range(-1,3):
                for yi in range(-1,3):
                        for zi in range(-1,3):
                                key=[xi,yi,zi]
                                if key in lst:
                                        q=x[lst.index(key)]
                                        ar[xi+1][yi+1][zi+1]=q
        print(ar)

def prs(lst,x):
        if len(lst[0])==2:
                pr2(lst,x)
        elif len(lst[0])==3:
                pr3(lst,x)

def ijk2I(ijk):
        i,j,k=ijk
        return (i+1)*4*4+(j+1)*4+(k+1)

def pri(lst,q,cutoff):
        cnt=0
        out=[]
        for qi in q:
                if abs(qi)>cutoff:
                        out.append("%d %.12f"%(ijk2I(lst[cnt]),qi[0]))
                        cnt=cnt+1
        for i in range (cnt,10):
                out.append("%d %s"%(-1,0.0))
        print(" ".join(out))

if __name__=="__main__":
        import sys
        #crd=map(float,sys.argv[1:])
        #lst,q=GetQ(crd)
        #prs(lst,q)
        #pri(lst,q,1e-12)
        #sys.exit()
        gzs=[0.1*(0.5+i) for i in range(0,10)]
        for a in gzs:
                for b in gzs:
                        for c in gzs:
                                crd=[a,b,c]
                                lst,q=GetQ(crd)
                                pri(lst,q,1e-100)
        
                                        
        

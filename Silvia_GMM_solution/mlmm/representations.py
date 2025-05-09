from .machine import *
#from qml.representations import *
#from qml.fchl import generate_representation,get_local_kernels,get_local_symmetric_kernels
from itertools import islice
#from mendeleev import element as elx
import pandas as pd
import collections


def count_lines(file):
    n = 0
    with open(file) as f:
        while True:
            buf = f.readline()
            if not buf:
                break
            n = n+1
    return n
    
class VelocityData(object):
    def __init__(self,fxyz,frames=None):
        self.xfile = fxyz
        self.frames = frames
        self.update()
    def update(self):
        self.nframe = count_lines(self.xfile)
        if self.frames is None:
            self.frames = [1,self.nframe]
        assert self.frames[1]<=self.nframe
    def getRep(self,*args,**kwargs):
        kw = {}
        kw.update(kwargs)
        self.rep = kw.get('rep','2norm')
        self.nuc = kw.get('nuc',None)
        print("Generating Representation")
        bar = pyprind.ProgBar(self.frames[1]-self.frames[0],bar_char='#')
        X = []
        y = []
        with open(self.xfile) as fx:
            for i in range(self.frames[1]):
                v = fx.readline().split()
                try:
                    v = [float(elem) for elem in v]
                    if len(v) != 6:
                        print ("ERROR: Expecting data with six columns only")
                        quit()
                    u = v[:3]
                    v = v[3:]
                    if self.rep == 'ColVel1norm':
                        X.append([np.linalg.norm(u,1)])
                        y.append(np.linalg.norm(v,1))
                    elif self.rep == 'vel2norm':
                        X.append([np.linalg.norm(u,2)**2, np.linalg.norm(v,2)**2])
                        y.append(0)
                    elif self.rep == 'vx2':
                        X.append([u[0]**2,v[0]**2])
                        y.append(0)
                    elif self.rep == 'vy2':
                        X.append([u[1]**2,v[1]**2])
                        y.append(0)  
                    elif self.rep == 'vz2':
                        X.append([u[2]**2,v[2]**2])
                        y.append(0)
                    elif self.rep == 'vx':
                        X.append([u[0],v[0]])
                        y.append(0)
                    elif self.rep == 'vy':
                        X.append([u[1],v[1]])
                        y.append(0)  
                    elif self.rep == 'vz':
                        X.append([u[2],v[2]])
                        y.append(0)
                    elif self.rep == 'vxvyvz':
                        X.append([u[0],u[1],u[2],v[0],v[1],v[2]])
                        y.append(0)
                    elif self.rep == 'ColVelu':
                        X.append([u[0]])
                        y.append(v[0])
                    elif self.rep == 'ColVelv':
                        X.append([u[1]])
                        y.append(v[1])
                    elif self.rep == 'ColVelw':
                        X.append([u[2]])
                        y.append(v[2])
                    else:
                        print("ERROR: Unknown representation type:",(self.rep))
                        quit()
                except:
                    print ("Not a float")
                bar.update()
        #print (len(X),len(y))
        self.X=np.array(X)
        self.y=np.array(y)
        #print (self.y)

class OmegaData(object):
    def __init__(self,fxyz,frames=None):
        self.xfile = fxyz
        self.frames = frames
        self.update()
    def update(self):
        self.nframe = count_lines(self.xfile)
        if self.frames is None:
            self.frames = [1,self.nframe]
        assert self.frames[1]<=self.nframe
    def getRep(self,*args,**kwargs):
        kw = {}
        kw.update(kwargs)
        self.rep = kw.get('rep','2norm')
        self.nuc = kw.get('nuc',None)
        print("Generating Representation")
        bar = pyprind.ProgBar(self.frames[1]-self.frames[0],bar_char='#')
        X2 = []
        y2 = []
        with open(self.xfile) as fx:
            for i in range(self.frames[1]):
                v = fx.readline().split()
                try:
                    v = [float(elem) for elem in v]
                    if len(v) != 4:
                        print ("ERROR: Expecting data with four columns only")
                    omegin = v[:2]
                    omegout = v[2:]
                    if self.rep == 'omega1omega2':
                        X2.append([omegin[0],omegin[1],omegout[0],omegout[1]])
                        y2.append(0)
                    else:
                        print("ERROR: Unknown representation type:",(self.rep))
                        quit()
                except:
                    print ("Not a float")
                bar.update()
        #print (len(X),len(y))
        self.X2=np.array(X2)
        self.y2=np.array(y2)
        #print (self.y)
#######################################################################################################################
class VelocityDataOmegaData(object):
    def __init__(self,fxyz,frames=None):
        self.xfile = fxyz
        self.frames = frames
        self.update()
    def update(self):
        self.nframe = count_lines(self.xfile)
        if self.frames is None:
            self.frames = [1,self.nframe]
        assert self.frames[1]<=self.nframe
    def getRep(self,*args,**kwargs):
        kw = {}
        kw.update(kwargs)
        self.rep = kw.get('rep','2norm')
        self.nuc = kw.get('nuc',None)
        print("Generating Representation")
        bar = pyprind.ProgBar(self.frames[1]-self.frames[0],bar_char='#')
        X = []
        y = []
        with open(self.xfile) as fx:
            for i in range(self.frames[1]):
                v = fx.readline().split()
                try:
                    v = [float(elem) for elem in v]
                    if len(v) != 6:
                        print ("ERROR: Expecting data with six columns only")
                        quit()
                    u = v[:3]
                    v = v[3:]
                    if self.rep == 'ColVel1norm':
                        X.append([np.linalg.norm(u,1)])
                        y.append(np.linalg.norm(v,1))
                    elif self.rep == 'vel2norm':
                        X.append([np.linalg.norm(u,2)**2, np.linalg.norm(v,2)**2])
                        y.append(0)
                    elif self.rep == 'vx2':
                        X.append([u[0]**2,v[0]**2])
                        y.append(0)
                    elif self.rep == 'vy2':
                        X.append([u[1]**2,v[1]**2])
                        y.append(0)  
                    elif self.rep == 'vz2':
                        X.append([u[2]**2,v[2]**2])
                        y.append(0)
                    elif self.rep == 'vx':
                        X.append([u[0],v[0]])
                        y.append(0)
                    elif self.rep == 'vy':
                        X.append([u[1],v[1]])
                        y.append(0)  
                    elif self.rep == 'vz':
                        X.append([u[2],v[2]])
                        y.append(0)
                    elif self.rep == 'vxvyvzomega1omega2':
                        X.append([u[0],u[1],u[2],v[0],v[1],v[2]])
                        y.append(0)
                    elif self.rep == 'ColVelu':
                        X.append([u[0]])
                        y.append(v[0])
                    elif self.rep == 'ColVelv':
                        X.append([u[1]])
                        y.append(v[1])
                    elif self.rep == 'ColVelw':
                        X.append([u[2]])
                        y.append(v[2])
                    else:
                        print("ERROR: Unknown representation type:",(self.rep))
                        quit()
                except:
                    print ("Not a float")
                bar.update()
        #print (len(X),len(y))
        self.X=np.array(X)
        self.y=np.array(y)
        #print (self.y)
       
        







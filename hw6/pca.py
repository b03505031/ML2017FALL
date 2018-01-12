import numpy as np
from skimage import io
import sys
import os
import matplotlib.pyplot as plt
imageFolder=sys.argv[1]
dataF='./imageSet.npy'
meanF='./mean2.npy'
uF='./U2.npy'
sF='./S2.npy'
vF='./V2.npy'
avgF='./avg2.jpg'
eigenfaceF='./eigenfaces2/'
eigenfaceF_prefix='e'
reconstructF='./reconstruct2/'
IMAGE_NUM=415
EIGENFACES=10
def readImage(MAX):
    data=[]
    for i in range(MAX):
        filename = imageFolder +'/'+ str(i) + ".jpg"
        print("   Reading "+filename, end="\r")
        im = io.imread(filename)
        data.append(im.tolist())
    data=np.array(data)
    print("")
    print(str(MAX)+"   Images loaded and saved as "+dataF)
    np.save(dataF,data)
    return data

def getData():
    data=[]
    for i in range(IMAGE_NUM):
        filename=imageFolder+'/'+str(i) + ".jpg"
        #imageFolder +'/'+ str(i) + ".jpg"
        print("   Reading "+filename, end="\r")
        im = io.imread(filename)
        data.append(im.tolist())
    data=np.array(data)
    return data

def getMean(data):
    if '-recompute' in sys.argv or not(os.path.exists(meanF)):
        print("   Calculating mean")
        data_m=data.mean(axis=0,keepdims=True)
        # shape=(1,600,600,3)
        data_m=data_m.reshape(600,600,3)
        # shape=(600,600,3)
        np.save(meanF,data_m)
        print("   Mean caculated and saved as "+meanF)
        
    else:
        print("   Load mean form "+meanF)
        data_m=np.load(meanF)
    return data_m

def getSVD(data,data_m):
    if '-recompute' in sys.argv or not(os.path.exists(uF)) or not(os.path.exists(sF)) or not(os.path.exists(vF)):
        print("   Calculating eigenvectors and eigenvalues")
        U, S, V = np.linalg.svd(data- data_m,full_matrices=False)
            #Ut(single layer U) shape: (415,415)
            #St(single layer S) shape: (415,415)
            #Vt(single layer V) shape: (415,360000)
        print("   USV of RGB computed.")
        print("     U shape: "+str(U.shape))
        print("     S shape: "+str(S.shape))
        print("     V shape: "+str(V.shape))
        np.save(uF,U)
        np.save(sF,S)
        np.save(vF,V)
        print("   USV saved")
    else:
        U=np.load(uF)
        S=np.load(sF)
        V=np.load(vF)
        print("   USV of RGB loaded.")
        print("     U shape: "+str(U.shape))
        print("     S shape: "+str(S.shape))
        print("     V shape: "+str(V.shape))
    return U,S,V

#1 avg faces
if '--p1' in sys.argv:
    print("\n==P1======================\n")
    data=getData().reshape(415,360000*3)
    data_m=getMean(data).astype(int)
    print("   Generating average image")
    io.imsave(avgF,data_m)
    print("   Mean image saved as "+avgF)

#2 1st~4th eigenfaces
if '--p2' in sys.argv:
    print("\n==P2======================\n")
    data=getData().reshape(415,360000*3)
    data_m=getMean(data).reshape(1,360000*3)
    U,S,V=getSVD(data,data_m)
    data=None
    data_m=None
    U=None
    S=None
    #U shape: (415, 415)
    #S shape: (415, 415)
    #V shape: (415, 360000*3)
    eigv=V[:EIGENFACES,:]
    V=None
    print("   "+str(EIGENFACES)+ " eigenfaces generated")
    for i in range(EIGENFACES):
        print("   Eigenfaces saved: "+eigenfaceF+eigenfaceF_prefix+str(i)+'.jpg', end="\r")
        ev=eigv[i]
        #eigv[i]*=-1
        ev-=np.min(ev)
        ev /= np.max(ev)
        ev = (ev * 255).astype(np.uint8)
        #eigv[i]=255-eigv[i]
        if not os.path.exists(eigenfaceF):
            os.makedirs(eigenfaceF)
        io.imsave((eigenfaceF+eigenfaceF_prefix+str(i)+'.jpg'),ev.reshape(600,600,3))
    print("")
    print("   "+str(EIGENFACES)+ " eigenfaces saved")
    
   
    
#3 randomly pick 4 face[23, 160, 200, 371], and reconstruct by 1st~4th eigenvector
if '--p3' in sys.argv:
    print("\n==P3======================\n")
    faces=[23,160,200,371]
    data=getData().reshape(415,360000*3)
    todo=[data[23],data[160],data[200],data[371],data[50]]
    data_m=getMean(data).reshape(1,360000*3)
    U,S,V=getSVD(data,data_m)
    #data : 415,360000*3
    #data_m : 360000*3
    #V shape: (415, 360000*3)
    U=None
    S=None
    eigen=V[:4,:]
    for face in range(5):
        print("   Reconstructing image "+str(faces[face]))
        img=data[faces[face]]
        project = np.dot(img-data_m , eigen.T)
        reconstruct = np.dot(project, eigen) + data_m
        reconstruct-=np.min(reconstruct)
        reconstruct/=np.max(reconstruct)
        reconstruct=(reconstruct*255).astype(np.uint8)
        if not os.path.exists(reconstructF):
            os.makedirs(reconstructF)
        io.imsave((reconstructF+str(faces[face])+'.jpg'),reconstruct.reshape(600,600,3).astype(np.uint8))
        
    #project = np.dot(eigenvectors[:5, :], img - imgs_mean)
	#reconstruct = np.dot(eigenvectors[:5, :].T, project) + imgs_mean

#4 1st~4th eigenfaces explained variance ratio
if '--p4' in sys.argv:
    print("\n==P4======================\n")
    if not(os.path.exists(sF)) or '-recompute' in sys.argv:
        data=getData().reshape(415,360000*3)
        data_m=getMean(data).reshape(360000*3)
        U,S,V=getSVD(data,data_m)
        data=None
        data_m=None
        U=None
        V=None
    else:
        print("   Load S from "+sF)
        S=np.load(sF)
    #S shape = (415)
    evr=[(i / np.sum(S,axis=0)) for i in (S[:,])]
    print("   EVR for 1~4 eigenfaces under Merge mode: "+ str(evr[:4]))

if '--test' in sys.argv:
    
    
    S=np.load(sF)
    s=np.zeros((3,415))
    for i in range(415):
        for rgb in range(3):
            s[rgb,i]=S[rgb,i,i]
    print(s[2,0])
    
    '''
    U,S,V=Ut, st, Vt = np.linalg.svd(data[:,:,:]- data_m[:,:],full_matrices=False)
    print("     U shape: "+str(U.shape))
    print("     S shape: "+str(S.shape))
    print("     V shape: "+str(V.shape))
    U,S,V=getSVD(data,data_m)
    print("     U shape: "+str(U.shape))
    print("     S shape: "+str(S.shape))
    print("     V shape: "+str(V.shape))
    '''
if '--hw' in sys.argv:
    print("\n==Reconstruct======================\n")
    
    filename=os.path.join(sys.argv[1],sys.argv[2])
    data=getData().reshape(415,360000*3)
    data_m=getMean(data).reshape(360000*3)
    U,S,V=getSVD(data,data_m)
    #data : 415,360000*3
    #data_m : 360000*3
    #V shape: (415, 360000*3)
    U=None
    S=None
    eigen=V[:4,:]
    print("   Reconstructing image ")
    img=io.imread(filename).reshape(1,360000*3)
    project = np.dot(img-data_m , eigen.T)
    reconstruct = np.dot(project, eigen) + data_m
    reconstruct-=np.min(reconstruct)
    reconstruct/=np.max(reconstruct)
    reconstruct=(reconstruct*255).astype(np.uint8)
    io.imsave(('reconstruction.jpg'),reconstruct.reshape(600,600,3).astype(np.uint8))

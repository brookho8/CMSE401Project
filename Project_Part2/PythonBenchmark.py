import open3d as o3d
import pickle
import h5py
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from numpy import load, stack
from matplotlib.pyplot import subplots
from scipy import spatial
import time
from skimage import measure


def arrayToPointCloudCupy(d):
	pointListWhere = cp.where(d==1)
	pointListWhere = cp.array(pointListWhere)
	pointListWhere = pointListWhere.transpose()
	cloud = o3d.geometry.PointCloud()
	cloud.points = o3d.utility.Vector3dVector(cp.asnumpy(pointListWhere))
	return cloud

def arrayToPointCloud(d):
	pointListWhere = np.where(d==1)
	pointListWhere = np.array(pointListWhere)
	pointListWhere = pointListWhere.transpose()
	cloud = o3d.geometry.PointCloud()
	cloud.points = o3d.utility.Vector3dVector(pointListWhere)
	return cloud

def listRenum(d,x,y,z):
	for i in range(0,x):
		for j in range(0,y):
			for k in range(0,z):
				if d[i][j][k] >= 10:
					d[i][j][k] = 1
				else:
					d[i][j][k] = 0
	return d

def arrayToMesh(array):
	verts, faces, normals, values = measure.marching_cubes(array, 0)
	verts = o3d.utility.Vector3dVector(verts)
	faces = o3d.utility.Vector3iVector(faces)
	mesh = o3d.geometry.TriangleMesh(verts, faces)
	mesh.compute_vertex_normals()
	mesh = mesh.simplify_vertex_clustering(3)
	mesh = mesh.filter_smooth_taubin(number_of_iterations=100)#filter_smooth_simple(number_of_iterations=10)
	mesh.compute_vertex_normals()
	return mesh





topNum = 30
numTrials = 5

h = h5py.File('pred.h5','r')
d_orig = np.array(h['vol0'][0]).astype(np.int8)
h.close()

dic = {}
for i in range(1,topNum):
	for trial in range(1,numTrials):
		try:
			d = cp.tile(d_orig,i)
			if trial == 1:
				dic[i] = {"size":d.size,'timeRenum':[],'timeCloud':[]}
			t1 = time.time()
			d[d >= 10] = 1
			d[d != 1] = 0
			t2 = time.time()
			cloud = arrayToPointCloudCupy(d)
			t3 = time.time()
			print('=============')
			print(i)
			print(d.shape)
			print(d.size)
			print(t2 - t1)
			print(t3 - t2)
			print('=============')
			print()
			dic[i]['timeRenum'].append(t2-t1)
			dic[i]['timeCloud'].append(t3-t2)
			with open('timeOutputCupy.pkl','wb') as outFile:
				pickle.dump(dic, outFile)
		except Exception as e:
			print('Failed Cupy',i,trial)


dic = {}
for i in range(1,topNum):
	dic[i] = {"size":d.size,'timeRenum':[],'timeCloud':[]}
	for trial in range(1,numTrials):
		d = np.tile(d_orig,i)
		t1 = time.time()
		d[d >= 10] = 1
		d[d != 1] = 0
		t2 = time.time()
		cloud = arrayToPointCloud(d)
		t3 = time.time()
		print('=============')
		print(i)
		print(d.shape)
		print(d.size)
		print(t2 - t1)
		print(t3 - t2)
		print('=============')
		print()
		dic[i]['timeRenum'].append(t2-t1)
		dic[i]['timeCloud'].append(t3-t2)
		with open('timeOutputNumpy.pkl','wb') as outFile:
			pickle.dump(dic, outFile)

dic = {}
for i in range(1,topNum):
	dic[i] = {'timeRenum':[]}
	for trial in range(1,numTrials):
		d = np.tile(d_orig,i)
		x,y,z = d.shape
		d = d.tolist()
		t1 = time.time()
		d = listRenum(d,x,y,z)
		t2 = time.time()
		print('=============')
		print(i)
		# print(d.shape)
		# print(d.size)
		print(t2 - t1)
		#print(t3 - t2)
		print('=============')
		print()
		dic[i]['timeRenum'].append(t2-t1)
		with open('timeOutputList.pkl','wb') as outFile:
			pickle.dump(dic, outFile)


with open('timeOutputNumpy.pkl','rb') as inFile:
	n = pickle.load(inFile)
with open('timeOutputCupy.pkl','rb') as inFile:
	c = pickle.load(inFile)
with open('timeOutputList.pkl','rb') as inFile:
	l = pickle.load(inFile)

x = []
n1 = []
n2 = []
c1 = []
c2 = []
l1 = []

#I hardcoded in my results from my c program as that seemed like an easy way to graph it
#

x2 = [100000001, 349999994 ,600000023 ,850000023 ,1100000023 ,1350000023 ,1600000023 ,1850000023 ,2099999904 ,2349999904 ,2599999904 ,2849999904 ,3099999904 ,3349999904 ,3599999904 ,3849999904 ,4099999904]
myc = [0.020759, 0.070140, 0.119506, 0.168891, 0.218357, 0.267788, 0.323408, 0.373418, 0.425681, 0.518683, 0.625674, 0.735264, 0.843006, 0.957786, 1.054841, 1.161067, 1.264862]
for i in range(1,10):
	x.append(c[i]['size'])
	n1.append(np.mean(n[i]['timeRenum']))
	n2.append(np.mean(n[i]['timeCloud']))
	c1.append(np.mean(c[i]['timeRenum']))
	c2.append(np.mean(c[i]['timeCloud']))
	l1.append(np.mean(l[i]['timeRenum']))
plt.plot(x,n1,label='Numpy')
#plt.plot(x,n2,label='Numpy')
plt.plot(x,c1,label='Cupy')
#plt.plot(x,c2,label='Cupy')
plt.plot(x,l1,label='Lists')
plt.plot(x2, myc,label='My Cuda')
plt.legend()
plt.xlabel('Elements In Array')
plt.ylabel('Average Time')
plt.title("Renumber Task")
plt.show()
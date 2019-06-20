from scipy.sparse import coo_matrix
import numpy as np
import matplotlib as plt
import matplotlib.image as mpimg
from scipy.fftpack import dct, idct
from scipy.sparse import find
from sklearn.linear_model import Lasso, MultiTaskLasso
from matplotlib.pyplot import plot, show, title,imshow,figure
from sklearn.metrics import mean_squared_error 
from numpy import ndarray
from PIL import Image
import pywt
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from skimage.measure import compare_psnr as PSNR
from skimage.measure import compare_ssim as SSIM
from skimage.measure import compare_nrmse as NRMSE
#have to use wavelet transform?
#converting into a grey-scale image 
#use of lenna,escher/waterfall 
#def rgb2gray(img):
#	pictures that are absolutely sparse
#20 percent sparsity ?? high sparsity 

#1.sparse_approx
def Sparse_rep(matrix,m,n,limit):
	#tmp=np.fabs(matrix)
	tmp=matrix
	tmp=np.sort(tmp,axis=None,kind='mergesort')# smallest to biggest
	lim=tmp[limit-1]#lim is 10000-500   
	for i in range(m):
		for j in range(n):
			if np.fabs(matrix[i][j])<lim:
				matrix[i][j]=0
	return matrix	

'''def Sparse_rep_wavelet(matrix,m,n,limit):
	cA,(cH,cV,cD)=matrix
	print cA.shape
	print CountSparse(cA)
	print cA
	print cH.shape
	print CountSparse(cH)
	print cV.shape
	print CountSparse(cV)
	print cD.shape
	print CountSparse(cD)
	m,n=cA.shape
	lim=0.1*m*n

	cA=np.asarray(cA).copy()
	cA=Sparse_rep(cA,m,n,lim)
	cH=Sparse_rep(cH,m,n,lim)
	cV=Sparse_rep(cV,m,n,lim)
	cD=Sparse_rep(cD,m,n,lim)
	out=CA,(cH,cV,cD)
	print type(out)
	return out
'''
def CountSparse(mat):
	return np.count_nonzero(mat)

#2.image_manip
def Convert_Image_Gray(image):
	img_gray=image.convert('L')
	#print(image.format,image.size,image.mode) 
	print(img_gray.format,img_gray.size,img_gray.mode) 
	img_gray=img_gray.resize( (200,200),1 )
	print(img_gray.format,img_gray.size,img_gray.mode) 
	#img=np.array(img_gray,dtype =np.float)
	return img_gray 

def Convert_img_BW(img):
	bw=np.asarray(img).copy()
	print bw.shape
	bw[bw<128]=0
	bw[bw>=128]=255
	return bw

#3.transform
def Wavelet_1D(image):
	out=pywt.dwt(image,'db10')
	return out

def iWavelet_1(image):
	out=pywt.idwt(iamge,'db10')

def wavelet_2D(image):
	W=pywt.dwt2(image,'db10')
	#out=np.asarray(W,dtype=np.float64)
	#return out
	return W

def iwavelet_2D(image):
	return pywt.idwt2(image,'db10')

def dct2D(image):
	return dct(dct(image.T,norm='ortho').T,norm='ortho') #.T= transpose

def get_2d_idct(coefficients):
    return idct(idct(coefficients.T, norm='ortho').T, norm='ortho')	

def fft2D(image):
	img=np.fft.fft2(image,norm='ortho')
	img=np.array(img,dtype=np.float)
	return img	 

#4.Measurements
def Take_measurement_2D(matrix,m,n,index1,index2):
	Y=matrix[index1][index2]
	print Y.shape
	return Y

def Take_measurement(matrix,m,n,index):
	X=matrix.reshape(m*n,1)
	Y=X[index]
	#Y=np.expand_dims(X,axis=1)
	print "shape of my heart"
	print Y.shape
	#Y=X
	return Y
 
 #5. CSmat
def create_Amat_kron(m,n,index):
	A1=dct(np.identity(m),norm='ortho',axis=0)
	A2=dct(np.identity(n),norm='ortho',axis=0)
	A=np.kron(A1,A2)
	A=A[index]
	return A

def create_Amat_rand(k,m,n,index):
	A1=np.zeros((k,m*n))
	A2=np.zeros((m,n)).reshape(1,m*n) 
	for i,j in enumerate(index):
		A2[0,j]=1
		A1[i,:]=dct(A2)
		A2[0,j]=0
	print A1.shape
	return A1	

def create_Amat_rand_Wave(k,m,n,index):
	A1=np.zeros((k,m*n))
	A2=np.zeros((m,n)).reshape(1,m*n)

	for i,j in enumerate(index):
		A2[0,j]=1
		(t1,t2)=Wavelet_1D(A2)
		tmp=np.concatenate((t1,t2),axis=1)
		A1[i,:]=tmp
		A2[0,j]=0
	print A1.shape
	return A1	

def create_Amat_1D(m,n,index):
	A1=dct(np.eye(m*n),norm='ortho')
	A=A1[index]
	print A.shape
	return A
def create_Amat_1DW(m,n,mea):
	A1,A2=Wavelet_1D(np.eye(m*n))
	A=np.concatenate((A1,A2),axis=1)
	A=A[:mea,:]
	print A.shape
	return A

def create_Amat_JL(m,n,index):
	mea=m*n*0.3
	#A1=dct(np.eye(m*n),norm='ortho')
	#A=A1[index]
	#print np.shape(A)
	#Trans=GRP(n_components=10000)
	#A_new=Trans.fit_transform(A)
	#print np.shape(A_new)
	A_new=np.random.normal(0,(1/np.sqrt(mea)),(mea,10000))
	print A_new
	return A_new

#img=Image.open("C:/Users/a0116439/Downloads/S&G3.jpg")
img=Image.open("C:\Users\Linhong\Downloads\S&G3.jpg")
#img=Image.open("C:\Python27\Scripts\img\S&G.jpg")
#img=Image.open("C:\Python27\Scripts\img\lenaTest1.jpg")
print(img.format,img.size,img.mode) 
image_=Convert_Image_Gray(img)
#image_.show()
image_=Convert_img_BW(image_)
#RESULT=Image.fromarray(image_)
#RESULT.show()

#init
#np.reshape(image_,(100,100) )
Nx,Ny=image_.shape
N_samp=Nx * Ny 
M=int(Nx*Ny*0.3)
print M  
#print CountSparse(image_)

#'''
#2D dct
print "## dct approx ##"
trans=dct2D(image_)  
print CountSparse(trans)
a,b=image_.shape
lim=a*b*92/100
trans=Sparse_rep(trans,a,b,lim)
print "threshold nnz ="
print CountSparse(trans)
Simage_=get_2d_idct(trans)
print image_.dtype
print Simage_.dtype
AOimage_=image_.astype("uint32")
Aimage_=Simage_.astype("uint32")
print Aimage_.dtype
print AOimage_.dtype
print "PSNR val is"
print PSNR(AOimage_,Aimage_)
RES=Image.fromarray(Aimage_)
#RES.convert('RGB')
RES.show()
#'''
'''
#2D wavelet
print "## d wavelet trans approx ##"
trans=wavelet_2D(image_)
cA,(cH,cV,cD)=trans
a,b=cA.shape
#print cA.shape
lim=a*b*90/100
print CountSparse(cA)
cA=Sparse_rep(cA,a,b,lim)
cH=Sparse_rep(cH,a,b,lim)
cV=Sparse_rep(cV,a,b,lim)
cD=Sparse_rep(cD,a,b,lim)
trans=cA,(cH,cV,cD)
print 'wavelet done'
print type(trans)
Simage_=iwavelet_2D(trans)
RES=Image.fromarray(Simage_)
RES.show()
'''


print "# creating matrix #"  
index= np.random.randint(0,N_samp,(M,))
index=np.sort(index)

#index1=np.random.randint(0,100,(M,))
#index2=np.random.randint(0,100,(M,))
#Y=Take_measurement_2D(Simage_,Nx,Ny,index1,index2)

Y=Take_measurement(Simage_,Nx,Ny,index)

#A=create_Amat_kron(Nx,Ny,index)
#A=create_Amat_1D(Nx,Ny,index)
A=create_Amat_rand(M,Nx,Ny,index)
#A=create_Amat_rand_Wave(M,Nx,Ny,index)
#A=create_Amat_JL(Nx,Ny,index)
#A=create_Amat_1DW(Nx,Ny,M)

print "# reconstruction #"  
val=.001
count =1 # 0.0001,0.001,0.01,0.1
while(count!=0):
	#7.OMP
	#'''
	non_Zcoeff=CountSparse(Simage_)
	omp=OMP()
	omp.fit(A,Y)
	print "omp done"
	recon=idct(omp.coef_).reshape((Nx,Ny))
	print type(recon)
	print omp.coef_
	#'''


#6.Lasso
	'''
	lasso = Lasso(alpha=val)
	lasso.fit(A,Y)
	print (lasso.coef_).shape
	val=val*10
	recons_sparse = coo_matrix(lasso.coef_)
	sparsity = 1 - float(recons_sparse.getnnz())/len(lasso.coef_)
	print "solution is %{0} sparse".format(100.*sparsity)
	recon=idct(lasso.coef_).reshape((Nx,Ny))
	#recon=(lasso.coef_).reshape((Nx,Ny))
	#cA=lasso.coef_[0:11250]
	#cD=lasso.coef_[11250:]
	#print cA.shape
	#print cD.shape
	#recon=pywt.idwt(cA,cD,'haar','symmetric').reshape(Nx,Ny)
	#recon2=pywt.idwt(cD,cA,'haar').reshape(Nx,Ny)
	#imshow(recon)
	#show()
	'''

#8. Results
	recon=recon.astype("uint32")
	print "PSNR val is"
	Simage_=Simage_.astype("uint32")
	print PSNR(Simage_,recon)
	print "SSIM val is"
	print SSIM(Simage_,recon)
	print "NRMSE is "
	print NRMSE(Simage_,recon)
	RESULT=Image.fromarray(recon)
	#RESULT.convert('RGB')
	RESULT.show()
	#RESULT.save("recover.jpg")
	print "#absolute error#"
	err=Simage_-recon
	RESER=Image.fromarray(err)
	RESER.show()
	

	count=count-1
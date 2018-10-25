import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = sio.loadmat('face.mat')
X = data['X']
l = data['l']

X_train, X_test, y_train, y_test = train_test_split(X.T, l.T, test_size=0.25, random_state=19)
#X_train = X_train.T
#X_test = X_test.T
#y_train = y_train.T
#y_test = y_test.T


def computeAvgFace(X_train):
    return np.mean(X_train, axis=0)

def computeMatA(X_train, avgFace):
    A = np.empty((390,2576))
    for index, face in enumerate(X_train):
        A[index] = face - avgFace
    return A.T

def computeMatS(A):
    N = min(A.shape) #total number of images in training set
    return np.matmul(A, A.T)/N
    
def computeEigenVectsVals(S):
    w, v = np.linalg.eigh(S)
    w = np.flip(w, axis=0) #turn ascending into descending
    v = np.flip(v, axis=1) #turn ascending into descending
    return w, v

def pickTopEigenvectors(v, number):
    return v[0:number]

def plot_face(face):
    face2 = np.resize(face, (46, 56))
    plt.figure()
    plt.imshow(face2.T, cmap='gray')
            
avgFace = computeAvgFace(X_train)
A = computeMatA(X_train, avgFace)

S = computeMatS(A)
S_low = computeMatS(A.T)
w, v = computeEigenVectsVals(S)
w_low, v_low = computeEigenVectsVals(S_low)

u = np.matmul(A, v_low)/min(A.shape)
from sklearn.preprocessing import normalize
u = normalize(u, axis=0)

topEigenvectors = pickTopEigenvectors(u.T, 389)

w_0=[]
sum_terms=[]
for u_i in topEigenvectors:
    w_0.append(np.dot(A.T[5], u_i))
    sum_terms.append(w_0[-1]*u_i)
phi_reconstructed = sum(sum_terms)
face_reconstructed = phi_reconstructed + avgFace
plot_face(face_reconstructed)
plot_face(X_train[5])
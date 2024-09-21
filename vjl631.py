import numpy as np
from typing import Callable

"""
HUSK DETTE INDEN I AFLEVERER:
	1. 	Fjern eller udkommenter print statements.
	2. 	Fjern eller udkommenter funktionskald uden for de definerede funktioner.
	3. 	Tjek navnene er de samme som der blev givet i Skabelon.py.
	4. 	Tjek jeres funktioner returnerer det beregnede resultat og at jeres
		funktioner ikke har 'return None' i sig.
	5.	Tjek jeres funktioner returnerer den ønskede type.

VIGTIG AT VIDE TIL AFLEVERING:
	1.	Du må ikke ændre funktions navnene
	2.	Du må gerne ændre argument navnene men ikke den orden de har
		2.1	Du må ikke addere flere argumenter ind i funktionerne
	3.	Du må gerne lave flere funktioner
		3.1	Hvis du laver flere funktioner så skal de laves i de allerede
			lavede funktioner. 
	8.	Du behøver ikke at bruge alle argumenter men det er klogt hvis du gør
	9.	Du må kun bruge de pakker som er givet i skabelonfilen.
		9.1	i.e numpy

NOTATION I FUNKTIONERNE:
	Enhver funktion vi har givet har det som man kalder 'Type Annotation' i sig.
	Dette er både for funktionens argumenter, men også for dens returnerings 
	værdi. 

	Dette betyder groft at funktionerne kun kan tage visse typer ind og kun kan
	spytte en type ud. Lad os kigge på Lambda funktionen:

		def Lambda(x: int | float) -> np.ndarray:
		
	Denne notation betyder at funktionen Lambda tager et x som argument hvor det
	x kun kan være en integer eller en float. Samtidig retunerer Lambda en
	array af typen np.ndarray (i.e en normal numpy array) 

	Bemærk også at der er en type kaldet 'Callable'. Dette er bare en funktions
	type. Så her skal du angive en funktion hvis du ønsker at bruge funktionen.
	Bemærk at funktionsargument navnene er af samme navn som andre funktioner.
	Dette er højst sandsynligt ikke en fejl ;).
"""

#Opgave A
def Lambda(x: int | float) -> np.ndarray:
	#Implement code
	A=np.zeros((3,3))

	if x <= 65:
		A[0,2]=0.0005+10**(5.88+0.038*x-10)
		A[0,1]=0.0004+10**(4.54+0.06*x-10)
		A[0,0]=-(A[0,2]+A[0,1])
		A[1,0]=2.0058*np.exp(-0.117*x)
		A[1,2]=2*(0.0005+10**(5.88+0.038*x-10))
		A[1,1]=-(A[1,0]+A[1,2])
		A[2,0]=0
		A[2,1]=0
		A[2,2]=0
	else:
		A[0,2]=0.0005+10**(5.88+0.038*x-10)
		A[0,1]=0.0004+10**(4.54+0.06*x-10)
		A[0,0]=-(A[0,2]+A[0,1])
		A[1,0]=2.0058*np.exp(-0.117*x)
		A[1,2]=0.0005+10**(5.88+0.038*x-10)
		A[1,1]=-(A[1,0]+A[1,2])
		A[2,0]=0
		A[2,1]=0
		A[2,2]=0
	#Type<return> skal være np.ndarray
	return A

#print(Lambda(50))




#Opgave B
def Prodint(Lambda: Callable, s: int | float, t: int | float, n: int) -> np.ndarray:
	#Implement code
	h = (t - s) / n
	y0 = np.identity(Lambda(s).shape[0])
	x0=s
	for i in range(n):
		k1 = h * np.matmul(y0,Lambda(x0))
		k2 = h * np.matmul(y0+(1/2)*k1,Lambda(x0 + 0.5 * h))
		k3 = h * np.matmul(y0+(1/2)*k2,Lambda(x0 + 0.5 * h))
		k4 = h * np.matmul(y0+k3,Lambda(x0 + h))
		y0 = y0 + (1/6)*(k1+2*k2+2*k3+k4)
		x0+=h
	#Type<return> skal være np.ndarray
	return y0


#print((16/15)*Prodint(Lambda,40,70,2000)-(1/15)*Prodint(Lambda,40,70,1000))




#Opgave E
def R(x: int | float, mu: float) -> np.ndarray:
	#Implement code
	A=np.identity(3)
	A[2,2]=0
	if x <=65:
		A[0,0]=-mu
	#Type<return> skal være np.ndarray
	return A


  #Opgave E
def M(s: int | float, t: int | float, Lambda: Callable, R: Callable, Prodint: Callable, mu: float, rint: float, N: int) -> np.ndarray:
	#Implement code
	def COX(x):
		N=Lambda(x).shape[0]
		TL=Lambda(x)-rint*np.identity(N)
		TR=R(x,mu)
		BL=np.zeros((3,3))
		BR=Lambda(x)
		svar=np.block([[TL,TR],[BL,BR]])
		return svar

	A=Prodint(COX,s,t,N)
	RES= A[:3,3:]
	#Type<return> skal være np.ndarray
	return RES

#print(M(40,70,Lambda,R,Prodint,0.5,0.01,5000))



#Opgave F
def Reserve(t: int | float, TT: int | float, Lambda: Callable, R: Callable, Prodint: Callable, mu: float, rint: float, n: int) -> np.ndarray:
	#Implement code
	e1=np.array([1,0,0])
	e2=np.array([[1],[1],[1]])
	L=np.matmul(M(t,TT,Lambda,R,Prodint,mu,rint,n),e2)
	res=np.matmul(e1,L)
	#Type<return> skal være np.ndarray
	return res

#print(Reserve(40,100,Lambda,R,Prodint,0.5,0.01,5000))



#Opgave G
def Equiv_Premium(a: int | float, b: int | float, age: int, agelimit: int, Lambda: Callable, R: Callable, Reserve: Callable, rint: float, N: int) -> float:
	#Implement code
	f1=Reserve(age,agelimit,Lambda,R,Prodint,a,rint,5000)
	f2=Reserve(age,agelimit,Lambda,R,Prodint,b,rint,5000)
	for i in range(N):
		mu=0.5*(b+a)
		f3=Reserve(age,agelimit,Lambda,R,Prodint,mu,rint,5000)
		if f3==0:
			return mu
		if f2*f3<0:
			a=mu
			f1=f3
		else:
			b=mu
			f2=f3
	return mu
#print(Equiv_Premium(0,1,40,100,Lambda,R,Reserve,0.01,50),np.round(Equiv_Premium(0,1,40,100,Lambda,R,Reserve,0.01,50)*100000))
	


#Opgave J
def Unif_Matexp(A: np.ndarray, x: int | float, eps: float) -> np.ndarray:
	#Implement code
	A=A*x
	x=1
	M=np.eye(A.shape[0])+x*A
	termin=x*A
	n=2
	while np.linalg.norm(termin,ord='fro')>eps:
		termin=np.dot(termin,A)/n
		M+=termin
		n+=1
	#Type<return> skal være np.ndarray
	return M
	


#Opgave K
def Aprodint(Lambda: Callable, Unif_Matexp: Callable, s: int | float, t: int | float, eps: float) -> np.ndarray:
	if s==t:
		resultat=np.identity(len(Lambda(s))) #hvis s=t er det klart identitetsmatrixen
	else: 
		k=np.ceil(t-s).astype('int')-1
		if k==0:
			resultat=Unif_Matexp(Lambda,((s+k+t)/2),t-(s+k),eps)
		else:
			produkt=Unif_Matexp(Lambda(s+0.5),1,eps)
			for i in range(1,k):
				st=Unif_Matexp(Lambda(s+(0.5+i)),1,eps)
				produkt=np.matmul(produkt,st)
			resultat=np.matmul(produkt,Unif_Matexp(Lambda((s+k+t)/2),t-(s+k),eps))
	#Type<return> skal være np.ndarray
	return resultat
#print(Aprodint(Lambda,Unif_Matexp,0,11,1e-6))




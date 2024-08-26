:Comment : Linearized by Torbjorn Ness 2013
:Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON	{
	SUFFIX Ih_linearized_v2_frozen
	NONSPECIFIC_CURRENT ihcn
	RANGE gIhbar, V_R, ihcn, wInf, ehcn, phi, wTau
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gIhbar = 0.00001	(S/cm2) 
	ehcn =  -45.0 		(mV)
	V_R		(mV)
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)
	wInf
	wTau
	wAlpha
	a1
	a2
	a3
	b1
	b2
	b3
	foo
	dwinf
	wBeta
    phi
}

BREAKPOINT	{
	:SOLVE states METHOD cnexp
	ihcn  = gIhbar*wInf*(v-ehcn) - wInf*phi
}


INITIAL{
	a1	= 0.001*6.43
	a2 	= 154.9
	a3 	= 11.9
	b1 	= 0.001*193
	b2 	= 33.1
	wAlpha 	= a1*(V_R+a2)/(exp((V_R+a2)/a3)-1)
	wBeta  	=  b1*exp(V_R/b2)
	wInf 	= wAlpha/(wAlpha + wBeta)
	foo 	= wAlpha/(a3*a1) * exp((V_R + a2)/a3) + (V_R + a2)/b2 - 1
    wTau 	= 1/(wAlpha + wBeta)
    dwinf 	= - wBeta*wAlpha *wTau * wTau /(V_R + a2) * foo
    phi = gIhbar * dwinf *(ehcn - V_R)
}

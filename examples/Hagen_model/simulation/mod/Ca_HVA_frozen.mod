:Comment :
:Reference : :		Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993

NEURON	{
	SUFFIX Ca_HVA_frozen
	USEION ca READ eca WRITE ica
	RANGE gCa_HVAbar, gCa_HVA, ica, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gCa_HVAbar = 0.00001 (S/cm2)
	V_R (mV)
}

ASSIGNED	{
    m
    h
	v	(mV)
	eca	(mV)
	ica	(mA/cm2)
	gCa_HVA	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
}

BREAKPOINT	{
	gCa_HVA = gCa_HVAbar*m*m*h
	ica = gCa_HVA*(v-eca)
}


INITIAL{
	UNITSOFF
		mAlpha =  (0.055*(-27-V_R))/(exp((-27-V_R)/3.8) - 1)
		mBeta  =  (0.94*exp((-75-V_R)/17))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
		hAlpha =  (0.000457*exp((-13-V_R)/50))
		hBeta  =  (0.0065/(exp((-V_R-15)/28)+1))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = 1/(hAlpha + hBeta)
	UNITSON

	m = mInf
	h = hInf
}

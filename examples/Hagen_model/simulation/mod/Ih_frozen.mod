:Comment :
:Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON	{
	SUFFIX Ih_frozen
	NONSPECIFIC_CURRENT ihcn
	RANGE gIhbar, gIh, ihcn
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gIhbar = 0.00001 (S/cm2)
	ehcn =  -45.0 (mV)
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)
	gIh	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
    m
}

BREAKPOINT	{
	gIh = gIhbar*m
	ihcn = gIh*(v-ehcn)
}

INITIAL{
	UNITSOFF
        if(v == -154.9){
            v = v + 0.0001
        }
		mAlpha =  0.001*6.43*(v+154.9)/(exp((v+154.9)/11.9)-1)
		mBeta  =  0.001*193*exp(v/33.1)
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
	m = mInf
}

:Comment :
:Reference : :		Characterization of a Shaw-related potassium channel family in rat brain, The EMBO Journal, vol.11, no.7,2473-2486 (1992)

NEURON	{
	SUFFIX SKv3_1_frozen
	USEION k READ ek WRITE ik
	RANGE gSKv3_1bar, gSKv3_1, ik, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gSKv3_1bar = 0.00001 (S/cm2)
	V_R (mV)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gSKv3_1	(S/cm2)
	mInf
	mTau
    m
}

BREAKPOINT	{
	gSKv3_1 = gSKv3_1bar*m
	ik = gSKv3_1*(v-ek)
}

INITIAL{
	UNITSOFF
		mInf =  1/(1+exp(((V_R -(18.700))/(-9.700))))
		mTau =  0.2*20.000/(1+exp(((V_R -(-46.560))/(-44.140))))
	UNITSON
	m = mInf
}

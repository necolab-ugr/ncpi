:Reference :Colbert and Pan 2002
:comment: took the NaTa and shifted both activation/inactivation by 6 mv

NEURON	{
	SUFFIX NaTs2_t_frozen
	USEION na READ ena WRITE ina
	RANGE gNaTs2_tbar, gNaTs2_t, ina, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNaTs2_tbar = 0.00001 (S/cm2)
	V_R (mV)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNaTs2_t	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
    m
    h
}

BREAKPOINT	{
	gNaTs2_t = gNaTs2_tbar*m*m*m*h
	ina = gNaTs2_t*(v-ena)
}

INITIAL{
  LOCAL qt
  qt = 2.3^((34-21)/10)

	UNITSOFF
    if(V_R == -32){
    	V_R = V_R+0.0001
    }
		mAlpha = (0.182 * (V_R- -32))/(1-(exp(-(V_R- -32)/6)))
		mBeta  = (0.124 * (-V_R -32))/(1-(exp(-(-V_R -32)/6)))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = (1/(mAlpha + mBeta))/qt

    if(V_R == -60){
      V_R = V_R + 0.0001
    }
		hAlpha = (-0.015 * (V_R- -60))/(1-(exp((V_R- -60)/6)))
		hBeta  = (-0.015 * (-V_R -60))/(1-(exp((-V_R -60)/6)))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = (1/(hAlpha + hBeta))/qt
	UNITSON
	m = mInf
	h = hInf
}

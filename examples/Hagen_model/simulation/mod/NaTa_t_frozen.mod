:Reference :Colbert and Pan 2002

NEURON	{
	SUFFIX NaTa_t_frozen
	USEION na READ ena WRITE ina
	RANGE gNaTa_tbar, gNaTa_t, ina, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNaTa_tbar = 0.00001 (S/cm2)
	V_R (mV)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNaTa_t	(S/cm2)
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
	gNaTa_t = gNaTa_tbar*m*m*m*h
	ina = gNaTa_t*(v-ena)
}

INITIAL{
  LOCAL qt
  qt = 2.3^((34-21)/10)

  UNITSOFF
    if(V_R == -38){
    	V_R = V_R+0.0001
    }
		mAlpha = (0.182 * (V_R- -38))/(1-(exp(-(V_R- -38)/6)))
		mBeta  = (0.124 * (-V_R -38))/(1-(exp(-(-V_R -38)/6)))
		mTau = (1/(mAlpha + mBeta))/qt
		mInf = mAlpha/(mAlpha + mBeta)

    if(V_R == -66){
      V_R = V_R + 0.0001
    }

		hAlpha = (-0.015 * (V_R- -66))/(1-(exp((V_R- -66)/6)))
		hBeta  = (-0.015 * (-V_R -66))/(1-(exp((-V_R -66)/6)))
		hTau = (1/(hAlpha + hBeta))/qt
		hInf = hAlpha/(hAlpha + hBeta)
	UNITSON
	m = mInf
	h = hInf
}

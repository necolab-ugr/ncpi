:Comment : mtau deduced from text (said to be 6 times faster than for NaTa)
:Comment : so I used the equations from NaT and multiplied by 6
:Reference : Modeled according to kinetics derived from Magistretti & Alonso 1999
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX Nap_Et2_linearized
	USEION na READ ena WRITE ina
	RANGE gNap_Et2bar, gNap_Et2, ina, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNap_Et2bar = 0.00001 (S/cm2)
    V_R (mV)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNap_Et2	(S/cm2)
	mInf
    dminf
	mTau
	mAlpha
	mBeta
	hInf
	dhinf
    hTau
	hAlpha
	hBeta
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	ina = gNap_Et2bar*(mInf^3*hInf*(v-4*V_R + 3*ena) + 3*mInf^2 * hInf * (V_R - ena)*(dminf * m + mInf) + mInf^3 * (V_R - ena) * (dhinf * h + hInf))
}

DERIVATIVE states	{
	m' = (v - V_R - m)/mTau
	h' = (v - V_R - h)/hTau
}

INITIAL{
  LOCAL qt
  qt = 2.3^((34-21)/10)
  UNITSOFF
		mInf = 1.0/(1+exp((V_R- -52.6)/-4.6))
		mAlpha = (0.182 * (V_R- -38))/(1-(exp(-(V_R- -38)/6)))
		mBeta  = (0.124 * (-V_R -38))/(1-(exp(-(-V_R -38)/6)))
		mTau = 6*(1/(mAlpha + mBeta))/qt
        dminf = - exp((V_R - - 52.6)/-4.6) / (-4.6*(1 + exp((V_R - - 52.6)/-4.6))^2)
		hInf = 1.0/(1+exp((V_R- -48.8)/10))
        hAlpha = -2.88e-6 * (V_R + 17) / (1 - exp((V_R + 17)/4.63))
        hBeta = 6.94e-6 * (V_R + 64.4) / (1 - exp(-(V_R + 64.4)/2.63))
		hTau = (1/(hAlpha + hBeta))/qt
        dhinf = - exp((V_R - - 48.8)/10) / (10*(1 + exp((V_R - - 48.8)/10))^2)
  UNITSON
	m = 0
	h = 0
}

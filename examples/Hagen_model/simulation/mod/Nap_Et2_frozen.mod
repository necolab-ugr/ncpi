:Comment : mtau deduced from text (said to be 6 times faster than for NaTa)
:Comment : so I used the equations from NaT and multiplied by 6
:Reference : Modeled according to kinetics derived from Magistretti & Alonso 1999
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX Nap_Et2_frozen
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
	gNap_Et2 = gNap_Et2bar*m*m*m*h
	ina = gNap_Et2*(v-ena)
}

INITIAL{
  LOCAL qt
  qt = 2.3^((34-21)/10)

	UNITSOFF
		mInf = 1.0/(1+exp((V_R- -52.6)/-4.6))
    if(V_R == -38){
    	V_R = V_R+0.0001
    }
		mAlpha = (0.182 * (V_R- -38))/(1-(exp(-(V_R- -38)/6)))
		mBeta  = (0.124 * (-V_R -38))/(1-(exp(-(-V_R -38)/6)))
		mTau = 6*(1/(mAlpha + mBeta))/qt

  	if(V_R == -17){
   		V_R = V_R + 0.0001
  	}
    if(V_R == -64.4){
      V_R = V_R+0.0001
    }

		hInf = 1.0/(1+exp((V_R- -48.8)/10))
    hAlpha = -2.88e-6 * (V_R + 17) / (1 - exp((V_R + 17)/4.63))
    hBeta = 6.94e-6 * (V_R + 64.4) / (1 - exp(-(V_R + 64.4)/2.63))
		hTau = (1/(hAlpha + hBeta))/qt
	UNITSON
	m = mInf
	h = hInf
}

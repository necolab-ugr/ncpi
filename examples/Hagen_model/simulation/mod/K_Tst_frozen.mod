:Comment : The transient component of the K current
:Reference : :		Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients,Korngreen and Sakmann, J. Physiology, 2000
:Comment : shifted -10 mv to correct for junction potential
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX K_Tst_frozen
	USEION k READ ek WRITE ik
	RANGE gK_Tstbar, gK_Tst, ik, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gK_Tstbar = 0.00001 (S/cm2)
	V_R (mV)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gK_Tst	(S/cm2)
	mInf
	mTau
	hInf
	hTau
    m
    h
}

BREAKPOINT	{
	gK_Tst = gK_Tstbar*(m^4)*h
	ik = gK_Tst*(v-ek)
}

INITIAL{
  LOCAL qt
  qt = 2.3^((34-21)/10)
	UNITSOFF
		V_R = V_R + 10
		mInf =  1/(1 + exp(-(V_R+0)/19))
		mTau =  (0.34+0.92*exp(-((V_R+71)/59)^2))/qt
		hInf =  1/(1 + exp(-(V_R+66)/-10))
		hTau =  (8+49*exp(-((V_R+73)/23)^2))/qt
		V_R = V_R - 10
	UNITSON
	m = mInf
	h = hInf
}

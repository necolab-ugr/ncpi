:Comment : LVA ca channel. Note: mtau is an approximation from the plots
:Reference : :		Avery and Johnston 1996, tau from Randall 1997
:Comment: shifted by -10 mv to correct for junction potential
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX Ca_LVAst_frozen
	USEION ca READ eca WRITE ica
	RANGE gCa_LVAstbar, gCa_LVAst, ica, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gCa_LVAstbar = 0.00001 (S/cm2)
	V_R (mV)
}

ASSIGNED	{
	v	(mV)
	eca	(mV)
	ica	(mA/cm2)
	gCa_LVAst	(S/cm2)
	mInf
	mTau
	hInf
	hTau
    h
    m
}

BREAKPOINT	{
	gCa_LVAst = gCa_LVAstbar*m*m*h
	ica = gCa_LVAst*(v-eca)
}

INITIAL{
  LOCAL qt
  qt = 2.3^((34-21)/10)

	UNITSOFF
		V_R = V_R + 10
		mInf = 1.0000/(1+ exp((V_R - -30.000)/-6))
		mTau = (5.0000 + 20.0000/(1+exp((V_R - -25.000)/5)))/qt
		hInf = 1.0000/(1+ exp((V_R - -80.000)/6.4))
		hTau = (20.0000 + 50.0000/(1+exp((V_R - -40.000)/7)))/qt
		V_R = V_R - 10
	UNITSON
	m = mInf
	h = hInf
}

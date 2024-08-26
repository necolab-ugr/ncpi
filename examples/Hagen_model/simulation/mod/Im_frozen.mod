:Reference : :		Adams et al. 1982 - M-currents and other potassium currents in bullfrog sympathetic neurones
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX Im_frozen
	USEION k READ ek WRITE ik
	RANGE gImbar, gIm, ik, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gImbar = 0.00001 (S/cm2)
	V_R (mV)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gIm	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
    m
}

BREAKPOINT	{
	gIm = gImbar*m
	ik = gIm*(v-ek)
}

INITIAL{
  LOCAL qt
  qt = 2.3^((34-21)/10)

	UNITSOFF
		mAlpha = 3.3e-3*exp(2.5*0.04*(V_R - -35))
		mBeta = 3.3e-3*exp(-2.5*0.04*(V_R - -35))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = (1/(mAlpha + mBeta))/qt
	UNITSON
	m = mInf
}

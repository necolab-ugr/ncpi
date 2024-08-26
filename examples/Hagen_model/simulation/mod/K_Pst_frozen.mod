:Comment : The persistent component of the K current
:Reference : :		Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients,Korngreen and Sakmann, J. Physiology, 2000
:Comment : shifted -10 mv to correct for junction potential
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX K_Pst_frozen
	USEION k READ ek WRITE ik
	RANGE gK_Pstbar, gK_Pst, ik, V_R
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gK_Pstbar = 0.00001 (S/cm2)
	V_R (mV)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gK_Pst	(S/cm2)
	mInf
	mTau
	hInf
	hTau
    m
    h
}

BREAKPOINT	{
	gK_Pst = gK_Pstbar*m*m*h
	ik = gK_Pst*(v-ek)
}

INITIAL{
    LOCAL qt
    qt = 2.3^((34-21)/10)
	UNITSOFF
		V_R = V_R + 10
		mInf =  (1/(1 + exp(-(V_R+1)/12)))
        if(V_R<-50){
		    mTau =  (1.25+175.03*exp(-V_R * -0.026))/qt
        }else{
            mTau = ((1.25+13*exp(-V_R*0.026)))/qt
        }
		hInf =  1/(1 + exp(-(V_R+54)/-11))
		hTau =  (360+(1010+24*(V_R+55))*exp(-((V_R+75)/48)^2))/qt
		V_R = V_R - 10
	UNITSON
	m = mInf
	h = hInf
}

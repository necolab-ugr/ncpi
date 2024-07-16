: SK-type calcium-activated potassium current
: Reference : Kohler et al. 1996

NEURON {
       SUFFIX SK_E2_frozen
       USEION k READ ek WRITE ik
       USEION ca READ cai
       RANGE gSK_E2bar, gSK_E2, ik
}

UNITS {
      (mV) = (millivolt)
      (mA) = (milliamp)
      (mM) = (milli/liter)
}

PARAMETER {
          v            (mV)
          gSK_E2bar = .000001 (mho/cm2)
          ek           (mV)
          cai          (mM)
}

ASSIGNED {
         zInf
         ik            (mA/cm2)
         gSK_E2	       (S/cm2)
         z
}


BREAKPOINT {
           gSK_E2  = gSK_E2bar * z
           ik   =  gSK_E2 * (v - ek)
}
INITIAL {
          if(cai < 1e-7){
	              cai = cai + 1e-07
          }
          zInf = 1/(1 + (0.00043 / cai)^4.8)
        z = zInf
}
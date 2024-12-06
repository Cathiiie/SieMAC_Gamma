import h5py
import numpy as np
def setCanopychar(Nrcha,NrTYP) -> np.ndarray:
    Canopychar = np.zeros((Nrcha,NrTYP))
    Canopychar[0,:],Canopychar[2,:], Canopychar[3,:] = 16, 0.1,24
    Canopychar[4,:],Canopychar[5,:], Canopychar[6,:] = 0.2, 0.8,0.057
    Canopychar[7,:],Canopychar[9,:], Canopychar[10,:] = 0.389, 0.95,1.25
    Canopychar[11,:],Canopychar[12,:], Canopychar[13,:] = 0.06, -0.06,700
    Canopychar[14,:],Canopychar[15,:], Canopychar[16,:] = 150, 0.7,0.2
    Canopychar[0,3] = 1
    Canopychar[0,4] =0.5
    Canopychar[0,5] = 1
    Canopychar[1,0] = 0.005
    Canopychar[1,1] = 0.05
    Canopychar[1,2] = 0.05
    Canopychar[1,3] = 0.015
    Canopychar[1,4] = 0.01
    Canopychar[1,5] = 0.02
    Canopychar[2,4] = 0.15
    Canopychar[2,5] = 0.15
    Canopychar[3,3] = 2.
    Canopychar[3,4] = 0.5
    Canopychar[3,5] = 1.0
    Canopychar[8,0] = 0.85
    Canopychar[8,1] = 1.1
    Canopychar[8,2] = 0.9
    Canopychar[8,3] = 0.85
    Canopychar[8,4] = 0.7
    Canopychar[8,5] = 0.65
    Canopychar[10,3] = 1
    return Canopychar
def setVPgausDis()-> np.ndarray:
    VPgausDis = np.zeros(5)
    VPgausDis[0]   = 0.0469101
    VPgausDis[1]   = 0.2307534
    VPgausDis[2]   = 0.5
    VPgausDis[3]   = 0.7692465
    VPgausDis[4]   = 0.9530899
    return VPgausDis
def setVPgausWt()-> np.ndarray:
    VPgausWt = np.zeros(5)
    VPgausWt[0] = 0.1184635
    VPgausWt[1] = 0.2393144
    VPgausWt[2] = 0.284444444
    VPgausWt[3] =0.2393144
    VPgausWt[4] = 0.1184635
    return VPgausWt
def CalcEccentricity(JDATE:int)->float:
    ecc = 1+0.033*np.cos(2*3.14*(JDATE*1.0-10)/365)
    return ecc
def WaterVapPres(Dens: float, Pres: float)->float:
    WaterAirRatio = 18.016/28.97
    WaterP = (Dens/(Dens+WaterAirRatio))*Pres
    return WaterP
def Stability(Canchar1: float, Canchar2: float, Solar: float)->float:
    Trateboundary = 500
    if (Solar > Trateboundary):
        Stability = Canchar1
    elif(Solar >0):
        Stability = Canchar1 - ((Trateboundary-Solar)/Trateboundary)*(Canchar1-Canchar2);
    else:
        Stability = Canchar2
    return Stability
def LeafIR(Tk: float, Eps: float)->float:
    Sb = 0.0000000567
    IR = Eps * Sb * 2 * (Tk ** 4)
    return IR
def ResSC(Par:float) -> float:
    SCadj = (0.0027*1.066*Par)/np.sqrt(1+0.0027*0.0027*Par*Par)
    Res = 200 if SCadj<0.1 else 200/SCadj
    return Res
def LHV21(Tk: float) -> float:
    return 2501000 - (2370 * (Tk - 273.0))
def LeafLE(Tleaf: float, Ambvap:float,LatHv: float,GH: float, \
           StomRes: float, TranspireType: float) -> float:
    LeafRes = (1 / (1.075 * (GH / 1231))) + StomRes
    mid = (-2937.4 / Tleaf) - (4.9283 * np.log10(Tleaf)) + 23.5518
    Svp = 10**mid
    SvdTk = 0.2165 * Svp / Tleaf
    Vapdeficit = (SvdTk - Ambvap)
    LE = TranspireType * (1 / LeafRes) * LatHv * Vapdeficit
    return max(0, LE)
def LeafBLC(GHforced:float, Tdelta:float, Llength: float)->float:
    GhFree = 0.
    if (Tdelta >= 0):
        Ghmid = 160000000*Tdelta/(Llength**3.)
        GhFree = 0.5*0.00253*(Ghmid**0.25)/Llength
    BLC = GHforced + GhFree
    return BLC
def LeafH(Tdelta: float, GH: float)->float:
    return 2*GH*Tdelta
def GAMTLD(T1:float, T_Daily:float) -> float:
    T240 = T_Daily
    Ct2 = 230
    if (T1 <260):
        return 0.0
    else:
        #// Temperature at which maximum emission occurs
        Topt = 312.5 + 0.6 * (T240 - 297.)
        X = ((1.0/Topt) - (1.0/ T1))/0.00831
        #// Maximum emission (relative to emission at 30 C)
        Eopt = 2. * np.exp(0.05 * (T_Daily - 297.0)) *np.exp(0.05*(T240-297.0))
        GAMMA= Eopt * Ct2 * np.exp(95. * X) /(Ct2 - 95. * (1. - np.exp(Ct2 * X)))
    return GAMMA
def GAMP21(PPFD1:float,PPFD_Daily:float,PSTD:float) -> float:
    if PPFD_Daily < 0.01:
        return 0.0
    Alpha = 0.004 - 0.0005 * np.log(PPFD_Daily)
    C1 =  0.0468 * np.exp(0.0005 * (PPFD_Daily - PSTD)) * pow(PPFD_Daily,0.6)
    GAMMA= (Alpha * C1 * PPFD1) / np.sqrt(1 + Alpha*Alpha*PPFD1*PPFD1)
    return GAMMA
def Ealti99(temp:float) -> float:
    tdf = 0.13
    TS = 303.15
    GAMMA = np.exp(tdf*(temp-TS))
    return GAMMA

with h5py.File('./CANTYPE_MODISonly_2013.h5', 'r') as file:
    # List all groups/datasets in the file
    CTS = file['CTS'][:]
print(np.shape(CTS)) 
Sinbeta, TairK0, Ws0, PPFD  = 0.00171437, 303.788, 4.86349, 14.3303
WaterQv, LAI, Pres, T_Daily = 0.0190635, 0.19552,99089.4, 303.695
PPFD_Daily = 14.3303
JDATE = 142
I = 1
J = 1

CTF = CTS[:,J-1,I-1]
print(np.shape(CTF))
Layer = 5
StomataDI, Nrcha, NrTYP = 1, 17, 6
TotalCT, ADJUST_FACTOR_LD, ADJUST_FACTOR_LI = 0., 0., 0.
TotalCT = np.sum(CTF)*0.01
Canopychar = setCanopychar(Nrcha,NrTYP)
VPgausDis = setVPgausDis()
VPgausWt = setVPgausWt()
Sb = 0.0000000567
ConvertShadePPFD = 4.6
ConvertSunPPFD = 4.0
Solarconstant = 1367
Solar = PPFD/2.25
MaxSolar = Sinbeta*Solarconstant*CalcEccentricity(JDATE)

VPslwWT = np.zeros(Layer, dtype=np.float32)
Sunfrac = np.zeros(Layer, dtype=np.float32)
SunQn = np.zeros(Layer, dtype=np.float32)
SunQv = np.zeros(Layer, dtype=np.float32)
ShadeQv = np.zeros(Layer, dtype=np.float32)
ShadeQn = np.zeros(Layer, dtype=np.float32)
SunPPFD = np.zeros(Layer, dtype=np.float32)
ShadePPFD = np.zeros(Layer, dtype=np.float32)
QsAbsn = np.zeros(Layer, dtype=np.float32)
QsAbsV = np.zeros(Layer, dtype=np.float32)
QdAbsn = np.zeros(Layer, dtype=np.float32)
QdAbsV = np.zeros(Layer, dtype=np.float32)
Ldepth = np.zeros(Layer, dtype=np.float32)
TairK= np.zeros(Layer, dtype=np.float32)
HumidairPa = np.zeros(Layer, dtype=np.float32)
Wsh = np.zeros(Layer, dtype=np.float32)
Ws = np.zeros(Layer, dtype=np.float32)
ShadeleafIR = np.zeros(Layer, dtype=np.float32)
SunleafIR = np.zeros(Layer, dtype=np.float32)
SunleafTk = np.zeros(Layer, dtype=np.float32)
SunleafSH = np.zeros(Layer, dtype=np.float32)
SunleafLH= np.zeros(Layer, dtype=np.float32)
ShadeleafTk = np.zeros(Layer, dtype=np.float32)
ShadeleafSH = np.zeros(Layer, dtype=np.float32)
ShadeleafLH= np.zeros(Layer, dtype=np.float32)
Ea1tLayer = np.zeros(Layer, dtype=np.float32)
Ea1pLayer = np.zeros(Layer, dtype=np.float32)
EatiLayer = np.zeros(Layer, dtype=np.float32)
Ea1Layer = np.zeros(Layer, dtype=np.float32)


if (TotalCT > 0 and LAI >0):
    for I_CT in range(NrTYP):
        if CTF[I_CT] > 0:
            # ! Solar Fraction Calculation
            ## Transmis Select
            if (MaxSolar <= 0):
                Transmis = 0.5
            elif (MaxSolar < Solar):
                Transmis = 1.0
            else:
                Transmist = Solar/MaxSolar
            ## PPFDdiffFrac Determine
            FracDiff = 0.156+0.86/(1+np.exp(11.1*(Transmis-0.53)))
            PPFDfrac = 0.55-Transmis*0.12
            PPFDdifFrac = FracDiff * (1.06+Transmis*0.4)
            PPFDdifFrac = 1.0 if PPFDdifFrac > 1.0 else PPFDdifFrac

            Qv = PPFDfrac*Solar
            Qdiffv = Qv*PPFDdifFrac
            Qbeamv = Qv-Qdiffv
            Qn = Solar-Qv
            Qdiffn = Qn*FracDiff
            Qbeamn = Qn-Qdiffn

            # WeightSLW
            for LL in range(Layer):
                VPslwWT[LL] = 0.63 + 0.37 * np.exp(-((LAI*VPgausDis[LL])-1))
            ## Canopy Rad
            CANTRAN = Canopychar[16,I_CT]
            LAIadj = LAI/(1-CANTRAN)
            if (((Qbeamv+Qdiffv)>0.001) and (Sinbeta>0.00002 and LAI >0.001)):
                ScatV = Canopychar[4,I_CT]
                ScatN = Canopychar[5,I_CT]
                RefldV = Canopychar[6,I_CT]
                RefldN = Canopychar[7,I_CT]
                Cluster = Canopychar[8,I_CT]
                Kb = Cluster*0.5/Sinbeta
                Kd = 0.8*Cluster

                # // CalcExtCoeff for visible
                P = np.sqrt(1-ScatV)
                ReflbV = 1-np.exp((-2 * ((1 - P) / (1 + P)) * Kb) / (1 + Kb))
                KbpV = Kb*P
                KdpV = Kd*P
                QbAbsV = Kb*Qbeamv*(1-ScatV)
                # // CalcExtCoeff for near IR
                P = np.sqrt(1-ScatN);
                ReflbN = 1-np.exp((-2 * ((1 - P) / (1 + P)) * Kb) / (1 + Kb));
                KbpN = Kb*P
                KdpN = Kd*P
                QbAbsN = Kb*Qbeamn*(1-ScatN);   

                for LL in range(Layer):
                    LAIdepth = LAI* VPgausDis[LL]
                    Sunfrac[LL] = np.exp(-Kb*LAIdepth)
                    # CalcRadComponents for Visible
                    QdAbsVL = Qdiffv*KdpV*(1-RefldV)*np.exp(-KdpV*LAIdepth)
                    QsAbsVL = Qbeamv*((KbpV*(1-ReflbV)*np.exp(-KbpV*LAIdepth))-(Kb*(1-ScatV)*np.exp(-Kb*LAIdepth)));
                    # CalcRadComponents for near IR
                    QdAbsNL = Qdiffn*KdpN*(1-RefldN)*np.exp(-KdpN*LAIdepth)
                    QsAbsNL = Qbeamn*((KbpN*(1-ReflbN)*np.exp(-KbpN*LAIdepth))-(Kb*(1-ScatN)*np.exp(-Kb*LAIdepth)))
                    # Convert to Sun PPFD and Shade PPFD
                    ShadePPFD[LL] = (QdAbsVL+QsAbsVL)*ConvertShadePPFD/(1-ScatV);
                    SunPPFD[LL] = ShadePPFD[LL]+(QbAbsV*ConvertSunPPFD/(1-ScatV));
                    QdAbsV[LL] = QdAbsVL
                    QsAbsV[LL] = QsAbsVL
                    QdAbsn[LL] = QdAbsNL
                    QsAbsn[LL] = QsAbsNL
                    ShadeQv[LL] = QdAbsVL+QsAbsVL
                    SunQv[LL] = ShadeQv[LL]+QbAbsV
                    ShadeQn[LL] = QdAbsNL+QsAbsNL
                    SunQn[LL] = ShadeQn[LL]+QbAbsN
            else:
                QbAbsV = 0
                QdAbsn = 0
                for LL in range(Layer):
                    Sunfrac[LL] = 0.2
                    SunQn[LL] = 0.
                    ShadeQn[LL] = 0.
                    SunQv[LL] = 0.
                    ShadeQv[LL] = 0.
                    SunPPFD[LL] = 0.
                    ShadePPFD[LL] = 0.
                    QdAbsV[LL]=0.
                    QsAbsV[LL] = 0.
                    QdAbsn[LL]=0.
                    QsAbsn[LL] = 0.  
            # !Finish Calculate Sunfrac, SunQn, ShadeQn, SunQv, ShadeQv, SunPPFD, ShadePPFD
            # !QdAbsV, QsAbsV, QdAbsn, QsAbsn
            # Humidity     
            HumidairPa0 = WaterVapPres(WaterQv,Pres)
            # Staibility
            Trate = Stability(Canopychar[11,I_CT],Canopychar[12,I_CT],Solar)
            # Canopy EB (Energy Balance)
            #
            Cdepth = Canopychar[0,I_CT]
            Lwidth = Canopychar[1,I_CT]
            Llength = Canopychar[2,I_CT]
            Cheight = Canopychar[3,I_CT]
            Eps = Canopychar[9,I_CT]
            TranspireType = Canopychar[10,I_CT] 
            # Determine Deltah
            if (TairK0 > 288):
                Deltah = Canopychar[13,I_CT]/Cheight
            elif (TairK0 >278):
                Deltah = (Canopychar[13,I_CT]-((288-TairK0)/10)*(Canopychar[13,I_CT]-Canopychar[14,I_CT]))/Cheight
            else:
                Deltah = Canopychar[14,I_CT]/Cheight
            #--TairK0 >288
            for LL in range(Layer):
                Ldepth[LL] = Cdepth*VPgausDis[LL]
                TairK[LL] = TairK0+(Trate*Ldepth[LL])
                HumidairPa[LL] = HumidairPa0+(Deltah*Ldepth[LL])
                Wsh[LL] = (Cheight-Ldepth[LL])-(Canopychar[15,I_CT]*Cheight)
                Ws[LL] = 0.05 if Wsh[LL] < 0.001 else (Ws0*np.log(Wsh[LL])/np.log(Cheight-Canopychar[15,I_CT]*Cheight))
                EmissAtm = 0.642 * (HumidairPa0 / TairK0) ** (1.0 / 7.0)
                IRin = LeafIR(TairK[LL],Eps)/2.; #LeafIR = Unexposed*2
                ShadeleafIR[LL]= 2 * IRin
                SunleafIR[LL]=0.5*LeafIR(TairK0, EmissAtm)/2 + 1.5*IRin
                Ws1 = 0.001 if (Ws[LL] <= 0) else Ws[LL]
                #  Air vapor density kg m-3
                HumidAirKgm3 = 0.002165*HumidairPa[LL]/TairK[LL]
                #  Heat convection coefficient (W m-2 K-1) for forced convection.
                GHforced =  0.0259 / (0.004 * (pow(Llength/Ws[LL],0.5)))
                ### 1. LeafEB for Sun leaf
                #  Stomatal resistence s m-1
                StomRes = ResSC(SunPPFD[LL])
                IRoutairT = LeafIR(TairK[LL],Eps)
                #  Latent heat of vaporization  (J Kg-1)
                LatHv = LHV21(TairK[LL])
                # Latent heat flux
                LHairT = LeafLE(TairK[LL],HumidAirKgm3,LatHv,GHforced,StomRes,TranspireType)
                Q = SunQv[LL]+SunQn[LL]
                IRin = SunleafIR[LL]

                E1 = (Q+IRin-IRoutairT-LHairT)
                print("\n")
                print("LL:" + str(LL))
                E1 = -1 if E1==0 else E1
                Tdelt = 1.
                Balance = 10.
                for II in range(10):
                    if (abs(Balance) > 2):
                        # ! Boundary layer conductance
                        GH1 = LeafBLC(GHforced,Tdelt,Llength)
                        #! Convective heat flux
                        SH1 = LeafH(Tdelt,GH1)
                        #! Latent heat of vaporization (J Kg-1)
                        LatHv = LHV21(TairK[LL]+Tdelt)
                        LH = LeafLE(TairK[LL]+Tdelt,HumidAirKgm3,LatHv,GH1,StomRes,TranspireType)
                        LH1 = LH-LHairT
                        IRout = LeafIR(TairK[LL]+Tdelt,Eps)
                        IRout1 = IRout-IRoutairT
                        Tdelt = E1/((SH1+LH1+IRout1)/Tdelt)
                        Balance = Q+IRin-IRout-SH1-LH;  
                Tdelt = 10 if Tdelt >10 else Tdelt
                Tdelt = -10 if Tdelt < -10 else Tdelt
                Tleaf = TairK[LL]+Tdelt
                GH = LeafBLC(GHforced,Tleaf-TairK[LL],Llength)
                SH = LeafH(Tleaf-TairK[LL],GH)
                LH = LeafLE(Tleaf,HumidAirKgm3,LatHv,GH,StomRes,TranspireType)
                SunleafTk[LL]=Tleaf
                SunleafSH[LL]=SH
                SunleafLH[LL] = LH
                IRout = LeafIR(Tleaf,Eps)
                SunleafIR[LL] = SunleafIR[LL]-IRout

                ### 2. LeafEB for Shade leaf
                #Stomatal resistence s m-1
                StomRes = ResSC(ShadePPFD[LL])
                # Latent heat of vaporization  (J Kg-1)
                LatHv = LHV21(TairK[LL])
                LHairT = LeafLE(TairK[LL],HumidAirKgm3,LatHv,GHforced,StomRes,TranspireType)

                Q = ShadeQv[LL]+ShadeQn[LL]
                IRin = ShadeleafIR[LL]
                E1 = (Q+IRin-IRoutairT-LHairT)
                E1 = -1 if E1==0 else E1
                Tdelt = 1.
                Balance = 10.
                for II in range(10):
                    GH1 = LeafBLC(GHforced,Tdelt,Llength)
                    #! Convective heat flux
                    SH1 = LeafH(Tdelt,GH1)
                    #/! Latent heat of vaporization (J Kg-1)
                    LatHv = LHV21(TairK[LL]+Tdelt)
                    LH = LeafLE(TairK[LL]+Tdelt,HumidAirKgm3,LatHv,GH1,StomRes,TranspireType)
                    LH1 = LH-LHairT
                    IRout = LeafIR(TairK[LL]+Tdelt,Eps)
                    IRout1 = IRout-IRoutairT
                    Tdelt = E1/((SH1+LH1+IRout1)/Tdelt)
                    Balance = Q+IRin-IRout-SH1-LH
                Tdelt = 10 if Tdelt > 10 else Tdelt
                Tdelt = -10 if Tdelt < -10 else Tdelt
                Tleaf = TairK[LL]+Tdelt
                GH = LeafBLC(GHforced,Tleaf-TairK[LL],Llength)
                SH = LeafH(Tleaf-TairK[LL],GH)
                LH = LeafLE(Tleaf,HumidAirKgm3,LatHv,GH,StomRes,TranspireType)
                ShadeleafTk[LL]=Tleaf
                ShadeleafSH[LL]=SH
                ShadeleafLH[LL] = LH
                IRout = LeafIR(Tleaf,Eps)
                ShadeleafIR[LL] = ShadeleafIR[LL]-IRout
            for LL in range(Layer):
                    Ea1tLayer[LL] = 0.
                    Ea1tLayer[LL] = GAMTLD(SunleafTk[LL],T_Daily)*Sunfrac[LL]+GAMTLD(ShadeleafTk[LL], T_Daily)*(1-Sunfrac[LL])

                    Ea1pLayer[LL] = 0.
                    Ea1pLayer[LL] = GAMP21(SunPPFD[LL], PPFD_Daily*0.5,200)*Sunfrac[LL]+GAMP21(ShadePPFD[LL], PPFD_Daily*0.16,50)* (1-Sunfrac[LL])

                    Ea1Layer[LL] = 0.
                    Ea1Layer[LL] = GAMTLD(SunleafTk[LL],T_Daily) * GAMP21(SunPPFD[LL], PPFD_Daily*0.5,200)*Sunfrac[LL]
                    Ea1Layer[LL] = Ea1Layer[LL]+ GAMTLD(ShadeleafTk[LL], T_Daily) * GAMP21(ShadePPFD[LL], PPFD_Daily*0.16,50)* (1-Sunfrac[LL])

                    EatiLayer[LL] = 0.
                    EatiLayer[LL] = Ealti99(SunleafTk[LL])*Sunfrac[LL]+Ealti99(ShadeleafTk[LL])*(1-Sunfrac[LL])


            exit(1)
elif (TotalCT <0):
    print("Total CT can't be negative")
    exit(1)
else:
    FACTOR_LD = 1.0
    FACTOR_LI = 1.0

import numpy as np

m = float(input("masa kg: "))
t = float(input("temperatura st C: "))
rod = 998.56  #float(input("gęstość wody stanowiskowej w 20 st kg/m3: "))
deltat = float(input("dopuszczalna zmiana temperatury podczas pomiaru st C:   "))

def objetosc(m, t, rod):
    
    a0 = 999.83952             # Stałe wielomianu Kella
    a1 = 16.952577
    a2 = -0.0079905127
    a3 = -4.6241757 / 10 ** 5
    a4 = 1.0584601 / 10 ** 7
    a5 = -2.8103006 / 10 ** 10
    b = 0.016887236

    return 1001.06 * (998.2031 * m * (1 + b * t)) / (  
    (a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3  + a4 * t ** 4 + a5 
      * t ** 5 ) * rod ) 


Vc = objetosc(m, t, rod)

print ("Vc z wielomianu Kella Vc = " , Vc)

Vcmax = objetosc(m, t + deltat, rod)

Vcmin = objetosc(m, t - deltat, rod)

print (Vcmax , Vcmin)

beta1 = (Vcmax - Vc) / deltat
beta2 = (Vc - Vcmin) /deltat

#print (beta1, beta2)

avgbeta = round((beta1 + beta2)/2, 5)

#print (avgbeta)


Vmag = float(input("średnia objętość wody w cylindrze Vmag l:     "))

print(" Niepewność poprawki związanej z magazynowaniem masy")

uVmag = deltat * avgbeta * Vmag / np.sqrt(3)

print ( uVmag)

uVmagrel = round (uVmag / Vc *100, 2)

#print ("Względna niepewność poprawki związanej z magazynowaniem masy uVmagrel =" , uVmagrel ,"%")

um = float(input("standardowa niepewność ważenia kg: "))

ddm = ( objetosc(m + um, t, rod) - objetosc(m - um, t, rod)) / ( 2 * um)
#print ("Cm = " , ddm , "kg/l")

ut = float(input("standardowa niepewność pomiaru temperatury st C: "))

ddt = ( objetosc(m, t + ut, rod) - objetosc(m, t - ut, rod)) / ( 2 * ut)
#print ("Ct = " , ddt , "stC/l")


urod = float(input("standardowa niepewność wyznaczenia gęstości wody stan. kg/m3: "))
ddrod = ( objetosc(m, t, rod + urod) - objetosc(m, t, rod - urod)) / ( 2 * urod)
#print ("Crod = " , ddrod , "kg/m3/l")

#udziały

#print ( "Udziały wartości źródłowych w niepewności" )

masa = um * ddm
temp = ut * ddt
gestosc = urod * ddrod

#print ( " Udział niepewności ważenia:" , masa , " l " )
#print ( " Udział niepewności pomiaru temperatury wody:" , temp , " l " )
#print ( " Udział niepewności wyznaczenia gęstości wody:" , gestosc , " l " )

#niepewność standardowa

uvc = np.sqrt((um * masa)**2 + (ut * temp)**2 + (urod * gestosc)**2 + uVmag**2)

print ("Standardowa niepewność Vc:" , uvc , " l ")

uvcrel = round (uvc / Vc *100 , 2)

print ("Względna niepewność Vc:" , uvcrel , "%")
       

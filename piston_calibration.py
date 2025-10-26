import math
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# =========================================
# Stałe modelu (woda – Kell, ściśliwość)
# =========================================
# Wielomian Kella dla gęstości ρ(T) [kg/m³], T w °C
a0 = 999.83952
a1 = 16.952577
a2 = -0.0079905127
a3 = -4.6241757 / 10**5
a4 = 1.0584601 / 10**7
a5 = -2.8103006 / 10**10

# Liniowy współczynnik rozszerzalności cieplnej wody (w modelu masowym)
b = 0.016887236

# Ściśliwość wody B(T) [1/Pa] ≈ b0 + b1*T + b2*T²
b0 = 5.074e-11
b1 = -3.26e-13
b2 = 4.16e-15

# =========================================
# Parametry materiału i geometrii tłoka (stal)
# =========================================
ALPHA_V_STEEL = 52.5e-6   # [1/K] – rozszerzalność OBJĘTOŚCIOWA stali nierdzewnej
E_GPA_DEFAULT = 200.0     # [GPa]
NU_DEFAULT    = 0.30      # [-]
D_MM_DEFAULT  = 217.95    # [mm] średnica wewnętrzna cylindra
Z_MM_DEFAULT  = 3.5       # [mm] grubość ścianki

# Bufory serii
wyniki   = []  # Vc [l]
uvc_list = []  # u(Vc) [l] – z każdego pomiaru


# ------------------------ Narzędzia I/O ------------------------
def get_float(prompt: str) -> float:
    """Bezpieczne wprowadzanie liczby zmiennoprzecinkowej (obsługa przecinka)."""
    while True:
        try:
            return float(input(prompt).replace(",", "."))
        except ValueError:
            print("Błąd: podaj liczbę.")

def get_float_or(prompt: str, default: float) -> float:
    """Jak get_float, ale puste = domyślna."""
    s = input(f"{prompt} [Enter = {default}]: ").strip()
    if s == "":
        return float(default)
    try:
        return float(s.replace(",", "."))
    except ValueError:
        print("Błąd: podaj liczbę – używam domyślnej.")
        return float(default)


# ------------------------ Formatowanie V ± u (2 cyfry zn. dla u) ------------------------
def _round_to_sig(x: float, sig: int = 2) -> tuple[Decimal, int]:
    """
    Zwraca (x_zaokrąglone jako Decimal, places),
    gdzie 'places' to liczba miejsc po przecinku użyta do zaokrąglenia.
    ROUND_HALF_UP (metrologiczne 'połowa w górę').
    """
    if x == 0 or not math.isfinite(x):
        return Decimal("0"), 0
    d = math.floor(math.log10(abs(x)))
    places = sig - 1 - d                  # np. u=0.0123, sig=2 -> places=3
    quant = Decimal(f"1e{-places}")       # places=3 -> 1e-3; places=-2 -> 1e2
    xq = Decimal(str(x)).quantize(quant, rounding=ROUND_HALF_UP)
    return xq, places

def format_V_pm_u(value: float, u: float, unit: str = "l", sig: int = 2) -> str:
    """
    Zwraca 'V ± u unit': u do 'sig' cyfr znaczących, V do tego samego miejsca dzies.
    """
    if u <= 0 or not math.isfinite(u):
        return f"{value:.6f} {unit}"
    u_q, places = _round_to_sig(u, sig=sig)
    quant = Decimal(f"1e{-places}")
    v_q = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)

    if places >= 0:
        return f"{float(v_q):.{places}f} ± {float(u_q):.{places}f} {unit}"
    else:
        return f"{int(v_q)} ± {int(u_q)} {unit}"


# ------------------------ Woda: B(T) i Vc ------------------------
def B_of_T(T: float) -> float:
    """Ściśliwość wody B(T) [1/Pa]."""
    return b0 + b1 * T + b2 * (T ** 2)

def rho_of_T(T: float) -> float:
    """Gęstość wody wg Kella [kg/m³]."""
    return (a0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4 + a5*T**5)

def objetosc(m: float, T: float, p_bar: float, rod: float) -> float:
    """
    Objętość odniesienia Vc [l] (masa→objętość) z uwzględnieniem ściśliwości wody.
    p_bar to NADCIŚNIENIE (gauge). Kompresja liczona względem 0 bar(g).
    """
    compression = 1.0 + B_of_T(T) * (p_bar * 1e5)   # p w Pa
    rho_T = rho_of_T(T)
    return 1001.06 * (998.2031 * m * (1 + b * T)) / (rho_T * rod * compression)


# ------------------------ Stal: redukcja do warunków normalnych ------------------------
def steel_reduction_factor(T: float,
                           p_bar_gauge: float,
                           Tn: float = 20.0,
                           pn_bar_gauge: float = 0.0,
                           alpha_v: float = ALPHA_V_STEEL,
                           D_mm: float = D_MM_DEFAULT,
                           z_mm: float = Z_MM_DEFAULT,
                           E_GPa: float = E_GPA_DEFAULT,
                           nu: float = NU_DEFAULT,
                           closed_ends: bool = True) -> float:
    """
    Współczynnik F, przez który mnożymy V_meas (T, p[gauge]) aby otrzymać V w
    warunkach normalnych stali: Tn=20°C, pn=0 bar(g).
    F = F_T * F_p, gdzie:
      F_T = 1 / (1 + αV * (T - Tn))
      F_p = 1 / (1 + ΔV/V |_p)

    ΔV/V dla cienkościennego cylindra (zamknięte denka):
      dV/V = 2*ε_θ + ε_z
      σ_θ = p*r/t,  σ_z = p*r/(2t),  r=D/2
      ε_θ = (σ_θ - ν σ_z)/E,   ε_z = (σ_z - ν σ_θ)/E
      ⇒ dV/V = (p*D)/(4 E t) * (5 - 4ν)

    Jednostki: p [Pa], D,t [m], E [Pa].
    """
    # Temperatura
    dT = T - Tn
    F_T = 1.0 / (1.0 + alpha_v * dT)

    # Ciśnienie (NADCIŚNIENIE): p_n = 0 bar(g)
    dp_bar = (p_bar_gauge - pn_bar_gauge)
    p_Pa = dp_bar * 1e5

    D = D_mm / 1000.0
    t = z_mm / 1000.0
    E = E_GPa * 1e9

    if closed_ends:
        coeff = (5.0 - 4.0 * nu) / 4.0
    else:
        # otwarte denka (bez naprężenia osiowego)
        coeff = (1.0 - 0.5 * nu)

    dV_over_V = (p_Pa * D) / (E * t) * coeff
    F_p = 1.0 / (1.0 + dV_over_V)
    return F_T * F_p


def steel_reduction_sensitivities(T: float,
                                  p_bar_gauge: float,
                                  n: int,
                                  uT_single: float,
                                  up_single: float,
                                  **kw) -> tuple[float, float]:
    """
    Zwraca (|∂Vn/∂T|, |∂Vn/∂p_bar|) jako współczynniki dla *objętości po redukcji*,
    liczone dla niepewności średnich T̄ i p̄ (u/√n). Zwrócone wartości to
    współczynniki, które mnożysz przez Vc_mean, aby dostać wkład w [l].
    """
    alpha_v = kw.get("alpha_v", ALPHA_V_STEEL)
    Tn = kw.get("Tn", 20.0)
    dT = T - Tn

    F = steel_reduction_factor(T, p_bar_gauge, **kw)
    F_T = 1.0 / (1.0 + alpha_v * dT)

    D = kw.get("D_mm", D_MM_DEFAULT) / 1000.0
    t = kw.get("z_mm", Z_MM_DEFAULT) / 1000.0
    E = kw.get("E_GPa", E_GPA_DEFAULT) * 1e9
    nu = kw.get("nu", NU_DEFAULT)
    closed_ends = kw.get("closed_ends", True)

    if closed_ends:
        coeff = (5.0 - 4.0 * nu) / 4.0
    else:
        coeff = (1.0 - 0.5 * nu)

    p_Pa = p_bar_gauge * 1e5
    one_plus = 1.0 + (p_Pa * D) / (E * t) * coeff
    # F = F_T * 1/one_plus
    dVn_dT_coeff = abs(F * (-alpha_v) / (1.0 + alpha_v * dT))  # [1/K] → dla Vn/Vc_mean

    # dF_p/dp_bar (bar): k = (D/(E t))*coeff*1e5  (bar→Pa)
    kbar = (D / (E * t)) * coeff * 1e5
    dFdp_bar = -kbar / (one_plus ** 2)             # [1/bar]
    dVn_dpbar_coeff = abs(F_T * dFdp_bar)          # [1/bar] → dla Vn/Vc_mean

    # uśrednione niepewności T̄ i p̄
    uT_mean = uT_single / math.sqrt(max(1, n))
    up_mean = up_single / math.sqrt(max(1, n))

    # Zwracamy *współczynniki* już pomnożone przez ū
    return dVn_dT_coeff * uT_mean, dVn_dpbar_coeff * up_mean


# ------------------------ Główna pętla / main ------------------------
def main():
    print("=== Kalibracja tłoka – Vc, uA, redukcja stali do Tn=20°C i pn=0 bar(g) ===")

    # Dane stałe (możesz zmieniać między pomiarami)
    rod   = get_float("gęstość wody stanowiskowej w 20 °C [kg/m³]: ")
    um    = get_float("standardowa niepewność ważenia [kg]: ")
    uT    = get_float("standardowa niepewność temperatury [°C]: ")
    up    = get_float("standardowa niepewność ciśnienia (gauge) [bar]: ")
    urod  = get_float("standardowa niepewność gęstości wody stanowiskowej [kg/m³]: ")
    dTmax = get_float("dopuszczalna zmiana temperatury podczas pomiaru ΔT [°C]: ")

    # Parametry stali/geometrii (z domyślnymi)
    D_mm = get_float_or("średnica wewnętrzna cylindra D [mm]", D_MM_DEFAULT)
    z_mm = get_float_or("grubość ścianki z [mm]", Z_MM_DEFAULT)
    E_GPa = get_float_or("moduł Younga E [GPa]", E_GPA_DEFAULT)
    nu    = get_float_or("współczynnik Poissona ν [-]", NU_DEFAULT)
    alpha_v = get_float_or("rozszerzalność objętościowa stali αV [1/K]", ALPHA_V_STEEL)

    while True:
        # Dane pojedynczego pomiaru
        try:
            k = int(get_float("nr segmentu (liczba całkowita): "))
        except Exception:
            print("Błąd: 'nr segmentu' musi być liczbą całkowitą.")
            continue

        m = get_float("masa [kg]: ")
        T = get_float("temperatura [°C]: ")
        p = get_float("ciśnienie (nadciśnienie gauge) [bar]: ")

        # Objętość odniesienia (woda)
        Vc = objetosc(m, T, p, rod)
        print(f"Vc = {Vc:.6f} l")
        wyniki.append(Vc)

        # Współczynniki czułości – centralne różniczkowanie
        Cm   = 0.0 if (um   == 0) else (objetosc(m + um, T, p, rod)   - objetosc(m - um, T, p, rod))   / (2 * um)
        Ct   = 0.0 if (uT   == 0) else (objetosc(m, T + uT, p, rod)   - objetosc(m, T - uT, p, rod))   / (2 * uT)
        Crod = 0.0 if (urod == 0) else (objetosc(m, T, p, rod + urod) - objetosc(m, T, p, rod - urod)) / (2 * urod)
        Cp   = 0.0 if (up   == 0) else (objetosc(m, T, p + up, rod)   - objetosc(m, T, p - up, rod))   / (2 * up)

        # Składowe niepewności (wkłady w [l])
        U_masa = um   * Cm
        U_temp = uT   * Ct
        U_dens = urod * Crod
        U_pres = up   * Cp

        # Efekt „magazynowania masy” – Twoja geometria
        if dTmax != 0:
            Vc_max = objetosc(m, T + dTmax, p, rod)
            Vc_min = objetosc(m, T - dTmax, p, rod)
            beta1 = (Vc_max - Vc) / dTmax
            beta2 = (Vc - Vc_min) / dTmax
            avgbeta = (beta1 + beta2) / 2.0
        else:
            avgbeta = 0.0

        if k == 1:
            Vmag = 49.077
        else:
            Vmag = 49.953 - math.pi * (2.1795 ** 2) / 4.0 * (0.101 + 0.13402 + (k - 1) * 0.26804)
        uVmag = abs(dTmax * avgbeta * Vmag) / math.sqrt(3)  # prostokątny

        # Pozycjonowanie tłoka – trójkątny
        uVl = 3.7 * Vc / 10**5 / math.sqrt(6)

        uVc = math.sqrt(U_masa**2 + U_temp**2 + U_dens**2 + U_pres**2 + uVmag**2 + uVl**2)
        uvc_list.append(uVc)

        print("Składowe niepewności (wkłady [l]):")
        print(f" • ważenie:           {U_masa:.6g}")
        print(f" • temperatura:       {U_temp:.6g}")
        print(f" • ciśnienie:         {U_pres:.6g}")
        print(f" • gęstość (ρ20):     {U_dens:.6g}")
        print(f" • magazynowanie:     {uVmag:.6g}")
        print(f" • pozycja tłoka:     {uVl:.6g}")
        print(f"Standardowa niepewność u(Vc): {uVc:.6g} l")
        print("Wynik (2 sig):", format_V_pm_u(Vc, uVc, "l", sig=2))
        if Vc != 0:
            print(f"Względna u(Vc): {100.0 * uVc / Vc:.2f} %")

        # Kontynuacja?
        kont = input("Czy chcesz wykonać kolejny pomiar? (t/n): ").strip().lower()
        if kont != "t":
            break

        # Ewentualna zmiana danych stałych
        if input("Zmienić dane stałe (rod, um, uT, up, urod, ΔT, D, z, E, ν, αV)? (t/n): ").strip().lower() == "t":
            rod   = get_float("gęstość wody stanowiskowej w 20 °C [kg/m³]: ")
            um    = get_float("standardowa niepewność ważenia [kg]: ")
            uT    = get_float("standardowa niepewność temperatury [°C]: ")
            up    = get_float("standardowa niepewność ciśnienia (gauge) [bar]: ")
            urod  = get_float("standardowa niepewność gęstości wody stanowiskowej [kg/m³]: ")
            dTmax = get_float("dopuszczalna zmiana temperatury podczas pomiaru ΔT [°C]: ")
            D_mm  = get_float_or("średnica wewnętrzna cylindra D [mm]", D_mm)
            z_mm  = get_float_or("grubość ścianki z [mm]", z_mm)
            E_GPa = get_float_or("moduł Younga E [GPa]", E_GPa)
            nu    = get_float_or("współczynnik Poissona ν [-]", nu)
            alpha_v = get_float_or("rozszerzalność objętościowa stali αV [1/K]", alpha_v)

    # =========================================
    # PODSUMOWANIE SERII
    # =========================================
    if not wyniki:
        print("Brak danych do analizy.")
        return

    print("\n=== Podsumowanie serii pomiarów ===")
    Vc_array = np.array(wyniki, dtype=float)
    n = len(Vc_array)
    Vc_mean = float(np.mean(Vc_array))
    s = float(np.std(Vc_array, ddof=1)) if n > 1 else 0.0
    uA = s / math.sqrt(n) if n > 1 else 0.0

    # uB – RMS z u(Vc) pojedynczych pomiarów (stabilniejsze niż „ostatni pomiar”)
    uB = float(math.sqrt(np.mean(np.array(uvc_list, dtype=float)**2))) if uvc_list else 0.0

    print(f"Liczba pomiarów: {n}")
    print(f"Średnia objętość Vc̄ = {Vc_mean:.6f} l")
    if n > 1:
        print(f"Odchylenie standardowe s(Vc) = {s:.6g} l")
        print(f"Niepewność typu A uA(Vc̄) = {uA:.6g} l")

    # Średnie warunki serii do redukcji STALI do Tn=20°C, pn=0 bar(g)
    T_mean = get_float("Podaj średnią temperaturę serii T̄ [°C]: ")
    p_mean_g = get_float("Podaj średnie NADCIŚNIENIE (gauge) p̄ [bar]: ")

    # Współczynnik redukcji stali i V̄c -> V̄n (warunki normalne stali)
    F = steel_reduction_factor(
        T_mean, p_mean_g,
        Tn=20.0, pn_bar_gauge=0.0,
        alpha_v=alpha_v, D_mm=D_mm, z_mm=z_mm, E_GPa=E_GPa, nu=nu,
        closed_ends=True
    )
    Vn = Vc_mean * F
    rel_corr = (F - 1.0) * 100.0

    print(f"\nRedukcja stali do Tn=20°C, pn=0 bar(g): F = {F:.9f}  (ΔV/V = {rel_corr:.6f} %)")
    print(f"Objętość segmentu w warunkach normalnych: V̄n = {Vn:.6f} l")

    # Niepewność „pomiarowa” przed redukcją: u_meas = sqrt(uA^2 + uB^2)
    u_meas = math.sqrt(uA**2 + uB**2)

    # Wkłady od niepewności T̄ i p̄ w redukcji (uśrednione po serii)
    uT_corr, up_corr = steel_reduction_sensitivities(
        T_mean, p_mean_g, n=n, uT_single=uT, up_single=up,
        Tn=20.0, pn_bar_gauge=0.0, alpha_v=alpha_v,
        D_mm=D_mm, z_mm=z_mm, E_GPa=E_GPa, nu=nu, closed_ends=True
    )

    # Skalowanie u_meas przez F oraz dopisanie wkładów redukcji (w objętości)
    u_scaled  = F * u_meas
    u_T_part  = uT_corr * Vc_mean
    u_p_part  = up_corr * Vc_mean
    u_total_n = math.sqrt(u_scaled**2 + u_T_part**2 + u_p_part**2)
    k = 2.0
    U = k * u_total_n

    print(f"u(V̄n) = {u_total_n:.6g} l   ⇒   U(V̄n) (k=2) = {U:.6g} l")
    print("\nPrezentacja wyników (k=1):", format_V_pm_u(Vn, u_total_n, "l", sig=2))
    print("Prezentacja wyników (k=2):",  format_V_pm_u(Vn, U,          "l", sig=2))

    # Zapis CSV
    save = input("\nCzy zapisać wyniki do pliku CSV? (t/n): ").strip().lower()
    if save == "t":
        fname = input("Nazwa pliku (bez rozszerzenia): ").strip() or "wyniki_tlok"
        with open(fname + ".csv", "w", encoding="utf-8") as f:
            f.write("nr,Vc [l],u(Vc) [l],V±u (2 sig)\n")
            for i, (v, u) in enumerate(zip(Vc_array, uvc_list), start=1):
                f.write(f"{i},{v:.8f},{u:.8e},{format_V_pm_u(v, u, 'l', sig=2)}\n")
            f.write("\nPodsumowanie,,,\n")
            f.write(f"Średnia Vc̄,{Vc_mean:.8f},,\n")
            f.write(f"Odch.std s(Vc),{s:.8e},,\n")
            f.write(f"uA(Vc̄),{uA:.8e},,\n")
            f.write(f"uB(RMS),{uB:.8e},,\n")
            f.write(f"u_meas=sqrt(uA^2+uB^2),{u_meas:.8e},,\n")
            f.write(f"T̄ [°C],{T_mean:.4f},,\n")
            f.write(f"p̄ gauge [bar],{p_mean_g:.6f},,\n")
            f.write(f"F (redukcja stali),{F:.10f},,\n")
            f.write(f"V̄n (Tn=20°C, pn=0 bar[g]),{Vn:.8f},,\n")
            f.write(f"u(V̄n),{u_total_n:.8e},,{format_V_pm_u(Vn, u_total_n, 'l', sig=2)}\n")
            f.write(f"U(V̄n) (k=2),{U:.8e},,{format_V_pm_u(Vn, U, 'l', sig=2)}\n")
        print(f"Wyniki zapisano do pliku {fname}.csv")


if __name__ == "__main__":
    main()

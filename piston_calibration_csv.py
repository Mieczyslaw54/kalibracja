#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
piston_calibration_csv.py — wsad / ręczny, budżety PDF (compact/full), aliasy nagłówków
"""

import csv
import os
import math
import re
import numpy as np

# ===================== Woda: Kell + ściśliwość =====================
a0 = 999.83952
a1 = 16.952577
a2 = -0.0079905127
a3 = -4.6241757 / 10**5
a4 = 1.0584601 / 10**7
a5 = -2.8103006 / 10**10
b  = 0.016887236

b0 = 5.074e-11
b1 = -3.26e-13
b2 = 4.16e-15

# ===================== Stal/geometria – domyślne =====================
ALPHA_V_STEEL = 52.5e-6
E_GPA_DEFAULT = 200.0
NU_DEFAULT    = 0.30
D_MM_DEFAULT  = 217.95
Z_MM_DEFAULT  = 3.5

# ===================== CSV helpers =====================
def ensure_out_with_header(path, header, delimiter=';'):
    path = os.fspath(path)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=delimiter, lineterminator="\n")
        if write_header:
            w.writerow(header)

def write_row(path, row, delimiter=';'):
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=delimiter, lineterminator="\n")
        w.writerow(row)

def open_csv_detect_delimiter(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delim = dialect.delimiter
        except Exception:
            delim = ';'
        reader = csv.DictReader(f, delimiter=delim)
        rows = [r for r in reader]
        headers = reader.fieldnames or []
    return rows, delim, headers

def normalize_keys(d):
    return { (k or "").strip().lower(): (v.strip() if isinstance(v,str) else v) for k, v in d.items() }

def parse_float(val):
    if val is None or val == "":
        return None
    if isinstance(val, str):
        val = val.replace(",", ".").strip()
    try:
        return float(val)
    except Exception:
        return None

# =========================================
# Aliasy nazw kolumn (case-insensitive, po normalize_keys)
# =========================================
ALIASES = {
    'k':     ['k'],
    'm':     ['m','m [kg]','masa'],
    't':     ['t','t [°c]','t[°c]','temperatura','temperature','temp','t [c]','t [degc]'],
    'p':     ['p','p [bar]','p[bar]','p (bar)','p(g)','p_g','p[g]'],
    'rod':   ['rod','ρ_od','rho_od','rho20','rod [kg/m³]','rod [kg/m3]','rho [kg/m³]','rho [kg/m3]'],
    'um':    ['um','u(m)','um [kg]','u_m'],
    'ut':    ['ut','u(t)','ut [°c]','ut[°c]','u_t','ut [degc]','ut[degc]','ut [c]','ut [°c]','ut [oc]','ut [o c]','ut [°c]'],
    'up':    ['up','u(p)','up [bar]','u_p','up[bar]'],
    'urod':  ['urod','u(rod)','u_rho','u_ród','u_rho_od','u(ρ_od)','u(rho)'],
    'dtmax': ['dtmax','deltatmax','Δtmax','delta_t_max','dt_max','Δt_max','d t max','dtmax [°c]','dTmax'],
    'd_mm':  ['d_mm','d [mm]','dmm','średnica','srednica','średnica [mm]','srednica [mm]'],
    'z_mm':  ['z_mm','z [mm]','zmm','grubość','grubosc','grubość [mm]','grubosc [mm]','ścianka','scianka'],
    'e_gpa': ['e_gpa','e [gpa]','e[gpa]','e_gpa'],
    'nu':    ['nu','ν','nu [-]','poisson'],
    'alphav':['alphav','alpha_v','αv','alpha v','alpha v [1/k]']
}

def get_num(rn, key):
    """Spróbuj pobrać liczbę po kluczu i aliasach; rn ma klucze znormalizowane (lowercase/strip)."""
    # bezpośrednio
    v = rn.get(key)
    if v not in (None, ''):
        val = parse_float(v)
        if val is not None:
            return val
    # aliasy
    for a in ALIASES.get(key, []):
        v = rn.get(a)
        if v not in (None, ''):
            val = parse_float(v)
            if val is not None:
                return val
    # fuzzy
    patt = re.compile(r'[^a-z0-9]+')
    target = patt.sub('', key)
    for kname, v in rn.items():
        if v in (None, ''):
            continue
        kn = patt.sub('', kname)
        if target and target == kn:
            val = parse_float(v)
            if val is not None:
                return val
        if key in ('ut','um','up','urod') and kn.startswith(key):
            val = parse_float(v)
            if val is not None:
                return val
    return None

# ===================== Modele fizyczne =====================
def B_of_T(T): return b0 + b1*T + b2*T**2
def rho_of_T(T): return (a0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4 + a5*T**5)

def objetosc(m, T, p_bar, rod):
    compression = 1.0 + B_of_T(T) * (p_bar * 1e5)
    rho_T_poly = rho_of_T(T)
    return 1001.06 * (998.2031 * m * (1 + b * T)) / (rho_T_poly * rod * compression)

def steel_reduction_partials(T, p_bar_gauge, *, Tn=20.0, pn_bar_gauge=0.0,
                             alpha_v=ALPHA_V_STEEL, D_mm=D_MM_DEFAULT, z_mm=Z_MM_DEFAULT,
                             E_GPa=E_GPA_DEFAULT, nu=NU_DEFAULT, closed_ends=True):
    """
    Współczynnik redukcji F = F_T * F_p:
      • F_T – termiczny dla stali (objętościowy αV),
      • F_p – ciśnieniowy z ściśliwości stali (moduł objętościowy K), NIE z geometrii cylindra.
    Zwraca: F, dF/dT, dF/dp_bar, F_T, F_p
    """

    # --- Składnik temperaturowy (stal) ---
    dT = T - Tn
    F_T    = 1.0 / (1.0 + alpha_v * dT)
    dF_dT  = (-alpha_v) / (1.0 + alpha_v * dT)**2  # pochodna F_T po T
    # (na razie bez mnożenia przez F_p; zrobimy to po wyliczeniu F_p)

    # --- Składnik ciśnieniowy (ściśliwość stali) ---
    # p_bar_gauge w bar(g) -> Pa
    p_Pa = (p_bar_gauge - pn_bar_gauge) * 1e5
    E = E_GPa * 1e9
    # Moduł objętościowy: K = E / (3(1-2ν))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    # ΔV/V |_p = - p / K  ->  F_p = 1 / (1 - p/K)
    denom = 1.0 - (p_Pa / K)
    F_p = 1.0 / denom
    # dF_p / d(p_bar) — po bar(g): dp_bar * 1e5 = dp_Pa
    dF_dpbar = (1e5 / K) / (denom**2)

    # --- Składanie i pochodne łączne ---
    F        = F_T * F_p
    dF_dT    = dF_dT * F_p        # łańcuchowo: ∂F/∂T = (∂F_T/∂T) * F_p
    dF_dpbar = F_T * dF_dpbar     # ∂F/∂p = F_T * (∂F_p/∂p)
    return F, dF_dT, dF_dpbar, F_T, F_p


# ===================== Vmag =====================
def Vmag_for_segment(k):
    if k == 1: return 49.077
    return 49.953 - math.pi * (2.1795**2) / 4.0 * (0.101 + 0.13402 + (k - 1) * 0.26804)

# ===================== Budżet – zapis wierszy =====================
def append_budget_rows_piston(budget_path, meas_id, label, segment_k,
                              m, T, p, rod, um, uT, up, urod, dTmax,
                              D_mm, z_mm, E_GPa, nu, alphaV,
                              Vc, uVc, F, F_T, F_p, Vn, uVn,
                              U_masa, U_temp, U_rod, U_p, uVmag, uVl,
                              delimiter=';'):
    ensure_out_with_header(
        budget_path,
        ["pomiar_id","segment_k","etykieta","wielkość","Xi","xi","jedn_xi","u(xi)","jedn_u",
         "rozkład","Ci","jedn_Ci","ui(y)","jedn_ui","% udział"],
        delimiter=delimiter
    )

    def Vn_calc(m_, T_, p_, rod_):
        F_, _, _, _, _ = steel_reduction_partials(T_, p_, Tn=20.0, pn_bar_gauge=0.0,
                                                 alpha_v=alphaV, D_mm=D_mm, z_mm=z_mm, E_GPa=E_GPa, nu=nu, closed_ends=True)
        Vc_ = objetosc(m_, T_, p_, rod_)
        return F_ * Vc_

    Cm   = 0.0 if um==0   else (Vn_calc(m+um, T, p, rod)   - Vn_calc(m-um, T, p, rod))   / (2*um)
    Ct   = 0.0 if uT==0   else (Vn_calc(m, T+uT, p, rod)   - Vn_calc(m, T-uT, p, rod))   / (2*uT)
    Cp   = 0.0 if up==0   else (Vn_calc(m, T, p+up, rod)   - Vn_calc(m, T, p-up, rod))   / (2*up)
    Crod = 0.0 if urod==0 else (Vn_calc(m, T, p, rod+urod) - Vn_calc(m, T, p, rod-urod)) / (2*urod)

    def pct(u_total, u):
        return ((u*u)/(u_total*u_total)*100.0) if (u_total and u_total>0) else 0.0

    rows = []
    rows.append(["m",    m,   "kg",   um,   "kg",   "normalny",   Cm,   "l/kg",     abs(Cm)*um,     "l", pct(uVn, abs(Cm)*um)])
    rows.append(["T",    T,   "°C",   uT,   "°C",   "normalny",   Ct,   "l/°C",     abs(Ct)*uT,     "l", pct(uVn, abs(Ct)*uT)])
    rows.append(["p",    p,   "bar",  up,   "bar",  "normalny",   Cp,   "l/bar",    abs(Cp)*up,     "l", pct(uVn, abs(Cp)*up)])
    rows.append(["ρ_od", rod, "kg/m³",urod, "kg/m³","normalny",   Crod, "l·m³/kg",  abs(Crod)*urod, "l", pct(uVn, abs(Crod)*urod)])
    rows.append(["ΔV_mag", "", "l", dTmax, "°C", "prostokątny", 1.0, "–", uVmag, "l", pct(uVn, uVmag)])
    rows.append(["ΔV_l",   "", "l", "",    "",   "trójkątny",   1.0, "–", uVl,   "l", pct(uVn, uVl)])
    # SUMA w CSV nie jest wiążąca dla PDF; w PDF przeliczamy z komponentów
    rows.append(["SUMA", "", "", "", "", "", "", "", uVn, "l", 100.0])

    for r in rows:
        write_row(budget_path, [meas_id, segment_k, label, "Vn [l]"] + r, delimiter=delimiter)

# ===================== Prezentacja V ± u =====================
from decimal import Decimal, ROUND_HALF_UP

def _round_to_sig(x: float, sig: int = 2):
    if x == 0 or not math.isfinite(x):
        return Decimal("0"), 0
    d = math.floor(math.log10(abs(x)))
    places = sig - 1 - d
    quant = Decimal(f"1e{-places}")
    xq = Decimal(str(x)).quantize(quant, rounding=ROUND_HALF_UP)
    return xq, places

def format_V_pm_u(value: float, u: float, unit: str = "l", sig: int = 2) -> str:
    if u <= 0 or not math.isfinite(u):
        return f"{value:.6f} {unit}"
    u_q, places = _round_to_sig(u, sig=sig)
    quant = Decimal(f"1e{-places}")
    v_q = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    if places >= 0:
        return f"{float(v_q):.{places}f} ± {float(u_q):.{places}f} {unit}"
    else:
        return f"{int(v_q)} ± {int(u_q)} {unit}"

# ===================== Przetwarzanie pojedynczego pomiaru =====================
def process_measurement(row_norm, out_results, out_budgets, meas_id, title_label,
                        D_mm, z_mm, E_GPa, nu, alphaV, delimiter=';'):
    k     = int(get_num(row_norm, 'k') or 0)
    m     = get_num(row_norm, 'm')
    T     = get_num(row_norm, 't')
    p     = get_num(row_norm, 'p')
    rod   = get_num(row_norm, 'rod')

    um    = get_num(row_norm, 'um')   or 0.0
    uT    = get_num(row_norm, 'ut')   or 0.0
    up    = get_num(row_norm, 'up')   or 0.0
    urod  = get_num(row_norm, 'urod') or 0.0
    dTmax = get_num(row_norm, 'dtmax') or 0.0

    if any(v is None for v in [m, T, p, rod]):
        return False

    Vc = objetosc(m, T, p, rod)
    Cm   = 0.0 if um == 0 else (objetosc(m + um, T, p, rod)   - objetosc(m - um, T, p, rod))   / (2 * um)
    Ct   = 0.0 if uT == 0 else (objetosc(m, T + uT, p, rod)   - objetosc(m, T - uT, p, rod))   / (2 * uT)
    Crod = 0.0 if urod==0 else (objetosc(m, T, p, rod + urod) - objetosc(m, T, p, rod - urod)) / (2 * urod)
    Cp   = 0.0 if up == 0 else (objetosc(m, T, p + up, rod)   - objetosc(m, T, p - up, rod))   / (2 * up)

    U_masa = um * Cm
    U_temp = uT * Ct
    U_rod  = urod * Crod
    U_p    = up * Cp

    if dTmax != 0:
        Vc_max = objetosc(m, T + dTmax, p, rod)
        Vc_min = objetosc(m, T - dTmax, p, rod)
        beta1 = (Vc_max - Vc) / dTmax
        beta2 = (Vc - Vc_min) / dTmax
        avgbeta = (beta1 + beta2) / 2.0
    else:
        avgbeta = 0.0

    Vmag = Vmag_for_segment(k or 1)
    uVmag = abs(dTmax * avgbeta * Vmag) / math.sqrt(3)
    uVl   = 3.7 * Vc / 10**5 / math.sqrt(6)
    uVc   = math.sqrt(U_masa**2 + U_temp**2 + U_rod**2 + U_p**2 + uVmag**2 + uVl**2)

    F, dF_dT, dF_dpbar, F_T, F_p = steel_reduction_partials(
        T, p, Tn=20.0, pn_bar_gauge=0.0, alpha_v=alphaV, D_mm=D_mm, z_mm=z_mm, E_GPa=E_GPa, nu=nu, closed_ends=True
    )
    Vn = Vc * F
    u_meas = F * uVc
    u_T    = abs(Vc * dF_dT)    * uT
    u_p    = abs(Vc * dF_dpbar) * up
    uVn    = math.sqrt(u_meas**2 + u_T**2 + u_p**2)

    write_row(out_results, [
        row_norm.get("id") or meas_id, k, m, T, p, rod, um, uT, up, urod, dTmax,
        D_mm, z_mm, E_GPa, nu, alphaV,
        Vc, uVc, F_T, F_p, F, Vn, uVn, format_V_pm_u(Vn, uVn, "l", 2)
    ], delimiter=delimiter)

    append_budget_rows_piston(
        out_budgets, meas_id, title_label, (k or 1),
        m, T, p, rod, um, uT, up, urod, dTmax,
        D_mm, z_mm, E_GPa, nu, alphaV,
        Vc, uVc, F, F_T, F_p, Vn, uVn,
        U_masa, U_temp, U_rod, U_p, uVmag, uVl,
        delimiter=delimiter
    )
    return True

# ===================== PDF generator (compact/full) =====================
def generate_budgets_pdf(title, budget_csv_path, results_csv_path, out_pdf_path, mode='compact'):
    """
    mode: 'compact' -> Xi, xi, u(xi), rozkład, Ci, ui(y), % udział
          'full'    -> + jednostki: jedn_xi, jedn_u, jedn_Ci, jedn_ui
    Dodatki:
      • Nagłówek z trybem (COMPACT/FULL), żeby pliki nie wyglądały identycznie.
      • SUMA liczona w PDF: sqrt(sum(ui(y)^2)) z wierszy składników (bez wiązania do wiersza 'SUMA' z CSV).
      • Podsumowania segmentów -> jedna wspólna tabela (brak osobnych stron per k).
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import csv, os, math, numpy as np

    # --- Fonts ---
    def register_polish_font():
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/liberationsans-regular.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    pdfmetrics.registerFont(TTFont("PLSans", c))
                    return "PLSans"
                except Exception:
                    pass
        return "Helvetica"
    font_main = register_polish_font()

    def fmt(x, sig=6):
        try:
            xf = float(str(x).replace(",", ".").strip())
            return f"{xf:.{sig}g}"
        except Exception:
            return str(x)

    def fmt_pct(x):
        try:
            xf = float(str(x).replace(",", ".").strip())
            return f"{xf:.2f}"
        except Exception:
            return str(x)

    # Read results -> per-k arrays
    meas_to_k = {}
    per_k_vals = {}
    with open(results_csv_path, "r", encoding="utf-8-sig", newline="") as fr:
        rr = csv.DictReader(fr, delimiter=';')
        for row in rr:
            mid = str(row.get("id") or row.get("pomiar_id") or "").strip()
            try: k = int(float((row.get("k") or "0").replace(",", ".")))
            except: k = 0
            meas_to_k[mid] = k
            try: Vn = float((row.get("Vn [l]") or "nan").replace(",", "."))
            except: Vn = None
            try: uVn = float((row.get("u(Vn) [l]") or "nan").replace(",", "."))
            except: uVn = None
            if (Vn is not None) and (uVn is not None):
                per_k_vals.setdefault(k, {"Vn": [], "uVn": []})
                per_k_vals[k]["Vn"].append(Vn)
                per_k_vals[k]["uVn"].append(uVn)

    # Read budgets -> group by seg -> meas
    data_by_seg = {}
    with open(budget_csv_path, "r", encoding="utf-8-sig", newline="") as fb:
        rb = csv.DictReader(fb, delimiter=';')
        for row in rb:
            pid = str(row.get("pomiar_id","")).strip()
            seg = row.get("segment_k")
            if seg in (None, "", "0"):
                seg = meas_to_k.get(pid, 0)
            try: seg = int(seg)
            except: seg = 0
            data_by_seg.setdefault(seg, {})
            data_by_seg[seg].setdefault(pid, [])
            data_by_seg[seg][pid].append(row)

    # Document
    doc = SimpleDocTemplate(out_pdf_path, pagesize=landscape(A4),
                            topMargin=14, bottomMargin=14, leftMargin=10, rightMargin=10)
    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle("TitlePL",  parent=styles["Title"],    fontName=font_main, fontSize=16, leading=18)
    h2_style     = ParagraphStyle("H2PL",     parent=styles["Heading2"], fontName=font_main, fontSize=11, leading=13)
    h3_style     = ParagraphStyle("H3PL",     parent=styles["Heading3"], fontName=font_main, fontSize=9.5, leading=11)
    cell_style   = ParagraphStyle("CellPL",   parent=styles["BodyText"], fontName=font_main, fontSize=7.2, leading=8.6)

    story = []
    mode_name = "COMPACT" if mode=='compact' else "FULL (z jednostkami)"
    story.append(Paragraph(f"Budżety niepewności – kalibracja tłoka: {title}", title_style))
    story.append(Paragraph(f"Tryb eksportu PDF: {mode_name}", h3_style))
    story.append(Spacer(1, 6))

    # Summary per k as one joint table
    if per_k_vals:
        header = ["k","n","V̄n [l]","s(Vn) [l]","uA [l]","uB(RMS) [l]","u(V̄n) [l]","U(k=2) [l]","V̄n ± u","V̄n ± U"]
        sum_table = [header]
        all_Vn = []; all_u = []
        for k in sorted(per_k_vals.keys()):
            V = np.array(per_k_vals[k]["Vn"], dtype=float); Uv = np.array(per_k_vals[k]["uVn"], dtype=float)
            n = len(V); Vn_mean = float(np.mean(V))
            s = float(np.std(V, ddof=1)) if n>1 else 0.0
            uA = s / math.sqrt(n) if n>1 else 0.0
            uB = float(math.sqrt(np.mean(Uv**2))) if n>=1 else 0.0
            u  = math.sqrt(uA**2 + uB**2); U2 = 2.0*u
            sum_table.append([k, n, fmt(Vn_mean), fmt(s), fmt(uA), fmt(uB), fmt(u), fmt(U2),
                              format_V_pm_u(Vn_mean, u, "l", 2), format_V_pm_u(Vn_mean, U2, "l", 2)])
            all_Vn.extend(V.tolist()); all_u.extend(Uv.tolist())

        t = Table(sum_table, repeatRows=1)
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), font_main),
            ('FONTSIZE', (0,0), (-1,-1), 7.8),
            ('GRID', (0,0), (-1,-1), 0.35, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('ALIGN', (0,1), (-1,-1), 'CENTER'),
        ]))
        story.append(Paragraph("Podsumowanie per segment (po redukcji do 20°C, 0 bar[g])", h2_style))
        story.append(Spacer(1, 3)); story.append(t); story.append(Spacer(1, 8))

    # Table schema
    if mode == 'compact':
        header = ["Xi","xi","u(xi)","rozkład","Ci","ui(y)","% udział"]
        widths = [0.19, 0.18, 0.18, 0.12, 0.14, 0.14, 0.05]
        show_units = False
    else:
        header = ["Xi","xi","jedn_xi","u(xi)","jedn_u","rozkład","Ci","jedn_Ci","ui(y)","jedn_ui","% udział"]
        widths = [0.10,0.11,0.07,0.11,0.07,0.10,0.12,0.09,0.12,0.06,0.05]
        show_units = True

    usable_width = landscape(A4)[0] - doc.leftMargin - doc.rightMargin
    col_widths = [p * usable_width for p in widths]

    # Budgets per segment -> measurement
    for seg in sorted(data_by_seg.keys()):
        story.append(Paragraph(f"Segment k = {seg}", h2_style))
        story.append(Spacer(1, 3))

        # sort meas ids numerically when possible
        def _to_int(x):
            try: return int(x)
            except: return 10**9
        for pid in sorted(data_by_seg[seg].keys(), key=_to_int):
            story.append(Paragraph(f"Pomiar {pid}", h3_style))
            rows = data_by_seg[seg][pid]
            # Keep only Vn budgets
            rows = [r for r in rows if (r.get("wielkość") or "").startswith("Vn")]
            if not rows: 
                continue

            # Build table rows; compute SUM from components, not from CSV's last row
            table_data = [header]
            ui_components = []
            for r in rows:
                xi = r.get("Xi","")
                ui_val = r.get("ui(y)","")
                # acc components except explicit SUMA row
                if str(xi).strip().upper() != "SUMA":
                    try:
                        ui_components.append(float(str(ui_val).replace(",", ".")))
                    except Exception:
                        pass

                if show_units:
                    table_data.append([
                        r.get("Xi",""),
                        fmt(r.get("xi","")),
                        r.get("jedn_xi",""),
                        fmt(r.get("u(xi)","")),
                        r.get("jedn_u",""),
                        r.get("rozkład",""),
                        fmt(r.get("Ci","")),
                        r.get("jedn_Ci",""),
                        fmt(ui_val),
                        r.get("jedn_ui",""),
                        fmt_pct(r.get("% udział","")),
                    ])
                else:
                    table_data.append([
                        r.get("Xi",""),
                        fmt(r.get("xi","")),
                        fmt(r.get("u(xi)","")),
                        r.get("rozkład",""),
                        fmt(r.get("Ci","")),
                        fmt(ui_val),
                        fmt_pct(r.get("% udział","")),
                    ])

            # Append computed SUMA row
            rss = math.sqrt(sum(u*u for u in ui_components)) if ui_components else 0.0
            if show_units:
                table_data.append(["SUMA","","","","","", "", "", fmt(rss), "l", "100.00"])
            else:
                table_data.append(["SUMA","", "", "", "", fmt(rss), "100.00"])

            # Chunk long tables
            max_rows = 28
            from reportlab.platypus import Table
            t_chunks = []
            for i in range(0, len(table_data), max_rows):
                chunk = table_data[i:i+max_rows]
                t = Table(chunk, colWidths=col_widths, repeatRows=1)
                t.setStyle(TableStyle([
                    ('FONTNAME', (0,0), (-1,-1), font_main),
                    ('FONTSIZE', (0,0), (-1,-1), 7.1),
                    ('GRID', (0,0), (-1,-1), 0.35, colors.grey),
                    ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (1,1), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('LEFTPADDING', (0,0), (-1,-1), 1.5),
                    ('RIGHTPADDING', (0,0), (-1,-1), 1.5),
                    ('TOPPADDING', (0,0), (-1,-1), 1.5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
                ]))
                story.append(t)
                story.append(Spacer(1, 5))

    doc.build(story)
    return True

# ===================== Tryb CSV (główny) =====================
def run_mode_csv():
    in_path = input("Ścieżka do pliku wejściowego CSV: ").strip().strip('"')
    if not in_path or not os.path.isfile(in_path):
        print("Brak lub błędna ścieżka do CSV."); return

    rows, in_delim, headers = open_csv_detect_delimiter(in_path)
    if not rows:
        print("Brak danych w CSV."); return

    title = input("Tytuł serii (np. numer tłoka): ").strip() or "kalibracja_tloka"

    # Geometria/materiał – domyślne (mogą być nadpisane kolumnami)
    def getf(prompt, default):
        s = input(f"{prompt} [Enter={default}]: ").strip().replace(",", ".")
        return float(s) if s else default

    D_mm  = getf("Średnica wewnętrzna cylindra D [mm]", D_MM_DEFAULT)
    z_mm  = getf("Grubość ścianki z [mm]", Z_MM_DEFAULT)
    E_GPa = getf("Moduł Younga E [GPa]", E_GPA_DEFAULT)
    nu    = getf("Współczynnik Poissona ν [-]", NU_DEFAULT)
    alphaV= getf("Rozszerzalność objętościowa stali αV [1/K]", ALPHA_V_STEEL)

    out_results = f"{title}_wyniki_i_budzety.csv"
    out_budgets = f"{title}_budzety.csv"
    out_summary = f"{title}_podsumowanie.csv"
    out_summary_k = f"{title}_podsumowanie_per_segment.csv"

    ensure_out_with_header(out_results, [
        "id","k","m [kg]","T [°C]","p [bar]","rod [kg/m³]","um [kg]","uT [°C]","up [bar]","urod [kg/m³]","dTmax [°C]",
        "D_mm","z_mm","E_GPa","nu","alphaV",
        "Vc [l]","u(Vc) [l]","F_T","F_p","F","Vn [l]","u(Vn) [l]","Vn ± u (2 sig)"
    ], delimiter=';')

    ensure_out_with_header(out_budgets, [
        "pomiar_id","segment_k","etykieta","wielkość","Xi","xi","jedn_xi","u(xi)","jedn_u",
        "rozkład","Ci","jedn_Ci","ui(y)","jedn_ui","% udział"
    ], delimiter=';')

    processed = 0
    meas_id = 0

    # Słowniki do podsumowania per k
    per_k_Vn = {}
    per_k_uVn = {}

    for r in rows:
        rn = normalize_keys(r)
        # nadpisania per-wiersz (opcjonalne)
        Dmm = parse_float(rn.get("d_mm"))  or D_mm
        zmm = parse_float(rn.get("z_mm"))  or z_mm
        EG  = parse_float(rn.get("e_gpa")) or E_GPa
        nu_ = parse_float(rn.get("nu"))    or nu
        aV  = parse_float(rn.get("alphav"))or alphaV

        meas_id += 1
        ok = process_measurement(rn, out_results, out_budgets, meas_id, title,
                                 Dmm, zmm, EG, nu_, aV, delimiter=';')
        if ok:
            processed += 1
            # Rekalkulacja kluczowych liczb do podsumowania i per k
            k     = int(get_num(rn, 'k') or 0)
            m     = get_num(rn, 'm'); T = get_num(rn, 't'); p = get_num(rn, 'p'); rod = get_num(rn, 'rod')
            um    = (get_num(rn, 'um') or 0.0)
            uT    = (get_num(rn, 'ut') or 0.0)
            up    = (get_num(rn, 'up') or 0.0)
            urod  = (get_num(rn, 'urod') or 0.0)
            dTmax = (get_num(rn, 'dtmax') or 0.0)

            Vc = objetosc(m,T,p,rod)
            Cm   = 0.0 if um == 0 else (objetosc(m + um, T, p, rod)   - objetosc(m - um, T, p, rod))   / (2 * um)
            Ct   = 0.0 if uT == 0 else (objetosc(m, T + uT, p, rod)   - objetosc(m, T - uT, p, rod))   / (2 * uT)
            Crod = 0.0 if urod==0 else (objetosc(m, T, p, rod + urod) - objetosc(m, T, p, rod - urod)) / (2 * urod)
            Cp   = 0.0 if up == 0 else (objetosc(m, T, p + up, rod)   - objetosc(m, T, p - up, rod))   / (2 * up)
            U_masa = um * Cm
            U_temp = uT * Ct
            U_rod  = urod * Crod
            U_p    = up * Cp
            if dTmax != 0:
                Vc_max = objetosc(m, T + dTmax, p, rod)
                Vc_min = objetosc(m, T - dTmax, p, rod)
                beta1 = (Vc_max - Vc) / dTmax
                beta2 = (Vc - Vc_min) / dTmax
                avgbeta = (beta1 + beta2) / 2.0
            else:
                avgbeta = 0.0
            Vmag = Vmag_for_segment(k or 1)
            uVmag = abs(dTmax * avgbeta * Vmag) / math.sqrt(3)
            uVl   = 3.7 * Vc / 10**5 / math.sqrt(6)
            uVc   = math.sqrt(U_masa**2 + U_temp**2 + U_rod**2 + U_p**2 + uVmag**2 + uVl**2)
            F, dF_dT, dF_dpbar, F_T, F_p = steel_reduction_partials(
                T, p, Tn=20.0, pn_bar_gauge=0.0, alpha_v=aV, D_mm=Dmm, z_mm=zmm, E_GPa=EG, nu=nu_, closed_ends=True
            )
            Vn = Vc * F
            u_meas = F * uVc
            u_T    = abs(Vc * dF_dT)    * uT
            u_p    = abs(Vc * dF_dpbar) * up
            uVn    = math.sqrt(u_meas**2 + u_T**2 + u_p**2)

            per_k_Vn.setdefault(k, []).append(Vn)
            per_k_uVn.setdefault(k, []).append(uVn)

    print(f"Przetworzono pomiarów: {processed}")

    # Podsumowanie globalne
    if processed > 0:
        # per segment CSV
        ensure_out_with_header(out_summary_k, [
            "k","n","V̄n [l]","s(Vn) [l]","uA(V̄n) [l]","uB(RMS) [l]","u(V̄n) [l]","U(k=2) [l]","V̄n ± u","V̄n ± U"
        ], delimiter=';')

        all_Vn = []
        all_u  = []
        for k in sorted(per_k_Vn.keys()):
            V = np.array(per_k_Vn[k], dtype=float)
            Uv= np.array(per_k_uVn[k], dtype=float)
            n = len(V)
            Vn_mean = float(np.mean(V))
            s = float(np.std(V, ddof=1)) if n > 1 else 0.0
            uA = s / math.sqrt(n) if n > 1 else 0.0
            uB = float(math.sqrt(np.mean(Uv**2))) if n >= 1 else 0.0
            u_total = math.sqrt(uA**2 + uB**2)
            U = 2.0 * u_total
            write_row(out_summary_k, [
                k, n, Vn_mean, s, uA, uB, u_total, U,
                format_V_pm_u(Vn_mean, u_total, "l", 2),
                format_V_pm_u(Vn_mean, U, "l", 2)
            ], delimiter=';')
            all_Vn.extend(V.tolist())
            all_u.extend(Uv.tolist())

        # global summary (cała seria)
        Vn_arr = np.array(all_Vn, dtype=float)
        un_arr = np.array(all_u, dtype=float)
        n = len(Vn_arr)
        Vn_mean = float(np.mean(Vn_arr))
        s = float(np.std(Vn_arr, ddof=1)) if n > 1 else 0.0
        uA = s / math.sqrt(n) if n > 1 else 0.0
        uB = float(math.sqrt(np.mean(un_arr**2))) if n >= 1 else 0.0
        u_total = math.sqrt(uA**2 + uB**2)
        U = 2.0 * u_total

        ensure_out_with_header(out_summary, [
            "n","V̄n [l]","s(Vn) [l]","uA(V̄n) [l]","uB(RMS) [l]","u(V̄n) [l]","U(k=2) [l]","V̄n ± u","V̄n ± U"
        ], delimiter=';')
        write_row(out_summary, [
            n, Vn_mean, s, uA, uB, u_total, U,
            format_V_pm_u(Vn_mean, u_total, "l", 2),
            format_V_pm_u(Vn_mean, U, "l", 2)
        ], delimiter=';')

        # PDF (compact + full)
        gen = input("Wygenerować PDF z budżetami? (T/N): ").strip().upper()
        if gen == "T":
            ok1 = generate_budgets_pdf(title, out_budgets, out_results, f"{title}_budzety_compact.pdf", mode='compact')
            ok2 = generate_budgets_pdf(title, out_budgets, out_results, f"{title}_budzety_full.pdf", mode='full')
            if ok1 or ok2:
                print(f"PDF zapisane: {title}_budzety_compact.pdf, {title}_budzety_full.pdf")

# ===================== Tryb ręczny – interaktywny =====================
def run_mode_manual():
    print("=== Tryb ręczny (pojedyncze pomiary, zapis do CSV) ===")
    title = input("Tytuł serii (np. numer tłoka): ").strip() or "kalibracja_tloka"

    def getf(prompt, default):
        s = input(f"{prompt} [Enter={default}]: ").strip().replace(",", ".")
        return float(s) if s else default

    D_mm  = getf("D [mm]", D_MM_DEFAULT)
    z_mm  = getf("z [mm]", Z_MM_DEFAULT)
    E_GPa = getf("E [GPa]", E_GPA_DEFAULT)
    nu    = getf("ν [-]",   NU_DEFAULT)
    alphaV= getf("αV [1/K]",ALPHA_V_STEEL)

    out_results = f"{title}_wyniki_i_budzety.csv"
    out_budgets = f"{title}_budzety.csv"
    out_summary = f"{title}_podsumowanie.csv"
    out_summary_k = f"{title}_podsumowanie_per_segment.csv"

    ensure_out_with_header(out_results, [
        "id","k","m [kg]","T [°C]","p [bar]","rod [kg/m³]","um [kg]","uT [°C]","up [bar]","urod [kg/m³]","dTmax [°C]",
        "D_mm","z_mm","E_GPa","nu","alphaV",
        "Vc [l]","u(Vc) [l]","F_T","F_p","F","Vn [l]","u(Vn) [l]","Vn ± u (2 sig)"
    ], delimiter=';')
    ensure_out_with_header(out_budgets, [
        "pomiar_id","segment_k","etykieta","wielkość","Xi","xi","jedn_xi","u(xi)","jedn_u",
        "rozkład","Ci","jedn_Ci","ui(y)","jedn_ui","% udział"
    ], delimiter=';')

    meas_id = 0
    per_k_Vn = {}
    per_k_uVn = {}

    while True:
        meas_id += 1
        def getv(prompt): return float(input(prompt).strip().replace(",", "."))

        k     = int(getv("k (nr segmentu) [np. 1]: "))
        m     = getv("m [kg]: "); T = getv("T [°C]: "); p = getv("p (gauge) [bar]: "); rod = getv("ρ_od [kg/m³] (20°C): ")
        um    = (parse_float(input("u(m) [kg] (Enter=0): ").replace(",",".")) or 0.0)
        uT    = (parse_float(input("u(T) [°C] (Enter=0): ").replace(",",".")) or 0.0)
        up    = (parse_float(input("u(p) [bar] (Enter=0): ").replace(",",".")) or 0.0)
        urod  = (parse_float(input("u(ρ_od) [kg/m³] (Enter=0): ").replace(",",".")) or 0.0)
        dTmax = (parse_float(input("ΔT_max [°C] (Enter=0): ").replace(",",".")) or 0.0)

        rn = {"id": str(meas_id), "k": str(k), "m": str(m), "t": str(T), "p": str(p), "rod": str(rod),
              "um": str(um), "ut": str(uT), "up": str(up), "urod": str(urod), "dtmax": str(dTmax)}

        ok = process_measurement(rn, out_results, out_budgets, meas_id, title,
                                 D_mm, z_mm, E_GPA, nu, alphaV, delimiter=';')
        if ok:
            # re-licz tylko do per-k
            Vc = objetosc(m,T,p,rod)
            Cm   = 0.0 if um == 0 else (objetosc(m + um, T, p, rod)   - objetosc(m - um, T, p, rod))   / (2 * um)
            Ct   = 0.0 if uT == 0 else (objetosc(m, T + uT, p, rod)   - objetosc(m, T - uT, p, rod))   / (2 * uT)
            Crod = 0.0 if urod==0 else (objetosc(m, T, p, rod + urod) - objetosc(m, T, p, rod - urod)) / (2 * urod)
            Cp   = 0.0 if up == 0 else (objetosc(m, T, p + up, rod)   - objetosc(m, T, p - up, rod))   / (2 * up)
            U_masa = um * Cm; U_temp = uT * Ct; U_rod = urod * Crod; U_p = up * Cp
            if dTmax != 0:
                Vc_max = objetosc(m, T + dTmax, p, rod)
                Vc_min = objetosc(m, T - dTmax, p, rod)
                beta1 = (Vc_max - Vc) / dTmax; beta2 = (Vc - Vc_min) / dTmax; avgbeta = (beta1 + beta2) / 2.0
            else:
                avgbeta = 0.0
            Vmag = Vmag_for_segment(k or 1); uVmag = abs(dTmax * avgbeta * Vmag) / math.sqrt(3); uVl = 3.7 * Vc / 10**5 / math.sqrt(6)
            uVc = math.sqrt(U_masa**2 + U_temp**2 + U_rod**2 + U_p**2 + uVmag**2 + uVl**2)
            F, dF_dT, dF_dpbar, F_T, F_p = steel_reduction_partials(T, p, Tn=20.0, pn_bar_gauge=0.0,
                                                                     alpha_v=alphaV, D_mm=D_mm, z_mm=z_mm, E_GPa=E_GPa, nu=nu, closed_ends=True)
            Vn = Vc * F; u_meas = F * uVc; u_T = abs(Vc * dF_dT) * uT; u_p = abs(Vc * dF_dpbar) * up; uVn = math.sqrt(u_meas**2 + u_T**2 + u_p**2)
            per_k_Vn.setdefault(k, []).append(Vn); per_k_uVn.setdefault(k, []).append(uVn)

        cont = input("Kolejny pomiar? (T/N): ").strip().upper()
        if cont != "T":
            break

    # podsumowanie per k + globalne
    if per_k_Vn:
        out_summary_k = f"{title}_podsumowanie_per_segment.csv"
        ensure_out_with_header(out_summary_k, [
            "k","n","V̄n [l]","s(Vn) [l]","uA(V̄n) [l]","uB(RMS) [l]","u(V̄n) [l]","U(k=2) [l]","V̄n ± u","V̄n ± U"
        ], delimiter=';')

        all_Vn = []; all_u = []
        for k in sorted(per_k_Vn.keys()):
            V = np.array(per_k_Vn[k], dtype=float); Uv = np.array(per_k_uVn[k], dtype=float)
            n = len(V); Vn_mean = float(np.mean(V))
            s = float(np.std(V, ddof=1)) if n>1 else 0.0; uA = s/math.sqrt(n) if n>1 else 0.0
            uB = float(math.sqrt(np.mean(Uv**2))) if n>=1 else 0.0; u_total = math.sqrt(uA**2 + uB**2); U = 2.0*u_total
            write_row(out_summary_k, [k, n, Vn_mean, s, uA, uB, u_total, U,
                                      format_V_pm_u(Vn_mean, u_total, "l", 2),
                                      format_V_pm_u(Vn_mean, U, "l", 2)], delimiter=';')
            all_Vn.extend(V.tolist()); all_u.extend(Uv.tolist())

        Vn_arr = np.array(all_Vn, dtype=float); un_arr = np.array(all_u, dtype=float)
        n = len(Vn_arr); Vn_mean = float(np.mean(Vn_arr)); s = float(np.std(Vn_arr, ddof=1)) if n>1 else 0.0
        uA = s/math.sqrt(n) if n>1 else 0.0; uB = float(math.sqrt(np.mean(un_arr**2))) if n>=1 else 0.0
        u_total = math.sqrt(uA**2 + uB**2); U = 2.0*u_total

        ensure_out_with_header(out_summary, [
            "n","V̄n [l]","s(Vn) [l]","uA(V̄n) [l]","uB(RMS) [l]","u(V̄n) [l]","U(k=2) [l]","V̄n ± u","V̄n ± U"
        ], delimiter=';')
        write_row(out_summary, [n, Vn_mean, s, uA, uB, u_total, U,
                                format_V_pm_u(Vn_mean, u_total, "l", 2),
                                format_V_pm_u(Vn_mean, U, "l", 2)], delimiter=';')

        gen = input("Wygenerować PDF z budżetami? (T/N): ").strip().upper()
        if gen == "T":
            ok1 = generate_budgets_pdf(title, out_budgets, out_results, f"{title}_budzety_compact.pdf", mode='compact')
            ok2 = generate_budgets_pdf(title, out_budgets, out_results, f"{title}_budzety_full.pdf", mode='full')
            if ok1 or ok2: print(f"PDF zapisane: {title}_budzety_compact.pdf, {title}_budzety_full.pdf")

# ===================== Main =====================
def main():
    print("=== Kalibracja tłoka – tryb wsadowy (CSV) lub ręczny ===")
    print("1 – Wczytaj dane z pliku CSV")
    print("2 – Tryb ręczny (interaktywny)")
    choice = input("Wybór (1/2): ").strip()
    if choice == "1":
        run_mode_csv()
    elif choice == "2":
        run_mode_manual()
    else:
        print("Nieprawidłowy wybór.")

if __name__ == "__main__":
    main()

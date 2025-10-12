import csv
import os
import sys
import numpy as np

# =========================================
# Stałe wielomianu Kella
# =========================================
a0 = 999.83952
a1 = 16.952577
a2 = -0.0079905127
a3 = -4.6241757 / 10 ** 5
a4 = 1.0584601 / 10 ** 7
a5 = -2.8103006 / 10 ** 10
b  = 0.016887236

# =========================================
# Narzędzia CSV / IO
# =========================================
def parse_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    return float(s)

def ensure_out_with_header(path, header, delimiter=';'):
    need_header = (not os.path.isfile(path)) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=delimiter, lineterminator="\n")
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
    return { (k or "").strip().lower(): v for k, v in d.items() }

# =========================================
# Wejście ręczne
# =========================================
def get_float(prompt, default=None):
    while True:
        val = input(f"{prompt} [{'Enter' if default is not None else ''}={default}]: ").strip()
        if val == "" and default is not None:
            return default
        try:
            return float(val.replace(",", "."))
        except ValueError:
            print("Błąd: podaj liczbę.")

def get_string(prompt, default=None):
    val = input(f"{prompt} [{'Enter' if default else ''}={default}]: ").strip()
    return val if val != "" else (default or "")

# =========================================
# Fizyka / model
# =========================================
def objetosc(m, T, p, rod):
    """Objętość odniesienia Vc [l] wg modelu."""
    return 1001.06 * (998.2031 * m * (1 + b * T)) / (
        ((a0 + a1 * T + a2 * T**2 + a3 * T**3 + a4 * T**4 + a5 * T**5) * rod) * (1 + 5.007 * p / 1e5)
    )

# =========================================
# Budżety (CSV) – z % udziałów
# =========================================
def append_budget_rows(budget_path, measurement_id, Q_txt,
                       m, T, p, rod, td, ts, pd, ps,
                       um, uT, urod, up, Vi, uread,
                       Vc, Cm, Ct, Crod, Cp, udVtapr, udVpapr,
                       eproc, ueproc):
    # Nagłówek pozostaje bez zmian:
    # ["pomiar_id","Q","wielkość","Xi","xi","jedn_xi","u(xi)","jedn_u",
    #  "rozkład","Ci","jedn_Ci","ui(y)","jedn_ui","% udział"]
    ensure_out_with_header(
        budget_path,
        ["pomiar_id","Q","wielkość","Xi","xi","jedn_xi","u(xi)","jedn_u",
         "rozkład","Ci","jedn_Ci","ui(y)","jedn_ui","% udział"],
        delimiter=';'
    )

    # ----- Budżet Vc [l]
    vc_rows = []
    vc_rows.append(["m",    m,   "kg",   um,   "kg",   "normalny",   Cm,   "l/kg",     abs(Cm)*um,     "l"])
    vc_rows.append(["t",    T,   "°C",   uT,   "°C",   "normalny",   Ct,   "l/°C",     abs(Ct)*uT,     "l"])
    vc_rows.append(["p",    p,   "bar",  up,   "bar",  "normalny",   Cp,   "l/bar",    abs(Cp)*up,     "l"])
    vc_rows.append(["ρ20",  rod, "kg/m³",urod, "kg/m³","normalny",   Crod, "l·m³/kg",  abs(Crod)*urod, "l"])
    vc_rows.append(["δVc-tapr", 0.0, "l", udVtapr, "l", "prostokątny", 1.0, "-", udVtapr, "l"])
    vc_rows.append(["δVc-papr", 0.0, "l", udVpapr, "l", "prostokątny", 1.0, "-", udVpapr, "l"])

    uVc_sq_list = [row[8]**2 for row in vc_rows]
    uVc_total = np.sqrt(sum(uVc_sq_list))

    # Wiersze składników
    for row, sq in zip(vc_rows, uVc_sq_list):
        pct = (sq/(uVc_total**2)*100.0) if uVc_total else 0.0
        write_row(budget_path, [measurement_id, Q_txt, "Vc [l]"] + row + [round(pct,2)], delimiter=';')

    # Wiersz SUMA (zgodnie z Twoim układem pól)
    # [pomiar_id, Q, wielkość, Xi(=Vc), xi, jedn_xi, u(xi), jedn_u, rozkład, Ci("SUMA"), jedn_Ci, ui(y), jedn_ui, %]
    write_row(
    budget_path,
    [measurement_id, Q_txt, "Vc [l]", "Vc", Vc, "l", "", "", "", "SUMA", "", uVc_total, "l", 100.0],
    delimiter=';'

    )

    # ----- Budżet e [%] (wkłady od Vi i Vc)
    dedVi = (((Vi + (uread or 0)) - Vc)/Vc - ((Vi - (uread or 0)) - Vc)/Vc) / (2*(uread or 1)) if (uread and uread>0) else 0.0
    dedVc = (((Vi - (Vc + (uVc_total or 0))) / Vc) - ((Vi - (Vc - (uVc_total or 0))) / Vc)) / (2*(uVc_total or 1)) if (uVc_total and uVc_total>0) else 0.0

    e_rows = []
    # Vi – prostokątny
    e_rows.append(["Vi", Vi, "l", uread,     "l", "prostokątny", dedVi, "1/l", abs(dedVi)*(uread or 0)*100.0, "%"])
    # Vc – normalny (skutek złożenia)
    e_rows.append(["Vc", Vc, "l", uVc_total, "l", "normalny",    dedVc, "1/l", abs(dedVc)*(uVc_total or 0)*100.0, "%"])

    ue_sq_list = [row[8]**2 for row in e_rows]
    ue_total_pct = np.sqrt(sum(ue_sq_list))

    for row, sq in zip(e_rows, ue_sq_list):
        pct = (sq/(ue_total_pct**2)*100.0) if ue_total_pct else 0.0
        write_row(budget_path, [measurement_id, Q_txt, "e [%]"] + row + [round(pct,2)], delimiter=';')

    # Wiersz SUMA (z eproc w kolumnie Xi)
    write_row(
    budget_path,
    [measurement_id, Q_txt, "e [%]", "e", eproc, "%", "", "", "", "SUMA", "", ue_total_pct, "%", 100.0],
    delimiter=';'

    )

# =========================================
# Generowanie PDF z budżetów (opcjonalne)
# =========================================
def generate_budgets_pdf(nr, budget_csv_path, out_pdf_path):
    import os
    import csv
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # ---------- Czcionka z polskimi znakami ----------
    def register_polish_font():
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:\\Windows\\Fonts\\DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\Arial.ttf",
            "C:\\Windows\\Fonts\\NotoSans-Regular.ttf",
            "C:\\Windows\\Fonts\\LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/NotoSans-Regular.ttf",
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    pdfmetrics.registerFont(TTFont("PolishMain", path))
                    return "PolishMain"
                except Exception:
                    pass
        return "Helvetica"

    font_main = register_polish_font()

    # ---------- Format liczb ----------
    def fmt_num(x, kind=None):
        try:
            v = float(x)
        except Exception:
            return x
        if kind == "percent":
            return f"{v:.2f}"
        if v == 0.0:
            return "0"
        if abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.4g}"

    # ---------- Wczytanie CSV i grupowanie ----------
    data_by_meas = {}
    with open(budget_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            pid = row["pomiar_id"]
            Q = row["Q"]
            wiel = row["wielkość"]
            data_by_meas.setdefault(pid, {}).setdefault(Q, {}).setdefault(wiel, []).append(row)

    # ---------- PDF: landscape A4 ----------
    doc = SimpleDocTemplate(
        out_pdf_path,
        pagesize=landscape(A4),
        topMargin=18, bottomMargin=18, leftMargin=18, rightMargin=18
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitlePL", parent=styles["Title"], fontName=font_main, fontSize=18, leading=22)
    h2_style = ParagraphStyle("H2PL", parent=styles["Heading2"], fontName=font_main, fontSize=12, leading=14)
    h3_style = ParagraphStyle("H3PL", parent=styles["Heading3"], fontName=font_main, fontSize=10.5, leading=12)
    cell_style = ParagraphStyle("CellPL", parent=styles["BodyText"], fontName=font_main, fontSize=8, leading=9.5)
    header_style = ParagraphStyle("HeaderPL", parent=styles["BodyText"], fontName=font_main, fontSize=8.5, leading=10.5)

    def P(txt):
        return Paragraph(str(txt), cell_style)

    story = []
    story.append(Paragraph(f"Budżety niepewności – wodomierz {nr}", title_style))
    story.append(Spacer(1, 8))

    # Kolumny tabeli
    header = ["Xi","xi","jedn_xi","u(xi)","jedn_u","rozkład","Ci","jedn_Ci","ui(y)","jedn_ui","% udział"]

    usable_width = landscape(A4)[0] - doc.leftMargin - doc.rightMargin
    proportions = [0.07,0.11,0.07,0.11,0.07,0.12,0.12,0.10,0.12,0.07,0.04]
    col_widths = [p * usable_width for p in proportions]

    # ---------- Skład treści ----------
    for pid in sorted(data_by_meas, key=lambda x: int(x)):
        for Q in data_by_meas[pid]:
            story.append(Paragraph(f"Pomiar #{pid} – Q: {Q}", h2_style))

            for wielkosc in ["Vc [l]", "e [%]"]:
                rows_for_table = data_by_meas[pid][Q].get(wielkosc, [])
                if not rows_for_table:
                    continue

                story.append(Paragraph(f"Budżet: {wielkosc}", h3_style))

                table_data = [[Paragraph(h, header_style) for h in header]]

                for r in rows_for_table:
                    Xi = fmt_num(r.get("Xi"))
                    xi = fmt_num(r.get("xi"))
                    jedn_xi = r.get("jedn_xi", "")
                    ux = fmt_num(r.get("u(xi)"))
                    jedn_u = r.get("jedn_u", "")
                    rozklad = r.get("rozkład", "")
                    Ci = r.get("Ci", "")
                    jedn_Ci = r.get("jedn_Ci", "")
                    uiy = fmt_num(r.get("ui(y)"))
                    jedn_uiy = r.get("jedn_ui", "")
                    pct = fmt_num(r.get("% udział"), kind="percent")

                    row_cells = [
                        P(Xi), P(xi), P(jedn_xi), P(ux), P(jedn_u),
                        P(rozklad), P(Ci), P(jedn_Ci), P(uiy), P(jedn_uiy), P(pct)
                    ]
                    table_data.append(row_cells)

                t = Table(table_data, colWidths=col_widths, repeatRows=1, splitByRow=True)
                t.setStyle(TableStyle([
                    ('FONT', (0,0), (-1,-1), font_main),
                    ('FONTSIZE', (0,0), (-1,-1), 8),
                    ('LEADING', (0,0), (-1,-1), 9.5),
                    ('GRID', (0,0), (-1,-1), 0.35, colors.grey),
                    ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (1,1), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('LEFTPADDING', (0,0), (-1,-1), 2),
                    ('RIGHTPADDING', (0,0), (-1,-1), 2),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ]))

                # Wyróżnienie wierszy SUMA (kolumna "Ci" == "SUMA")
                for i, r in enumerate(rows_for_table, start=1):  # start=1, bo 0 to nagłówek
                    if r.get("Ci") == "SUMA":
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0,i), (-1,i), colors.lightgrey),
                            ('FONTNAME', (0,i), (-1,i), font_main),
                            ('FONTSIZE', (0,i), (-1,i), 8.5),
                            ('TEXTCOLOR', (0,i), (-1,i), colors.black),
                            ('LINEABOVE', (0,i), (-1,i), 0.7, colors.grey),
                        ]))

                story.append(t)
                story.append(Spacer(1, 8))

            story.append(Spacer(1, 10))

    try:
        doc.build(story)
        return True
    except Exception as e:
        print(f"⚠️ Błąd przy tworzeniu PDF: {e}")
        return False


# =========================================
# Przetwarzanie (wspólna logika obliczeń)
# =========================================
def process_measurement(nr, Q_txt, m, T, p, rod, td, ts, pd, ps, um, uT, urod, up, Vi, uread,
                        out_results, out_budgets, groups, measurement_id):
    Vc = objetosc(m, T, p, rod)

    Cm   = 0.0 if not um   else (objetosc(m + um, T, p, rod)   - objetosc(m - um, T, p, rod))   / (2 * um)
    Ct   = 0.0 if not uT   else (objetosc(m, T + uT, p, rod)   - objetosc(m, T - uT, p, rod))   / (2 * uT)
    Crod = 0.0 if not urod else (objetosc(m, T, p, rod + urod) - objetosc(m, T, p, rod - urod)) / (2 * urod)
    Cp   = 0.0 if not up   else (objetosc(m, T, p + up, rod)   - objetosc(m, T, p - up, rod))   / (2 * up)

    udVtapr = (objetosc(m, td, p, rod) - objetosc(m, ts, p, rod)) / (2 * np.sqrt(3))
    udVpapr = (objetosc(m, T, ps, rod) - objetosc(m, T, pd, rod)) / (2 * np.sqrt(3))

    U_masa = (um or 0.0)   * Cm
    U_temp = (uT or 0.0)   * Ct
    U_rod  = (urod or 0.0) * Crod
    U_p    = (up or 0.0)   * Cp

    uV = np.sqrt(U_masa**2 + U_temp**2 + U_rod**2 + U_p**2 + udVtapr**2 + udVpapr**2)

    e = (Vi - Vc) / Vc
    eproc = e * 100.0

    # czułości dyskretne dla e względem Vi i Vc
    if uread and uread > 0 and Vc != 0:
        dedVi = (((Vi + uread) - Vc) / Vc - ((Vi - uread) - Vc) / Vc) / (2 * uread)
    else:
        dedVi = (1.0 / Vc) if Vc != 0 else 0.0

    if uV and uV > 0 and Vc != 0:
        dedVc = (((Vi - (Vc + uV)) / Vc) - ((Vi - (Vc - uV)) / Vc)) / (2 * uV)
    else:
        dedVc = (-Vi / (Vc * Vc)) if Vc != 0 else 0.0

    # niepewność pojedynczego e (w %)
    ue = np.sqrt((uread or 0.0)**2 * dedVi**2 + (uV or 0.0)**2 * dedVc**2)
    ueproc = ue * 100.0

    # Wkłady do e w % (potrzebne do agregacji "bez s_e")
    u_vi_pct = abs(dedVi) * (uread or 0.0) * 100.0
    u_vc_pct = abs(dedVc) * (uV    or 0.0) * 100.0

    # Zapis wyników
    write_row(
        out_results,
        [
            nr, Q_txt, m, T, p, rod, td, ts, pd, ps, um, uT, urod, up, Vi, uread,
            Vc, Cm, Ct, Crod, Cp,
            U_masa, U_temp, U_rod, U_p,
            uV, eproc, ueproc
        ],
        delimiter=';'
    )

    # Zapis budżetów
    append_budget_rows(
        out_budgets, measurement_id, Q_txt,
        m, T, p, rod, td, ts, pd, ps,
        um, uT, urod, up, Vi, uread,
        Vc, Cm, Ct, Crod, Cp, udVtapr, udVpapr,
        eproc, ueproc
    )

    # Dane do podsumowania
    groups.setdefault(Q_txt, {"e": [], "ue": [], "u_vi_pct": [], "u_vc_pct": []})
    groups[Q_txt]["e"].append(eproc)
    groups[Q_txt]["ue"].append(ueproc)
    groups[Q_txt]["u_vi_pct"].append(u_vi_pct)
    groups[Q_txt]["u_vc_pct"].append(u_vc_pct)

# =========================================
# Tryb CSV
# =========================================
def run_mode_csv():
    in_path = input("Ścieżka do pliku wejściowego CSV: ").strip().strip('"')
    if not in_path or not os.path.isfile(in_path):
        print("Brak lub błędna ścieżka do CSV.")
        return

    rows, in_delim, headers = open_csv_detect_delimiter(in_path)
    if not rows:
        print("Brak danych w pliku.")
        return

    required = ["q","m","t","p","rod","td","ts","pd","ps","um","ut","urod","up","vi","uread"]
    norm_hdrs = [ (h or "").strip().lower() for h in headers ]
    missing = [c for c in required if c not in norm_hdrs]
    if missing:
        print("Brak wymaganych kolumn:", ", ".join(missing))
        return

    # Numer wodomierza
    has_nr = any(h for h in headers if (h or "").strip().lower() == "nr")
    nr_vals = set()
    if has_nr:
        for r in rows:
            v = (r.get("nr") or "").strip()
            if v: nr_vals.add(v)
    if has_nr and len(nr_vals) == 1:
        nr = list(nr_vals)[0]
    else:
        nr = get_string("Numer wodomierza (do nazw plików)", "wodomierz")

    out_results = f"{nr}_results.csv"
    out_budgets = f"{nr}_budgets.csv"
    out_final   = f"{nr}_final.csv"

    ensure_out_with_header(out_results, [
        "nr", "Q", "m [kg]", "T [°C]", "p [bar]", "rod [kg/m³]", "td [°C]", "ts [°C]",
        "pd [bar]", "ps [bar]", "um [kg]", "uT [°C]", "urod [kg/m³]", "up [bar]",
        "Vi [l]", "uread [l]",
        "Vc [l]", "Cm [l/kg]", "Ct [l/°C]", "Crod [l·m³/kg]", "Cp [l/bar]",
        "Udział_masa [l]", "Udział_temp [l]", "Udział_gestosc [l]", "Udział_cisnienie [l]",
        "uV [l]", "e [%]", "u(e) [%]"
    ], delimiter=';')

    ensure_out_with_header(out_final, 
        ["Q","Średni błąd e [%]","u(średniej) [%]","U(średniej) [%]","n","tryb niepewności","tryb Vi"], 
        delimiter=';')

    groups = {}               # <<< WAŻNE: powstaje w tej funkcji
    measurement_id = 0
    processed = 0

    for r in rows:
        rnorm = normalize_keys(r)
        Q_txt = (r.get("Q") or r.get("q") or "").strip()
        try:
            m    = parse_float(rnorm.get("m"))
            T    = parse_float(rnorm.get("t"))
            p    = parse_float(rnorm.get("p"))
            rod  = parse_float(rnorm.get("rod"))
            td   = parse_float(rnorm.get("td"))
            ts   = parse_float(rnorm.get("ts"))
            pd   = parse_float(rnorm.get("pd"))
            ps   = parse_float(rnorm.get("ps"))
            um   = parse_float(rnorm.get("um"))
            uT   = parse_float(rnorm.get("ut"))
            urod = parse_float(rnorm.get("urod"))
            up   = parse_float(rnorm.get("up"))
            Vi   = parse_float(rnorm.get("vi"))
            uread= parse_float(rnorm.get("uread"))
        except Exception:
            continue

        key = [m,T,p,rod,td,ts,pd,ps,um,uT,urod,up,Vi,uread]
        if any(v is None for v in key):
            continue

        measurement_id += 1
        process_measurement(nr, Q_txt, m, T, p, rod, td, ts, pd, ps, um, uT, urod, up, Vi, uread,
                            out_results, out_budgets, groups, measurement_id)
        processed += 1

    if processed == 0:
        print("Nie przetworzono żadnego wiersza.")
        return

    # ===================== Podsumowanie (WEWNĄTRZ funkcji) =====================
    use_typeA = input("Liczyć niepewność średniej z danych (typ A, uA = s/√n)? (T/N): ").strip().upper() == "T"
    mode_vi = "-"
    if not use_typeA:
        mode_vi = input("Tryb odczytów Vi: [N]iezależne / [C]iągłe: ").strip().upper()
        if mode_vi not in ("N","C"):
            mode_vi = "N"

    k = 2.0
    for Q_txt, vals in groups.items():
        e_list  = np.array(vals["e"], dtype=float)
        n = len(e_list)
        mean_e = float(np.mean(e_list))

        if use_typeA:
            s_e = float(np.std(e_list, ddof=1)) if n > 1 else 0.0
            u_mean = s_e/np.sqrt(n)
            U_mean = k*u_mean
            write_row(out_final, [Q_txt, round(mean_e,2), round(u_mean,2), round(U_mean,2), n, "Typ A (z danych)", "-"], delimiter=';')
        else:
            u_vi_pct_sq = np.array(vals["u_vi_pct"], dtype=float)**2
            u_vc_pct_sq = np.array(vals["u_vc_pct"], dtype=float)**2

            if mode_vi == "C":   # serie ciągłe: Vi ~ 1/n
                u_mean_vi = np.sqrt(np.mean(u_vi_pct_sq)) / n
                mode_vi_txt = "serie ciągłe"
            else:                # niezależne: Vi ~ 1/√n
                u_mean_vi = np.sqrt(np.sum(u_vi_pct_sq)) / n
                mode_vi_txt = "niezależne"

            u_mean_vc = np.sqrt(np.sum(u_vc_pct_sq)) / n

            u_mean = float(np.sqrt(u_mean_vi**2 + u_mean_vc**2))
            U_mean = k*u_mean

            write_row(out_final, [Q_txt, round(mean_e,2), round(u_mean,2), round(U_mean,2), n, "Bez s_e (budżet)", mode_vi_txt], delimiter=';')

    print(f"OK. Zapisano:\n - wyniki: {out_results}\n - budżety: {out_budgets}\n - podsumowanie: {out_final}")

    # PDF?
    gen = input("Wygenerować PDF z budżetami? (T/N): ").strip().upper()
    if gen == "T":
        pdf_ok = generate_budgets_pdf(nr, out_budgets, f"{nr}_budgets.pdf")
        if pdf_ok:
            print(f"PDF zapisany: {nr}_budgets.pdf")

# =========================================
# Tryb ręczny (interaktywny)
# =========================================
def run_mode_manual():
    print("=== Tryb ręczny ===")
    nr = get_string("Numer wodomierza", "wodomierz")

    # Stałe (można zmieniać między pomiarami)
    Q    = get_float("Przepływ Q [m³/h]")
    rod  = get_float("Gęstość wody stanowiskowej w 20°C [kg/m³]")
    urod = get_float("Standardowa niepewność gęstości [kg/m³]")
    um   = get_float("Standardowa niepewność ważenia [kg]")
    uT   = get_float("Standardowa niepewność temperatury [°C]")
    up   = get_float("Standardowa niepewność ciśnienia [bar]")
    uread= get_float("Niepewność objętości odczytu wskazania [l]")

    out_results = f"{nr}_results.csv"
    out_budgets = f"{nr}_budgets.csv"
    out_final   = f"{nr}_final.csv"

    ensure_out_with_header(out_results, [
        "nr", "Q", "m [kg]", "T [°C]", "p [bar]", "rod [kg/m³]", "td [°C]", "ts [°C]",
        "pd [bar]", "ps [bar]", "um [kg]", "uT [°C]", "urod [kg/m³]", "up [bar]",
        "Vi [l]", "uread [l]",
        "Vc [l]", "Cm [l/kg]", "Ct [l/°C]", "Crod [l·m³/kg]", "Cp [l/bar]",
        "Udział_masa [l]", "Udział_temp [l]", "Udział_gestosc [l]", "Udział_cisnienie [l]",
        "uV [l]", "e [%]", "u(e) [%]"
    ], delimiter=';')
    ensure_out_with_header(out_final, 
        ["Q","Średni błąd e [%]","u(średniej) [%]","U(średniej) [%]","n","tryb niepewności","tryb Vi"], 
        delimiter=';')

    groups = {}            # <<< WAŻNE
    measurement_id = 0

    while True:
        # Ewentualna zmiana stałych
        if input("Zmienić wartości (Q, rod, urod, um, uT, up, uread)? (T/N): ").strip().upper() == "T":
            Q    = get_float("Przepływ Q [m³/h]", Q)
            rod  = get_float("Gęstość wody stanowiskowej w 20°C [kg/m³]", rod)
            urod = get_float("Standardowa niepewność gęstości [kg/m³]", urod)
            um   = get_float("Standardowa niepewność ważenia [kg]", um)
            uT   = get_float("Standardowa niepewność temperatury [°C]", uT)
            up   = get_float("Standardowa niepewność ciśnienia [bar]", up)
            uread= get_float("Niepewność objętości odczytu wskazania [l]", uread)

        # Dane pojedynczego pomiaru
        m  = get_float("masa [kg]")
        T  = get_float("temperatura [°C]")
        p  = get_float("ciśnienie [bar]")
        td = get_float("temperatura na wejściu [°C]")
        ts = get_float("temperatura na wyjściu [°C]")
        pd = get_float("ciśnienie na wejściu [bar]")
        ps = get_float("ciśnienie na wyjściu [°C]")  # UWAGA: jeśli to bar, popraw na "bar"
        Vi = get_float("objętość wskazana przez wodomierz [l]")

        measurement_id += 1
        process_measurement(nr, f"{Q}", m, T, p, rod, td, ts, pd, ps, um, uT, urod, up, Vi, uread,
                            out_results, out_budgets, groups, measurement_id)

        if input("Dodać kolejny pomiar? (T/N): ").strip().upper() != "T":
            break

    # ===================== Podsumowanie (WEWNĄTRZ funkcji) =====================
    use_typeA = input("Liczyć niepewność średniej z danych (typ A, uA = s/√n)? (T/N): ").strip().upper() == "T"
    mode_vi = "-"
    if not use_typeA:
        mode_vi = input("Tryb odczytów Vi: [N]iezależne / [C]iągłe: ").strip().upper()
        if mode_vi not in ("N","C"):
            mode_vi = "N"

    k = 2.0
    for Q_txt, vals in groups.items():
        e_list  = np.array(vals["e"], dtype=float)
        n = len(e_list)
        mean_e = float(np.mean(e_list))

        if use_typeA:
            s_e = float(np.std(e_list, ddof=1)) if n > 1 else 0.0
            u_mean = s_e/np.sqrt(n)
            U_mean = k*u_mean
            write_row(out_final, [Q_txt, round(mean_e,2), round(u_mean,2), round(U_mean,2), n, "Typ A (z danych)", "-"], delimiter=';')
        else:
            u_vi_pct_sq = np.array(vals["u_vi_pct"], dtype=float)**2
            u_vc_pct_sq = np.array(vals["u_vc_pct"], dtype=float)**2

            if mode_vi == "C":
                u_mean_vi = np.sqrt(np.mean(u_vi_pct_sq)) / n
                mode_vi_txt = "serie ciągłe"
            else:
                u_mean_vi = np.sqrt(np.sum(u_vi_pct_sq)) / n
                mode_vi_txt = "niezależne"

            u_mean_vc = np.sqrt(np.sum(u_vc_pct_sq)) / n

            u_mean = float(np.sqrt(u_mean_vi**2 + u_mean_vc**2))
            U_mean = k*u_mean

            write_row(out_final, [Q_txt, round(mean_e,2), round(u_mean,2), round(U_mean,2), n, "Bez s_e (budżet)", mode_vi_txt], delimiter=';')

    print(f"OK. Zapisano:\n - wyniki: {out_results}\n - budżety: {out_budgets}\n - podsumowanie: {out_final}")

    gen = input("Wygenerować PDF z budżetami? (T/N): ").strip().upper()
    if gen == "T":
        ok = generate_budgets_pdf(nr, out_budgets, f"{nr}_budgets.pdf")
        if ok:
            print(f"PDF zapisany: {nr}_budgets.pdf")

# =========================================
# Menu główne
# =========================================
def main():
    print("=== Program obliczania błędu i niepewności (CSV/Ręczny) ===")
    print("Wybierz tryb:")
    print("1 – Wczytaj dane z pliku CSV")
    print("2 – Wprowadź dane ręcznie")
    choice = input("Twój wybór (1/2): ").strip()

    if choice == "1":
        run_mode_csv()
    elif choice == "2":
        run_mode_manual()
    else:
        print("Nieprawidłowy wybór.")

if __name__ == "__main__":
    main()

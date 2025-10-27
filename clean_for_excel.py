#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# clean_for_excel.py
# Interaktywny skrypt porządkujący plik CSV/TSV tak, aby Excel poprawnie dzielił kolumny
# i aby nagłówek miał osobno nazwę i jednostkę (XLSX: 2-wierszowy nagłówek).
#
# Wejście: dowolny CSV/TSV; separator wykryty automatycznie (sep=None).
# Wyjścia (4 formaty, w tym samym katalogu co wejście):
#   1) <stem>_clean.xlsx        — XLSX z 2-wierszowym nagłówkiem: nazwa / jednostka
#   2) <stem>_clean_dot.csv     — CSV; separator ';', kropka jako separator dziesiętny
#   3) <stem>_clean_comma.csv   — CSV; separator ';', przecinek jako separator dziesiętny (PL Excel)
#   4) <stem>_clean.tsv         — TSV; tab jako separator, kropka dziesiętna
#
# Uruchomienie:
#   python clean_for_excel.py [ścieżka_do_pliku]
#   (jeśli ścieżka nie zostanie podana, skrypt zapyta interaktywnie)

import sys
from pathlib import Path
import pandas as pd

def split_name_unit(col: str):
    """Rozdziel 'Vi [l]' -> ('Vi','l'). Gdy brak jednostki: ('Vi','')."""
    s = str(col).strip().replace("\ufeff","" )
    lb = s.rfind("["); rb = s.rfind("]")
    if lb != -1 and rb != -1 and rb > lb and rb == len(s) - 1:
        return s[:lb].strip(), s[lb+1:rb].strip()
    return s, ""

def _unquote(s: str) -> str:
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s

def _normalize_path(raw: str) -> Path:
    import os
    s = raw.replace('\ufeff','').strip()
    s = _unquote(s)
    s = os.path.expandvars(os.path.expanduser(s))
    return Path(s)

def _open_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title='Wybierz plik CSV/TSV',
            filetypes=[('CSV/TSV','*.csv *.tsv *.txt'), ('Wszystkie','*.*')]
        )
        return Path(path) if path else None
    except Exception:
        return None

def read_table_interactive(arg_path: str | None):
    while True:
        if arg_path:
            path = arg_path
        else:
            path = input("Podaj ścieżkę do pliku CSV/TSV (albo zostaw puste, aby otworzyć okno wyboru): ")
        if not path:
            dlg = _open_file_dialog()
            if dlg is None:
                print("Nie wybrano pliku.")
                arg_path = None
                continue
            p = dlg
        else:
            p = _normalize_path(path)
        if not p.exists():
            print(f"Nie znaleziono pliku: {p}")
            parent = p.parent if p.parent.exists() else None
            if parent:
                try:
                    print("Zawartość katalogu:")
                    for i, f in enumerate(sorted(parent.iterdir())):
                        if i >= 20:
                            print("..."); break
                        print(" -", f.name)
                except Exception:
                    pass
            use_dlg = input("Otworzyć okno wyboru pliku? [T/n]: ").strip().lower()
            if use_dlg in ("n","no","nie"):
                arg_path = None
                continue
            dlg = _open_file_dialog()
            if dlg is None:
                arg_path = None
                continue
            p = dlg
        try:
            df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
            df.columns = [str(c).strip().replace("\ufeff","") for c in df.columns]
            print(f"\nWczytano {p}  (kształt: {df.shape[0]} wierszy x {df.shape[1]} kolumn)")
            print("\nPodgląd (5 pierwszych wierszy):")
            try:
                print(df.head(5).to_string(index=False))
            except Exception:
                print(df.head(5))
            ok = input("\nUżyć tego pliku? [T/n]: ").strip().lower()
            if ok in ("n", "no", "nie"):
                arg_path = None
                continue
            return df, p
        except Exception as e:
            print(f"Błąd wczytywania: {e}")
            arg_path = None
            continue

def to_multiindex_columns(df: pd.DataFrame):
    names, units = zip(*[split_name_unit(c) for c in df.columns])
    df_mi = df.copy()
    df_mi.columns = pd.MultiIndex.from_arrays([names, units])
    return df_mi

def _choose_xlsx_engine():
    try:
        import openpyxl  # noqa: F401
        return "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa: F401
            return "xlsxwriter"
        except Exception:
            return None

def save_xlsx_two_header(df: pd.DataFrame, out_path: Path):
    engine = _choose_xlsx_engine()
    if engine is None:
        print("[UWAGA] Brak bibliotek 'openpyxl' i 'xlsxwriter' — pomijam tworzenie XLSX.")
        print("        Zainstaluj np.:  py -m pip install openpyxl   (Windows)")
        return
    df_mi = to_multiindex_columns(df)
    with pd.ExcelWriter(out_path, engine=engine) as writer:
        df_mi.to_excel(writer, index=True, sheet_name="Dane")
        ws = writer.sheets["Dane"]
        if engine == "openpyxl":
            ws.freeze_panes = "B3"
        else:
            try:
                ws.freeze_panes(2, 1)
            except Exception:
                pass

def format_decimal_string(val, decimals=","):
    import pandas as pd
    if pd.isna(val):
        return ""
    if isinstance(val, (int,)) or (isinstance(val, float) and float(val).is_integer()):
        return f"{int(val)}"
    if isinstance(val, float):
        s = f"{val:.12g}"
        if decimals == ",":
            s = s.replace(".", ",").replace("e", "E")
        return s
    return str(val)

def save_csv_with_decimal(df: pd.DataFrame, out_path: Path, decimal=",", sep=";"):
    import pandas as pd
    df_out = df.copy()
    if decimal == ",":
        for c in df_out.columns:
            if pd.api.types.is_numeric_dtype(df_out[c]):
                df_out[c] = df_out[c].map(lambda v: format_decimal_string(v, decimals=","))
            else:
                df_out[c] = df_out[c].astype(str)
        df_out.to_csv(out_path, sep=sep, index=False, encoding="utf-8-sig")
    else:
        df_out.to_csv(out_path, sep=sep, index=False, encoding="utf-8-sig", float_format="%.12g")

def save_tsv(df: pd.DataFrame, out_path: Path):
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8", float_format="%.12g")

def main():
    print("=== clean_for_excel.py ===")
    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    df, in_path = read_table_interactive(arg_path)
    stem = in_path.with_suffix("").name
    out_dir = in_path.parent

    xlsx_path = out_dir / f"{stem}_clean.xlsx"
    save_xlsx_two_header(df, xlsx_path)
    print(f"[OK] XLSX: {xlsx_path}" if xlsx_path.exists() else "[INFO] XLSX pominięty.")

    csv_dot = out_dir / f"{stem}_clean_dot.csv"
    save_csv_with_decimal(df, csv_dot, decimal=".", sep=";")
    print(f"[OK] CSV (kropka): {csv_dot}")

    csv_comma = out_dir / f"{stem}_clean_comma.csv"
    save_csv_with_decimal(df, csv_comma, decimal=",", sep=";")
    print(f"[OK] CSV (przecinek): {csv_comma}")

    tsv_path = out_dir / f"{stem}_clean.tsv"
    save_tsv(df, tsv_path)
    print(f"[OK] TSV: {tsv_path}")

    print("\nGotowe.")

if __name__ == "__main__":
    main()

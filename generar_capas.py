# generar_capas.py
# -*- coding: utf-8 -*-
"""
Genera 3 capas GeoJSON para Mineral de la Reforma:
  1) colonias.geojson               (polígonos de colonias + LOC/NOM_LOC asignados)
  2) localidades_puntos.geojson     (puntos de localidades + TODAS las columnas del CSV del Censo)
  3) localidades_poligonos.geojson  (polígonos de localidades = unión de colonias asignadas + columnas del Censo)

Además, si encuentra el diccionario de datos, genera:
  4) diccionario.json               (mapeo MNEMÓNICO -> {indicador, descripcion})

Requisitos:
  pip install pandas numpy shapely fiona

Ejecución:
  python generar_capas.py
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import fiona

from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree


# =========================
# CONFIG
# =========================
MUN = "Mineral de la Reforma"

CSV_CENSO = "conjunto_de_datos_iter_13CSV20.csv"
SHP_COLONIAS = "Fraccionamientos y Colonias.shp"

# (Opcional) diccionario de datos (tú lo compartiste como diccionario_datos_iter_13CSV20.csv)
CSV_DICCIONARIO = "diccionario_datos_iter_13CSV20.csv"

OUT_COLONIAS = "colonias.geojson"
OUT_LOC_PUNTOS = "localidades_puntos.geojson"
OUT_LOC_POLYS = "localidades_poligonos.geojson"
OUT_DICCIONARIO_JSON = "diccionario.json"


# =========================
# HELPERS
# =========================

# DMS -> Decimal: 98°41'45.725" W  /  20°04'20.537" N
DMS_RE = re.compile(r'^\s*(\d+)\s*°\s*(\d+)\s*\'\s*(\d+(?:\.\d+)?)\s*"\s*([NSEW])\s*$')

def dms_to_dd(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    m = DMS_RE.match(s)
    if not m:
        # intenta decimal directo
        try:
            return float(s)
        except Exception:
            return np.nan
    deg = int(m.group(1))
    minu = int(m.group(2))
    sec = float(m.group(3))
    hemi = m.group(4).upper()
    dd = deg + minu / 60.0 + sec / 3600.0
    return -dd if hemi in ("S", "W") else dd

def to_int_or_none(x):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if s in ("", "*", "NA", "N/A", "null", "None"):
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def py(v):
    """Convierte valores pandas/numpy a tipos JSON-friendly."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (int, float, str)):
        return v
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return str(v)

def haversine_m(lon1, lat1, lon2, lat2):
    """Distancia aprox en metros entre dos coords (WGS84)."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def nearest_index(tree: STRtree, query_geom, points_list=None):
    """
    Devuelve el índice del punto más cercano.
    - Preferido: STRtree.query_nearest (Shapely 2)
    - Fallback: STRtree.nearest + map WKB
    """
    if hasattr(tree, "query_nearest"):
        idxs = tree.query_nearest(query_geom)
        return int(np.atleast_1d(idxs)[0])

    if points_list is None:
        raise RuntimeError("points_list es requerido para fallback nearest()")
    wkb_to_idx = {p.wkb: i for i, p in enumerate(points_list)}
    g = tree.nearest(query_geom)
    return int(wkb_to_idx[g.wkb])


def generar_diccionario_json(diccionario_csv: Path, out_json: Path) -> bool:
    """
    Lee el CSV del diccionario (tiene encabezado real en la fila 5 del archivo)
    y genera diccionario.json: {MNEMONICO: {indicador, descripcion}}
    """
    if not diccionario_csv.exists():
        return False

    # El archivo trae 4 filas “título”, el header real está en la fila 5 (index 4)
    dd = pd.read_csv(diccionario_csv, header=4, encoding="utf-8", low_memory=False)

    # limpiar columnas basura
    dd = dd.loc[:, [c for c in dd.columns if not str(c).startswith("Unnamed")]]
    dd = dd.dropna(subset=["Mnemónico", "Indicador"], how="any").copy()

    # normalizar strings
    dd["Mnemónico"] = dd["Mnemónico"].astype(str).str.strip()
    dd["Indicador"] = dd["Indicador"].astype(str).str.strip()
    dd["Descripción"] = dd["Descripción"].astype(str).str.strip()

    mapping_dict = {}
    for _, r in dd.iterrows():
        mnem = r["Mnemónico"]
        if not mnem or mnem.lower() == "nan":
            continue
        mapping_dict[mnem] = {
            "indicador": r["Indicador"],
            "descripcion": r["Descripción"],
            "rangos": str(r.get("Rangos", "")).strip(),
            "longitud": str(r.get("Longitud", "")).strip(),
        }

    # Campos auxiliares del script (no vienen en el diccionario oficial)
    mapping_dict["lon_dd"] = {"indicador": "Longitud (decimal)", "descripcion": "Longitud convertida a grados decimales (WGS84)."}
    mapping_dict["lat_dd"] = {"indicador": "Latitud (decimal)", "descripcion": "Latitud convertida a grados decimales (WGS84)."}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=2)

    return True


# =========================
# MAIN
# =========================
def main():
    base = Path(".")
    csv_path = base / CSV_CENSO
    shp_path = base / SHP_COLONIAS
    dict_path = base / CSV_DICCIONARIO

    if not csv_path.exists():
        raise FileNotFoundError(f"No existe {CSV_CENSO} en la carpeta actual.")
    if not shp_path.exists():
        raise FileNotFoundError(f"No existe {SHP_COLONIAS} en la carpeta actual (asegúrate de tener .shp/.shx/.dbf/.prj/.cpg).")

    # 0) Diccionario (opcional)
    if generar_diccionario_json(dict_path, base / OUT_DICCIONARIO_JSON):
        print(f"✅ Diccionario generado: {OUT_DICCIONARIO_JSON}")
    else:
        print("ℹ️  No se generó diccionario.json (no se encontró diccionario_datos_iter_13CSV20.csv).")

    # 1) Cargar CSV y filtrar municipio
    df = pd.read_csv(csv_path, low_memory=False)

    df_m = df[df["NOM_MUN"].astype(str).str.strip() == MUN].copy()
    df_m = df_m[~df_m["NOM_LOC"].astype(str).str.contains("Total del Municipio", case=False, na=False)]
    df_m = df_m[~df_m["LOC"].isin([0, 9998, 9999])]
    df_m = df_m.dropna(subset=["LONGITUD", "LATITUD", "NOM_LOC"]).copy()

    # DMS -> decimal
    df_m["lon_dd"] = df_m["LONGITUD"].apply(dms_to_dd)
    df_m["lat_dd"] = df_m["LATITUD"].apply(dms_to_dd)
    df_m = df_m.dropna(subset=["lon_dd", "lat_dd"]).copy()

    df_m["LOC_int"] = df_m["LOC"].apply(to_int_or_none)

    if df_m.empty:
        raise RuntimeError("No se encontraron localidades válidas. Revisa NOM_MUN / LONGITUD / LATITUD.")

    csv_cols = list(df_m.columns)  # todas las columnas, incluyendo las de vivienda y auxiliares

    # 2) Cargar colonias (SHP)
    colonias_polys = []
    colonias_props = []
    with fiona.open(shp_path) as src:
        for feat in src:
            geom = feat.get("geometry")
            if not geom:
                continue
            colonias_polys.append(shape(geom))
            p = feat.get("properties", {}) or {}
            colonias_props.append({
                "Nombre": p.get("Nombre"),
                "Pres_Hab": p.get("Pres. Hab"),
                "Caracter": p.get("Caracter"),
                "AREA_APROX": to_int_or_none(p.get("AREA APROX")),
            })

    if not colonias_polys:
        raise RuntimeError("El SHP no contiene polígonos válidos.")

    # 3) Armar STRtree de puntos de localidades
    loc_points = [Point(xy) for xy in zip(df_m["lon_dd"].values, df_m["lat_dd"].values)]
    loc_tree = STRtree(loc_points)

    # 4) Asignar colonia -> localidad más cercana
    loc_to_polys = {}   # LOC -> list[Polygon]
    colonia_to_loc = [] # por índice de colonia: (LOC, NOM_LOC, dist_m)

    for i, poly in enumerate(colonias_polys):
        rp = poly.representative_point()
        j = nearest_index(loc_tree, rp, points_list=loc_points)
        loc_row = df_m.iloc[j]

        loc_id = to_int_or_none(loc_row["LOC_int"])
        nom_loc = str(loc_row["NOM_LOC"])
        dist = haversine_m(rp.x, rp.y, float(loc_row["lon_dd"]), float(loc_row["lat_dd"]))

        colonia_to_loc.append((loc_id, nom_loc, dist))
        loc_to_polys.setdefault(loc_id, []).append(poly)

    # 5) GeoJSON: Colonias
    colonias_fc = {
        "type": "FeatureCollection",
        "name": "colonias",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": []
    }

    for i, (poly, p) in enumerate(zip(colonias_polys, colonias_props)):
        loc_id, nom_loc, dist = colonia_to_loc[i]
        colonias_fc["features"].append({
            "type": "Feature",
            "properties": {
                **p,
                "LOC": loc_id,
                "NOM_LOC": nom_loc,
                "NOM_MUN": MUN,
                "dist_m_a_localidad": float(dist),
                "col_id": int(i + 1),
            },
            "geometry": mapping(poly)
        })

    # 6) GeoJSON: Localidades (puntos) con TODAS las columnas del CSV
    loc_points_fc = {
        "type": "FeatureCollection",
        "name": "localidades_puntos",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": []
    }

    for _, row in df_m.iterrows():
        props = {c: py(row[c]) for c in csv_cols}
        # Asegurar llaves
        props["LOC"] = to_int_or_none(row["LOC_int"])
        props["NOM_LOC"] = str(row["NOM_LOC"])
        props["NOM_MUN"] = MUN
        props["lon_dd"] = float(row["lon_dd"])
        props["lat_dd"] = float(row["lat_dd"])

        loc_points_fc["features"].append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [props["lon_dd"], props["lat_dd"]]}
        })

    # 7) GeoJSON: Localidades (polígonos) = unión de colonias asignadas
    loc_polys_fc = {
        "type": "FeatureCollection",
        "name": "localidades_poligonos_union_colonias",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": []
    }

    for loc_id, poly_list in loc_to_polys.items():
        union_geom = unary_union(poly_list)

        # propiedades del CSV para esa localidad
        sub = df_m[df_m["LOC_int"] == loc_id]
        if sub.empty:
            # debería ser raro; pero queda seguro
            base_props = {"LOC": loc_id, "NOM_LOC": f"LOC {loc_id}", "NOM_MUN": MUN}
        else:
            row = sub.iloc[0]
            base_props = {c: py(row[c]) for c in csv_cols}
            base_props["LOC"] = loc_id
            base_props["NOM_LOC"] = str(row["NOM_LOC"])
            base_props["NOM_MUN"] = MUN
            base_props["lon_dd"] = float(row["lon_dd"])
            base_props["lat_dd"] = float(row["lat_dd"])

        base_props["colonias_count"] = int(len(poly_list))
        base_props["metodo_poligono"] = "union_colonias_asignadas_nearest"

        loc_polys_fc["features"].append({
            "type": "Feature",
            "properties": base_props,
            "geometry": mapping(union_geom)
        })

    # 8) Guardar
    with open(base / OUT_COLONIAS, "w", encoding="utf-8") as f:
        json.dump(colonias_fc, f, ensure_ascii=False)

    with open(base / OUT_LOC_PUNTOS, "w", encoding="utf-8") as f:
        json.dump(loc_points_fc, f, ensure_ascii=False)

    with open(base / OUT_LOC_POLYS, "w", encoding="utf-8") as f:
        json.dump(loc_polys_fc, f, ensure_ascii=False)

    print("✅ Listo. Archivos generados:")
    print("  -", OUT_COLONIAS)
    print("  -", OUT_LOC_PUNTOS)
    print("  -", OUT_LOC_POLYS)
    print(f"Localidades (puntos): {len(loc_points_fc['features'])}")
    print(f"Colonias (polígonos): {len(colonias_fc['features'])}")
    print(f"Localidades (polígonos): {len(loc_polys_fc['features'])}")


if __name__ == "__main__":
    main()

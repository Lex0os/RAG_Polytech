#!/usr/bin/env python3
"""Extracteur structuré des PDF DEFI (CDEFI-CGE-CTI).

Le programme cible les exports PDF FPDF du questionnaire « Données des Écoles
Françaises d'Ingénieurs » (DEFI). Il conserve les en-têtes hiérarchiques des
colonnes, les libellés de ligne et surtout toutes les valeurs numériques, y
compris les zéros.

Sorties :
- SQLite (table ``defi_values``) ;
- CSV plat ;
- CSV de contrôle des lignes à vérifier.

Exemple :
    python defi_extractor.py ./mes_pdf --sqlite defi.sqlite --csv defi.csv

Dépendance : pdfplumber >= 0.11
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pdfplumber


ROMAN = r"[IVX]+"
CODE_WORD_RE = re.compile(rf"^{ROMAN}\.\d+(?:\.bis)?$", re.I)
MAJOR_RE = re.compile(rf"^({ROMAN})\.\s+(.+)$")
NUMBER_RE = re.compile(r"[-+]?\d+(?:[ .]\d{{3}})*(?:[,.]\d+)?")
YES_NO_MARKERS = {"oui", "non", "x", "public", "privé", "prive", "aucune", "aucun"}
HEADER_HINTS = {
    "total", "hommes", "femmes", "nombre", "valeurs", "pays", "recherche",
    "enseignement", "gestion directe", "gestion en partenariat", "etp",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = (s.replace("\u00ad", "")
           .replace("\ufb01", "fi")
           .replace("\ufb02", "fl")
           .replace("\ufffe", "")
           .replace("\r", "\n"))
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("Universit -", "Université -").replace("é Prépa", "Prépa")
    return s


def normalize_code(value: str) -> str:
    s = clean_text(value).replace(" ", "")
    s = re.sub(r"(?i)([IVX]+\.\d+)et1bis$", r"\1_et_1bis", s)
    s = re.sub(r"(?i)([IVX]+\.\d+)bis$", r"\1.bis", s)
    s = s.replace("..", ".")
    return s


def numeric_token(value: str) -> str | None:
    s = clean_text(value)
    if not s or "/" in s or "@" in s or "://" in s:
        return None
    m = re.fullmatch(r"[-+]?\d+(?:[ .]\d{3})*(?:[,.]\d+)?\s*%?", s)
    return s if m else None


def parse_number(value: str) -> float | None:
    s = clean_text(value)
    token = numeric_token(s)
    if token is None:
        # Une valeur commentée telle que « 127 (77%...) » garde son premier nombre.
        m = re.match(r"^\s*([-+]?\d+(?:[,.]\d+)?)\b", s)
        if not m:
            return None
        token = m.group(1)
    token = token.replace("%", "").replace(" ", "").replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def all_numbers(value: str) -> list[float]:
    out: list[float] = []
    for raw in re.findall(r"[-+]?\d+(?:[,.]\d+)?", clean_text(value)):
        try:
            out.append(float(raw.replace(",", ".")))
        except ValueError:
            pass
    return out


def value_type(value: str) -> str:
    s = clean_text(value)
    if not s:
        return "blank"
    if numeric_token(s) is not None:
        return "number"
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
        return "date"
    if s.casefold() in YES_NO_MARKERS:
        return "boolean_or_category"
    if parse_number(s) is not None:
        return "number_with_text"
    return "text"


def is_value_marker(value: str) -> bool:
    s = clean_text(value)
    return bool(s) and (numeric_token(s) is not None or s.casefold() in YES_NO_MARKERS)


def dedupe_parts(parts: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        p = clean_text(part)
        if p and p.casefold() not in seen:
            seen.add(p.casefold())
            result.append(p)
    return result


@dataclass
class Record:
    source_file: str
    source_page: int
    major_section_code: str = ""
    major_section_title: str = ""
    subsection_title: str = ""
    section_code: str = ""
    section_title: str = ""
    table_title: str = ""
    row_group: str = ""
    row_name: str = ""
    column_group: str = ""
    column_name: str = ""
    column_path: str = ""
    column_headers_json: str = "[]"
    value_text: str = ""
    value_num: float | None = None
    numbers_json: str = "[]"
    value_type: str = "text"
    extraction_method: str = ""
    confidence: float = 0.0
    raw_context: str = ""


@dataclass
class SectionRegion:
    code: str
    title: str
    y0: float
    y1: float
    subsection_title: str = ""


def border_only(obj: dict[str, Any]) -> bool:
    """Conserve le texte et seulement les rectangles servant de bordures.

    Les PDF DEFI dessinent également les glyphes et les aplats sous forme de
    rectangles. Les exclure avant ``find_tables`` évite que ces objets soient
    pris pour des lignes de tableau.
    """
    if obj.get("object_type") != "rect":
        return True
    w = float(obj.get("width", 0.0))
    h = float(obj.get("height", 0.0))
    color = obj.get("non_stroking_color") or obj.get("stroking_color")
    gray = (
        isinstance(color, (tuple, list)) and len(color) >= 3
        and max(color[:3]) - min(color[:3]) < 0.015
        and 0.52 <= float(color[0]) <= 0.96
    )
    return bool(gray and ((w <= 1.15 and h >= 3.0) or (h <= 1.15 and w >= 3.0)))


TABLE_SETTINGS = {
    "snap_tolerance": 1,
    "join_tolerance": 1,
    "intersection_tolerance": 1,
    "edge_min_length": 3,
}


def filtered_tables(page: pdfplumber.page.Page) -> list[Any]:
    fp = page.filter(border_only)
    result = []
    for table in fp.find_tables(table_settings=TABLE_SETTINGS):
        x0, y0, x1, y1 = map(float, table.bbox)
        if x0 < -1 or y0 < -1 or x1 > page.width + 1 or y1 > page.height + 1:
            continue
        if len(table.cells) < 2 or x1 - x0 < 8 or y1 - y0 < 8:
            continue
        result.append(table)
    return sorted(result, key=lambda t: (float(t.bbox[1]), float(t.bbox[0]), -float(t.bbox[2] - t.bbox[0])))


def table_matrix(table: Any) -> list[list[str]]:
    raw = table.extract() or []
    ncols = max((len(r) for r in raw), default=0)
    return [[clean_text(r[c] if c < len(r) else "") for c in range(ncols)] for r in raw]


def find_major_heading(page: pdfplumber.page.Page) -> tuple[str, str] | None:
    lines = page.extract_text_lines(layout=False, strip=True, return_chars=False) or []
    for line in lines:
        text = clean_text(line.get("text", ""))
        m = MAJOR_RE.match(text)
        if m and text.upper() == text and len(text) > 8:
            return m.group(1), clean_text(m.group(2))
    return None


def find_subsection_heading(page: pdfplumber.page.Page) -> str:
    lines = page.extract_text_lines(layout=False, strip=True, return_chars=False) or []
    candidates: list[str] = []
    for line in lines:
        y = float(line.get("top", 0))
        x = float(line.get("x0", 0))
        text = clean_text(line.get("text", ""))
        if 40 <= y <= 140 and x < 80 and len(text) > 6:
            if re.match(r"^[A-Z]\.", text) or text in {"Formation", "Gouvernance et vie des personnels"}:
                candidates.append(text)
    return candidates[-1] if candidates else ""


def is_top_level_table(table: Any, matrix: Sequence[Sequence[str]]) -> bool:
    x0, _, x1, _ = map(float, table.bbox)
    if x0 > 100 or x1 - x0 < 700 or not matrix:
        return False
    if max((len(r) for r in matrix), default=0) != 3:
        return False
    first_cells = [normalize_code(r[0]) if r else "" for r in matrix]
    return any(re.match(rf"^{ROMAN}\.\d+", c, re.I) for c in first_cells)


def detect_sections(page: pdfplumber.page.Page, tables: Sequence[Any]) -> tuple[list[SectionRegion], set[int]]:
    sections: list[SectionRegion] = []
    top_ids: set[int] = set()
    for idx, table in enumerate(tables):
        matrix = table_matrix(table)
        if not is_top_level_table(table, matrix):
            continue
        top_ids.add(idx)
        active_code = ""
        active_title = ""
        for ridx, row_obj in enumerate(table.rows):
            row = matrix[ridx] if ridx < len(matrix) else []
            code_cell = clean_text(row[0] if row else "")
            title_cell = clean_text(row[1] if len(row) > 1 else "")
            code = normalize_code(code_cell)
            if re.match(rf"^{ROMAN}\.\d+", code, re.I):
                active_code = code
                active_title = title_cell
                sub = ""
            elif active_code:
                code = active_code
                sub = title_cell
                title_cell = active_title
            else:
                continue
            # La cellule du code peut être fusionnée verticalement (IX.1 et 1bis).
            # La cellule du titre donne alors les vraies limites de la sous-ligne.
            title_bbox = row_obj.cells[1] if len(row_obj.cells) > 1 else None
            bbox = title_bbox or row_obj.bbox
            y0, y1 = float(bbox[1]), float(bbox[3])
            sections.append(SectionRegion(code, title_cell, y0, y1, sub))

    # Fallback pour les pages dont la grande cellule de gauche se poursuit
    # jusqu'au bord de page et n'est donc pas reconnue comme tableau fermé.
    if not sections:
        words = page.extract_words(x_tolerance=1, y_tolerance=2, keep_blank_chars=False) or []
        code_words = [w for w in words if float(w["x0"]) < 55 and CODE_WORD_RE.match(clean_text(w["text"]))]
        if code_words:
            code = normalize_code(code_words[0]["text"])
            title_words = [
                w for w in words
                if 50 <= (float(w["x0"]) + float(w["x1"])) / 2 < 199
                and 40 <= (float(w["top"]) + float(w["bottom"])) / 2 <= page.height - 10
            ]
            title = words_to_text(title_words)
            sections.append(SectionRegion(code, title, 40.0, page.height - 8.0))
    return sorted(sections, key=lambda s: s.y0), top_ids


def words_to_text(words: Sequence[dict[str, Any]], tolerance: float = 2.5) -> str:
    if not words:
        return ""
    groups: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda w: (float(w["top"]), float(w["x0"]))):
        if not groups:
            groups.append([word])
            continue
        avg = sum(float(w["top"]) for w in groups[-1]) / len(groups[-1])
        if abs(float(word["top"]) - avg) <= tolerance:
            groups[-1].append(word)
        else:
            groups.append([word])
    return clean_text(" ".join(" ".join(str(w["text"]) for w in sorted(g, key=lambda z: float(z["x0"]))) for g in groups))


def header_paths(matrix: Sequence[Sequence[str]], header_rows: Sequence[int], data_start: int) -> list[list[str]]:
    ncols = max((len(r) for r in matrix), default=0)
    paths: list[list[str]] = [[] for _ in range(ncols)]
    previous_paths: list[tuple[str, ...]] = [tuple() for _ in range(ncols)]
    for ridx in header_rows:
        row = list(matrix[ridx]) + [""] * (ncols - len(matrix[ridx]))
        current = ""
        filled = [""] * ncols
        for c in range(data_start, ncols):
            cell = clean_text(row[c])
            if cell:
                # Dans DEFI, Total est une cellule sœur du bloc H/F plutôt
                # qu'un enfant fusionné de son en-tête de période.
                if (cell.casefold() == "total" and current
                        and re.search(r"\b(mois|promotion|ann[eé]e)\b", current, re.I)):
                    if current.casefold() not in {p.casefold() for p in paths[c]}:
                        paths[c].append(current)
                current = cell
                filled[c] = cell
            elif c > data_start and previous_paths[c] == previous_paths[c - 1]:
                filled[c] = current
            else:
                current = ""
        for c in range(data_start, ncols):
            if filled[c] and filled[c].casefold() not in {p.casefold() for p in paths[c]}:
                paths[c].append(filled[c])
        previous_paths = [tuple(p) for p in paths]
    return paths


def record_base(source: Path, page_no: int, major: tuple[str, str], subsection: str, section: SectionRegion) -> dict[str, Any]:
    return {
        "source_file": source.name,
        "source_page": page_no,
        "major_section_code": major[0],
        "major_section_title": major[1],
        "subsection_title": section.subsection_title or subsection,
        "section_code": section.code,
        "section_title": clean_text(section.title),
    }


def make_record(
    base: dict[str, Any], *, table_title: str = "", row_group: str = "", row_name: str = "",
    headers: Sequence[str] = ("Valeur",), value: str = "", method: str, confidence: float,
    raw_context: dict[str, Any] | None = None,
) -> Record:
    hp = dedupe_parts(headers) or ["Valeur"]
    v = clean_text(value)
    return Record(
        **base,
        table_title=clean_text(table_title),
        row_group=clean_text(row_group),
        row_name=clean_text(row_name),
        column_group=" > ".join(hp[:-1]),
        column_name=hp[-1],
        column_path=" > ".join(hp),
        column_headers_json=json.dumps(hp, ensure_ascii=False),
        value_text=v,
        value_num=parse_number(v),
        numbers_json=json.dumps(all_numbers(v), ensure_ascii=False),
        value_type=value_type(v),
        extraction_method=method,
        confidence=confidence,
        raw_context=json.dumps(raw_context or {}, ensure_ascii=False),
    )



CHAIR_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
CHAIR_DATE_RE = re.compile(
    r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-]((?:19|20)\d{2})\b"
)
CHAIR_YEAR_RANGE_RE = re.compile(
    r"\b((?:19|20)\d{2})\s*"
    r"(?:[-–—/]|à|a|au|jusqu['’]?à)\s*"
    r"((?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)


def chair_header_key(value: str) -> str:
    s = clean_text(value).casefold()
    s = (
        s.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ç", "c")
        .replace("’", "'")
    )
    return re.sub(r"\s+", " ", s).strip()


def is_chairs_table(matrix: Sequence[Sequence[str]]) -> bool:
    if not matrix:
        return False
    first_rows = " ".join(
        clean_text(cell)
        for row in matrix[:2]
        for cell in row
    ).casefold()
    return (
        "nom de la chaire" in first_rows
        and "debut" in chair_header_key(first_rows)
        and "fin" in chair_header_key(first_rows)
    )


def normalize_chair_date(
    day: str,
    month: str,
    year: str,
) -> str | None:
    """Valide une date du PDF et la convertit en ISO YYYY-MM-DD."""
    try:
        parsed = date(
            int(year),
            int(month),
            int(day),
        )
    except ValueError:
        return None
    return parsed.isoformat()


def chair_period_result(
    *,
    raw: str,
    status: str,
    confidence: float,
    start_year: int | None = None,
    end_year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    note: str = "",
) -> dict[str, str | int | float | None]:
    return {
        "raw": raw or None,
        "start_year": start_year,
        "end_year": end_year,
        "start_date": start_date,
        "end_date": end_date,
        "status": status,
        "confidence": confidence,
        "note": note,
    }


def parse_chair_period(
    value: str,
) -> dict[str, str | int | float | None]:
    """Conserve le texte exact et ajoute seulement une normalisation auditée."""
    raw = clean_text(value)

    if not raw:
        return chair_period_result(
            raw="",
            status="missing",
            confidence=0.0,
            note="Cellule de période vide.",
        )

    folded = chair_header_key(raw)
    valid_dates: list[tuple[str, int]] = []

    for match in CHAIR_DATE_RE.finditer(raw):
        iso_date = normalize_chair_date(
            match.group(1),
            match.group(2),
            match.group(3),
        )
        if iso_date is not None:
            valid_dates.append(
                (iso_date, int(match.group(3)))
            )

    if len(valid_dates) >= 2:
        start_date, start_year = valid_dates[0]
        end_date, end_year = valid_dates[1]

        if start_date > end_date:
            return chair_period_result(
                raw=raw,
                status="inconsistent_range",
                confidence=0.15,
                start_year=start_year,
                end_year=end_year,
                start_date=start_date,
                end_date=end_date,
                note="La date de début est postérieure à la date de fin.",
            )

        return chair_period_result(
            raw=raw,
            status="parsed_date_range",
            confidence=0.995,
            start_year=start_year,
            end_year=end_year,
            start_date=start_date,
            end_date=end_date,
            note="Deux dates exactes ont été reconnues.",
        )

    if len(valid_dates) == 1:
        parsed_date, parsed_year = valid_dates[0]

        if re.search(
            r"\b(fin|jusqu|termine|echeance|cloture)\b",
            folded,
        ):
            return chair_period_result(
                raw=raw,
                status="open_ended_end_date",
                confidence=0.93,
                end_year=parsed_year,
                end_date=parsed_date,
                note="Une date de fin explicite a été reconnue.",
            )

        if re.search(
            r"\b(debut|depuis|commence|lancee|creation)\b",
            folded,
        ):
            return chair_period_result(
                raw=raw,
                status="open_ended_start_date",
                confidence=0.93,
                start_year=parsed_year,
                start_date=parsed_date,
                note="Une date de début explicite a été reconnue.",
            )

        return chair_period_result(
            raw=raw,
            status="ambiguous_single_date",
            confidence=0.40,
            note=(
                "Une date unique est présente sans indication fiable "
                "de début ou de fin."
            ),
        )

    range_match = CHAIR_YEAR_RANGE_RE.search(raw)
    if range_match:
        start_year = int(range_match.group(1))
        end_year = int(range_match.group(2))

        if start_year > end_year:
            return chair_period_result(
                raw=raw,
                status="inconsistent_range",
                confidence=0.15,
                start_year=start_year,
                end_year=end_year,
                note="L'année de début est postérieure à l'année de fin.",
            )

        return chair_period_result(
            raw=raw,
            status="parsed_year_range",
            confidence=0.98,
            start_year=start_year,
            end_year=end_year,
            note="Une plage explicite de deux années a été reconnue.",
        )

    years = [
        int(year)
        for year in CHAIR_YEAR_RE.findall(raw)
    ]

    if len(years) >= 2:
        start_year = years[0]
        end_year = years[1]

        if start_year > end_year:
            return chair_period_result(
                raw=raw,
                status="inconsistent_range",
                confidence=0.15,
                start_year=start_year,
                end_year=end_year,
                note="Les années sont dans un ordre incohérent.",
            )

        return chair_period_result(
            raw=raw,
            status="parsed_year_range",
            confidence=0.92,
            start_year=start_year,
            end_year=end_year,
            note=(
                "Deux années ont été reconnues sans séparateur "
                "de plage standard."
            ),
        )

    if len(years) == 1:
        year = years[0]

        if re.search(
            r"\b(fin|jusqu|termine|echeance|cloture)\b",
            folded,
        ):
            return chair_period_result(
                raw=raw,
                status="open_ended_end_year",
                confidence=0.86,
                end_year=year,
                note="Une année de fin explicite a été reconnue.",
            )

        if re.search(
            r"\b(debut|depuis|commence|lancee|creation)\b",
            folded,
        ):
            return chair_period_result(
                raw=raw,
                status="open_ended_start_year",
                confidence=0.86,
                start_year=year,
                note="Une année de début explicite a été reconnue.",
            )

        return chair_period_result(
            raw=raw,
            status="ambiguous_single_year",
            confidence=0.35,
            note=(
                "Une année unique est présente sans indication fiable "
                "de début ou de fin."
            ),
        )

    return chair_period_result(
        raw=raw,
        status="invalid_period",
        confidence=0.0,
        note=(
            "Le texte brut est conservé mais aucune date ou année "
            "exploitable n'a été reconnue."
        ),
    )

def chair_record(
    base: dict[str, Any],
    *,
    table: Any,
    row_index: int,
    column_index: int,
    chair_name: str,
    header: str,
    value: str,
    confidence: float = 0.995,
    force_text: bool = False,
) -> Record:
    record = make_record(
        base,
        row_name=chair_name,
        headers=[header],
        value=value,
        method="pdfplumber_defi_chair_table",
        confidence=confidence,
        raw_context={
            "table_bbox": list(map(float, table.bbox)),
            "row": row_index,
            "column": column_index,
            "special_parser": "defi_v_1_chairs",
        },
    )
    if force_text:
        record.value_num = None
        record.value_type = "text" if clean_text(value) else "blank"
    return record


def parse_chairs_table(
    source: Path,
    page_no: int,
    major: tuple[str, str],
    subsection: str,
    section: SectionRegion,
    table: Any,
    include_empty: bool,
) -> list[Record]:
    """Parse V.1 avec des colonnes temporelles normalisées.

    Le PDF fournit une seule cellule « Années de début et de fin ». Le parseur
    générique gardait seulement le premier nombre de cette cellule. Ici, la
    période est séparée avant toute conversion numérique.
    """
    matrix = table_matrix(table)
    if not is_chairs_table(matrix):
        return []

    ncols = max((len(row) for row in matrix), default=0)
    matrix = [list(row) + [""] * (ncols - len(row)) for row in matrix]
    base = record_base(source, page_no, major, subsection, section)

    first_header = [chair_header_key(cell) for cell in matrix[0]]

    def find_column(*needles: str) -> int | None:
        for index, header in enumerate(first_header):
            if all(needle in header for needle in needles):
                return index
        return None

    name_col = find_column("nom", "chaire")
    implication_col = find_column("implication")
    partners_col = find_column("noms", "partenaires")
    institutions_col = find_column("enseignement superieur", "partenaires")
    period_col = find_column("debut", "fin", "chaire")
    funding_col = find_column("dotation", "chaires")
    domain_col = find_column("domaine", "recherche")

    if name_col is None or period_col is None:
        return []

    research_col: int | None = None
    teaching_col: int | None = None
    if len(matrix) > 1:
        for index, value in enumerate(matrix[1]):
            key = chair_header_key(value)
            if key == "recherche":
                research_col = index
            elif key == "enseignement":
                teaching_col = index

    if research_col is None:
        research_col = domain_col
    if teaching_col is None and domain_col is not None and domain_col + 1 < ncols:
        teaching_col = domain_col + 1

    records: list[Record] = []

    for row_index, row in enumerate(matrix[1:], start=1):
        chair_name = clean_text(row[name_col])
        if not chair_name:
            continue

        # Ignore la sous-ligne d'en-tête Recherche / Enseignement.
        if chair_header_key(chair_name) in {"recherche", "enseignement"}:
            continue

        def add_simple(column: int | None, header: str, confidence: float = 0.995) -> None:
            if column is None or column >= len(row):
                return
            value = clean_text(row[column])
            if not value and not include_empty:
                return
            records.append(
                chair_record(
                    base,
                    table=table,
                    row_index=row_index,
                    column_index=column,
                    chair_name=chair_name,
                    header=header,
                    value=value,
                    confidence=confidence if value else 0.75,
                )
            )

        add_simple(implication_col, "% d'implication de l'école")
        add_simple(partners_col, "Noms des partenaires")
        add_simple(
            institutions_col,
            "Établissements d'enseignement supérieur partenaires",
        )
        add_simple(funding_col, "Dotation globale de l'ensemble des chaires")
        add_simple(research_col, "Domaine de recherche/enseignement > Recherche")
        add_simple(teaching_col, "Domaine de recherche/enseignement > Enseignement")

        raw_period = clean_text(row[period_col])
        period = parse_chair_period(raw_period)

        # Le texte brut reste disponible pour l'audit, mais les années sont
        # désormais de vraies dimensions séparées.
        if raw_period or include_empty:
            records.append(
                chair_record(
                    base,
                    table=table,
                    row_index=row_index,
                    column_index=period_col,
                    chair_name=chair_name,
                    header="Période de la chaire - texte brut",
                    value=raw_period,
                    confidence=0.995 if raw_period else 0.75,
                    force_text=True,
                )
            )

        # Les champs suivants sont des dérivations auditables. Ils ne
        # remplacent jamais la cellule brute et ne filtrent aucune chaire.
        temporal_fields = (
            ("Année de début de la chaire", period["start_year"], False),
            ("Année de fin de la chaire", period["end_year"], False),
            ("Date de début de la chaire", period["start_date"], True),
            ("Date de fin de la chaire", period["end_date"], True),
            ("Statut d'analyse de la période", period["status"], True),
            (
                "Confiance d'analyse de la période",
                period["confidence"],
                False,
            ),
            ("Note d'analyse de la période", period["note"], True),
        )

        for header, temporal_value, force_text in temporal_fields:
            always_keep = header in {
                "Statut d'analyse de la période",
                "Confiance d'analyse de la période",
                "Note d'analyse de la période",
            }

            if (
                temporal_value is None
                and not include_empty
                and not always_keep
            ):
                continue

            records.append(
                chair_record(
                    base,
                    table=table,
                    row_index=row_index,
                    column_index=period_col,
                    chair_name=chair_name,
                    header=header,
                    value=(
                        ""
                        if temporal_value is None
                        else str(temporal_value)
                    ),
                    confidence=(
                        float(period["confidence"])
                        if always_keep
                        else max(
                            0.50,
                            float(period["confidence"]),
                        )
                    ),
                    force_text=force_text,
                )
            )

    return records

def is_group_label(text: str) -> bool:
    s = clean_text(text)
    low = s.casefold()
    if not s or re.fullmatch(r"(?:[1-5]A\s*PB|TB|B|AB|Sans|N\.?C|Hommes|Femmes)", s, re.I):
        return False
    return any(k in low for k in (
        "sous statut", "voie de recrutement", "stagiaires de la formation continue",
        "diplôme", "cycle préparatoire", "cycle ingénieur", "ingénieur en année",
        "totaux", "total (", "baccalauréats",
    ))


def parse_generic_table(
    source: Path, page_no: int, major: tuple[str, str], subsection: str,
    section: SectionRegion, table: Any, include_empty: bool,
) -> list[Record]:
    matrix = table_matrix(table)
    if not matrix:
        return []
    ncols = max(len(r) for r in matrix)
    matrix = [list(r) + [""] * (ncols - len(r)) for r in matrix]

    # Certains mini-tableaux DEFI placent directement un couple « libellé :
    # valeur » dans chaque cellule, sans ligne d'en-tête ni première colonne
    # dédiée aux libellés. C'est notamment le cas du récapitulatif I.1 :
    # « Femmes : 341 » et « Total : 736 ». Sans ce traitement, ces valeurs
    # sont prises pour des en-têtes et ne sont jamais enregistrées.
    nonempty_cells: list[tuple[int, int, str]] = [
        (r, c, clean_text(cell))
        for r, row in enumerate(matrix)
        for c, cell in enumerate(row)
        if clean_text(cell)
    ]
    inline_pairs: list[tuple[int, int, str, str]] = []
    for r, c, cell in nonempty_cells:
        match = re.match(r"^(.+?)\s*:\s*(.+)$", cell)
        if not match:
            continue
        label = clean_text(match.group(1))
        value = clean_text(match.group(2))
        if label and value and (is_value_marker(value) or parse_number(value) is not None):
            inline_pairs.append((r, c, label, value))

    labels = {label.casefold() for _, _, label, _ in inline_pairs}
    is_i1_summary = (
        section.code == "I.1"
        and bool(labels)
        and labels.issubset({"hommes", "femmes", "total"})
    )
    if inline_pairs and len(inline_pairs) == len(nonempty_cells) and (len(inline_pairs) >= 2 or is_i1_summary):
        base = record_base(source, page_no, major, subsection, section)
        table_title = "Total (Cycle préparatoire + Cycle ingénieur)" if is_i1_summary else ""
        return [
            make_record(
                base,
                table_title=table_title,
                row_group=table_title,
                row_name=label,
                headers=["Valeur"],
                value=value,
                method="pdfplumber_defi_inline_pair_table",
                confidence=0.98,
                raw_context={
                    "table_bbox": list(map(float, table.bbox)),
                    "row": r,
                    "column": c,
                },
            )
            for r, c, label, value in inline_pairs
        ]

    marker_rows = [r for r, row in enumerate(matrix) if any(is_value_marker(c) for c in row)]
    first_data: int | None = min(marker_rows) if marker_rows else None
    if first_data is None:
        # Table textuelle : la première ligne est habituellement l'en-tête.
        for r in range(1, len(matrix)):
            if len([c for c in matrix[r] if c]) >= 2 and matrix[r][0]:
                first_data = r
                break
    if first_data is None:
        return []

    candidate = matrix[first_data]
    leading_groups = []
    for r in range(first_data):
        nonempty = [clean_text(c) for c in matrix[r] if clean_text(c)]
        if len(nonempty) == 1 and is_group_label(nonempty[0]):
            leading_groups.append(nonempty[0])
    column_header_rows = [r for r in range(first_data) if len([c for c in matrix[r] if c]) >= 2]

    first_cell = clean_text(candidate[0])
    col0_has_header = any(clean_text(matrix[r][0]) for r in column_header_rows)
    if is_value_marker(first_cell) and col0_has_header:
        data_start = 0
    elif first_cell:
        data_start = 1 if ncols > 1 else 0
    else:
        nonempty_positions = [i for i, c in enumerate(candidate) if clean_text(c)]
        data_start = min(nonempty_positions) if nonempty_positions else 0

    paths = header_paths(matrix, column_header_rows, data_start)
    base = record_base(source, page_no, major, subsection, section)
    table_title = " | ".join(dedupe_parts(leading_groups))
    current_group = leading_groups[-1] if leading_groups else ""
    records: list[Record] = []

    for r in range(first_data, len(matrix)):
        row = matrix[r]
        nonempty = [clean_text(c) for c in row if clean_text(c)]
        if not nonempty:
            continue
        has_marker = any(is_value_marker(c) for c in row)
        text_data = not marker_rows and len(nonempty) >= 2 and bool(clean_text(row[0]))
        if not has_marker and not text_data:
            if len(nonempty) == 1 and is_group_label(nonempty[0]):
                current_group = nonempty[0]
                if not table_title:
                    table_title = current_group
            continue

        row_name = clean_text(row[0]) if data_start > 0 else ""
        merged_targets = set()
        for c in range(data_start, ncols):
            if c in merged_targets:
                continue
            value = clean_text(row[c])
            if not value and not include_empty:
                continue
            semantic_column = c
            cell = table.rows[r].cells[c]
            if value and cell:
                # Une valeur fusionnée sur tout un bloc H/F/Total est le
                # total du bloc, pas la valeur du premier sous-groupe H.
                targets = []
                for target in range(c + 1, ncols):
                    if (not paths[target] or paths[target][-1].casefold() != "total"
                            or clean_text(row[target]) or table.rows[r].cells[target] is not None):
                        continue
                    if paths[target][:-1] and paths[target][:-1] != paths[c][:-1]:
                        continue
                    for header_row in column_header_rows:
                        header_cell = table.rows[header_row].cells[target]
                        if (header_cell and cell[0] <= header_cell[0] + 0.5
                                and header_cell[2] <= cell[2] + 0.5):
                            targets.append(target)
                            break
                if len(targets) == 1:
                    semantic_column = targets[0]
                    merged_targets.add(semantic_column)
            hp = dedupe_parts(paths[semantic_column] if semantic_column < len(paths) else [])
            if not hp:
                if data_start == 1 and ncols == 2:
                    hp = ["Valeur"]
                else:
                    hp = [f"Colonne {c + 1 - data_start}"]
            confidence = 0.97 if hp[0] != "Colonne 1" else 0.82
            records.append(make_record(
                base,
                table_title=table_title,
                row_group=current_group,
                row_name=row_name,
                headers=hp,
                value=value,
                method="pdfplumber_defi_table",
                confidence=confidence if value else 0.76,
                raw_context={"table_bbox": list(map(float, table.bbox)), "row": r, "column": c,
                             "semantic_column": semantic_column},
            ))
    return records


def group_content_lines(
    page: pdfplumber.page.Page, y0: float, y1: float, excluded_bboxes: Sequence[tuple[float, float, float, float]],
) -> list[str]:
    words = page.extract_words(x_tolerance=1, y_tolerance=2, keep_blank_chars=False) or []
    kept: list[dict[str, Any]] = []
    for w in words:
        cx = (float(w["x0"]) + float(w["x1"])) / 2
        cy = (float(w["top"]) + float(w["bottom"])) / 2
        if not (199 <= cx <= page.width - 8 and y0 <= cy <= y1):
            continue
        if any(x0 - .5 <= cx <= x1 + .5 and t - .5 <= cy <= b + .5 for x0, t, x1, b in excluded_bboxes):
            continue
        kept.append(w)
    if not kept:
        return []
    groups: list[list[dict[str, Any]]] = []
    for w in sorted(kept, key=lambda z: (float(z["top"]), float(z["x0"]))):
        if not groups:
            groups.append([w])
            continue
        avg = sum(float(x["top"]) for x in groups[-1]) / len(groups[-1])
        if abs(float(w["top"]) - avg) <= 2.6:
            groups[-1].append(w)
        else:
            groups.append([w])
    return [clean_text(" ".join(str(w["text"]) for w in sorted(g, key=lambda z: float(z["x0"])))) for g in groups]


def parse_scalar_content(
    source: Path, page_no: int, page: pdfplumber.page.Page, major: tuple[str, str], subsection: str,
    section: SectionRegion, excluded_bboxes: Sequence[tuple[float, float, float, float]],
    only_explicit: bool, include_empty: bool,
) -> list[Record]:
    lines = [x for x in group_content_lines(page, section.y0, section.y1, excluded_bboxes) if x and x not in {"/", "-"}]
    if not lines:
        return []
    base = record_base(source, page_no, major, subsection, section)
    records: list[Record] = []
    used: set[int] = set()

    # Libellé : valeur sur la même ligne.
    for i, line in enumerate(lines):
        m = re.match(r"^(.{2,}?)\s*:\s*(.+)$", line)
        if m and clean_text(m.group(2)):
            records.append(make_record(
                base, row_name=m.group(1), value=m.group(2), headers=["Valeur"],
                method="pdfplumber_defi_scalar_pair", confidence=0.94,
                raw_context={"line": i, "region": [section.y0, section.y1]},
            ))
            used.add(i)

    if only_explicit:
        # Les valeurs commentées commençant par un nombre (ex. V.6).
        for i, line in enumerate(lines):
            if i not in used and re.match(r"^\s*\d", line):
                records.append(make_record(
                    base, value=line, headers=["Valeur"], method="pdfplumber_defi_scalar",
                    confidence=0.88, raw_context={"line": i},
                ))
        return records

    # Couple un bloc de libellé (éventuellement sur plusieurs lignes) avec
    # la valeur qui le suit.
    pending: list[tuple[int, str]] = []
    unlabeled_index = 0
    for i, line in enumerate(lines):
        if i in used:
            continue
        scalar_value = is_value_marker(line) or parse_number(line) is not None or bool(re.match(r"^\s*\d", line))
        if scalar_value:
            label = clean_text(" ".join(text.rstrip(" :") for _, text in pending))
            for idx, _ in pending:
                used.add(idx)
            pending.clear()
            if not label:
                unlabeled_index += 1
                row_name = str(unlabeled_index) if sum(1 for x in lines if is_value_marker(x)) > 1 else ""
            else:
                row_name = label
            records.append(make_record(
                base, row_name=row_name, value=line, headers=["Valeur"],
                method="pdfplumber_defi_scalar_pair" if label else "pdfplumber_defi_scalar",
                confidence=0.91 if label else 0.86, raw_context={"line": i},
            ))
            used.add(i)
        else:
            pending.append((i, line))

    remaining = [line for i, line in enumerate(lines) if i not in used and line.strip(": ")]
    # Ignore les libellés vides « ETP: » sauf demande explicite des cellules vides.
    if include_empty:
        for line in remaining:
            if line.endswith(":"):
                records.append(make_record(
                    base, row_name=line.rstrip(": "), value="", headers=["Valeur"],
                    method="pdfplumber_defi_scalar_empty", confidence=0.72,
                ))
    remaining = [x for x in remaining if not x.endswith(":")]
    if remaining:
        if len(remaining) == 1:
            records.append(make_record(
                base, value=remaining[0], headers=["Valeur"], method="pdfplumber_defi_scalar",
                confidence=0.88, raw_context={"region": [section.y0, section.y1]},
            ))
        else:
            for idx, text in enumerate(remaining, start=1):
                records.append(make_record(
                    base, row_name=str(idx), value=text, headers=["Valeur"],
                    method="pdfplumber_defi_scalar_list", confidence=0.84,
                    raw_context={"item": idx},
                ))
    return records


def special_xi_section(matrix: Sequence[Sequence[str]], page_no: int) -> SectionRegion | None:
    if not matrix or not matrix[0]:
        return None
    first = clean_text(matrix[0][0])
    flat = clean_text(" ".join(c for row in matrix[:3] for c in row))
    if "Totaux des" in first:
        return SectionRegion("XI.Q1.TOTAUX", "Totaux des inscrits pour l'obtention du diplôme d'ingénieur", 0, 0)
    if first.casefold().startswith("type de bac"):
        code = "XI.Q2.BAC_BAC1" if "Bac et" in first or "bac+1" in first.casefold() else "XI.Q2.BAC2_PLUS"
        return SectionRegion(code, "Type de baccalauréat des primo-entrants", 0, 0)
    if re.search(r"\bBAC\b", first, re.I):
        level = re.sub(r"\s+", "", first.upper()).replace("AUTRE(VAE)", "VAE")
        return SectionRegion(f"XI.Q1.{level}", "Niveau académique des primo-entrants", 0, 0)
    if "Parmi les bacheliers" in flat:
        return SectionRegion("XI.Q2.COMPLEMENT", "Questions complémentaires sur le baccalauréat", 0, 0)
    return None


def parse_metadata(source: Path, page_no: int, page: pdfplumber.page.Page, tables: Sequence[Any]) -> list[Record]:
    major = ("META", "Métadonnées DEFI")
    subsection = ""
    records: list[Record] = []
    text = clean_text(page.extract_text() or "")
    if page_no == 1:
        lines = [clean_text(x) for x in (page.extract_text() or "").splitlines() if clean_text(x)]
        useful = [x for x in lines if not x.startswith("Conférence des Directeurs") and x != "/"]
        labels = ["Établissement", "Ville", "Document", "Campagne"]
        for i, value in enumerate(useful[-4:]):
            section = SectionRegion(f"META.{i+1}", labels[i], 0, 0)
            records.append(make_record(record_base(source, page_no, major, subsection, section), value=value, method="pdfplumber_defi_metadata", confidence=0.95))
    elif page_no == 2:
        pairs = [
            ("META.NOM_LEGAL", "Nom légal de l'École", r"Nom légal de l['’]École\s*:\s*(.+?)(?=Nom / Sigle / Appellation|L['’]école a pris|$)"),
            ("META.SIGLE", "Nom / Sigle / Appellation", r"Nom / Sigle / Appellation\s*:\s*(.+?)(?=L['’]école a pris|$)"),
        ]
        for code, title, pat in pairs:
            m = re.search(pat, text, re.I)
            if m:
                section = SectionRegion(code, title, 0, 0)
                records.append(make_record(record_base(source, page_no, major, subsection, section), value=m.group(1), method="pdfplumber_defi_metadata", confidence=0.97))
    elif page_no in {3, 4}:
        section = SectionRegion("META.PUBLICATION" if page_no == 3 else "META.RESEAUX", "Autorisations de publication" if page_no == 3 else "Réseaux et accès aux données", 0, page.height)
        for table in tables:
            matrix = table_matrix(table)
            if matrix:
                records.extend(parse_generic_table(source, page_no, major, subsection, section, table, False))
    return records


def extract_pdf(source: Path, include_empty: bool = False) -> list[Record]:
    records: list[Record] = []
    current_major = ("", "")
    current_subsection = ""
    previous_section: SectionRegion | None = None

    with pdfplumber.open(source) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            heading = find_major_heading(page)
            if heading:
                current_major = heading
                previous_section = None
            sub = find_subsection_heading(page)
            if sub:
                current_subsection = sub
            tables = filtered_tables(page)

            if page_no <= 4:
                records.extend(parse_metadata(source, page_no, page, tables))
                continue

            # Questionnaire XI : les tableaux portent eux-mêmes leur contexte.
            # La détection repose sur le contenu, car les longues listes peuvent
            # décaler les numéros de page d'une école à l'autre.
            if current_major[0] == "XI":
                found_xi = False
                for table in tables:
                    matrix = table_matrix(table)
                    section = special_xi_section(matrix, page_no)
                    if not section:
                        continue
                    found_xi = True
                    section.y0, section.y1 = float(table.bbox[1]), float(table.bbox[3])
                    recs = parse_generic_table(source, page_no, current_major, current_subsection, section, table, include_empty)
                    records.extend(recs)
                if found_xi:
                    continue

            sections, top_ids = detect_sections(page, tables)
            inner = [(idx, t) for idx, t in enumerate(tables) if idx not in top_ids and float(t.bbox[0]) >= 190 and float(t.bbox[2] - t.bbox[0]) > 20]

            # Partie supérieure d'une page de continuation avant la première nouvelle section.
            if sections and previous_section and not heading and sections[0].y0 > 65:
                cont = SectionRegion(previous_section.code, previous_section.title, 40.0, sections[0].y0 - 1, previous_section.subsection_title)
                sections.insert(0, cont)
            elif not sections and previous_section and inner:
                sections = [SectionRegion(previous_section.code, previous_section.title, 40.0, page.height - 8, previous_section.subsection_title)]

            for section in sections:
                matching = [
                    t for _, t in inner
                    if section.y0 - 2 <= (float(t.bbox[1]) + float(t.bbox[3])) / 2 <= section.y1 + 2
                ]
                section_records: list[Record] = []
                for table in matching:
                    matrix = table_matrix(table)
                    if section.code == "V.1" and is_chairs_table(matrix):
                        section_records.extend(
                            parse_chairs_table(
                                source,
                                page_no,
                                current_major,
                                current_subsection,
                                section,
                                table,
                                include_empty,
                            )
                        )
                    else:
                        section_records.extend(
                            parse_generic_table(
                                source,
                                page_no,
                                current_major,
                                current_subsection,
                                section,
                                table,
                                include_empty,
                            )
                        )
                records.extend(section_records)
                excluded = [tuple(map(float, t.bbox)) for t in matching]
                records.extend(parse_scalar_content(
                    source, page_no, page, current_major, current_subsection, section, excluded,
                    only_explicit=bool(section_records), include_empty=include_empty,
                ))
                previous_section = section

            # Pages sans tableau détecté : observations ou certification.
            page_text = clean_text(page.extract_text() or "")
            if not sections and "Observations" in page_text:
                section = SectionRegion("OBSERVATIONS", "Observations", 40, page.height)
                records.extend(parse_scalar_content(source, page_no, page, ("XI", "ORIGINE ACADÉMIQUE"), "", section, [], False, include_empty))
            elif not sections and "Je certifie" in page_text:
                content = re.sub(r"^.*?Conférence des Directeurs des Écoles Françaises d'Ingénieurs\s*/?", "", page_text)
                if content:
                    section = SectionRegion("META.CERTIFICATION", "Certification des données", 0, page.height)
                    records.append(make_record(record_base(source, page_no, ("META", "Métadonnées DEFI"), "", section), value=content, method="pdfplumber_defi_metadata", confidence=0.88))

    # Harmonise les trois valeurs du récapitulatif I.1. « Hommes » peut être
    # récupéré comme texte libre alors que « Femmes » et « Total » proviennent
    # du mini-tableau voisin ; ils doivent pourtant partager le même titre de
    # tableau dans la base.
    i1_summary_title = "Total (Cycle préparatoire + Cycle ingénieur)"
    for record in records:
        if (
            record.section_code == "I.1"
            and record.row_name.casefold() in {"hommes", "femmes", "total"}
            and record.column_path == "Valeur"
            and record.value_num is not None
        ):
            record.table_title = i1_summary_title
            record.row_group = i1_summary_title
            record.confidence = max(record.confidence, 0.98)

    # Déduplication stricte, sans supprimer deux valeurs identiques placées dans
    # des dimensions différentes.
    unique: list[Record] = []
    seen: set[tuple[Any, ...]] = set()
    for r in records:
        key = (
            r.source_file, r.source_page, r.section_code, r.section_title, r.subsection_title, r.table_title,
            r.row_group, r.row_name, r.column_path, r.value_text,
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS defi_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    source_page INTEGER NOT NULL,
    major_section_code TEXT,
    major_section_title TEXT,
    subsection_title TEXT,
    section_code TEXT NOT NULL,
    section_title TEXT,
    table_title TEXT,
    row_group TEXT,
    row_name TEXT,
    column_group TEXT,
    column_name TEXT,
    column_path TEXT,
    column_headers_json TEXT,
    value_text TEXT,
    value_num REAL,
    numbers_json TEXT,
    value_type TEXT,
    extraction_method TEXT,
    confidence REAL,
    raw_context TEXT
);
CREATE INDEX IF NOT EXISTS idx_defi_code ON defi_values(section_code);
CREATE INDEX IF NOT EXISTS idx_defi_source ON defi_values(source_file, source_page);
CREATE INDEX IF NOT EXISTS idx_defi_dimensions ON defi_values(section_code, row_name, column_path);
"""

FIELDS = list(Record.__dataclass_fields__.keys())


def write_sqlite(records: Sequence[Record], path: Path) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        con.executescript(SQL_SCHEMA)
        placeholders = ",".join("?" for _ in FIELDS)
        con.executemany(
            f"INSERT INTO defi_values ({','.join(FIELDS)}) VALUES ({placeholders})",
            [[getattr(r, f) for f in FIELDS] for r in records],
        )
        con.commit()
    finally:
        con.close()


def write_csv(records: Sequence[Record], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
        writer.writeheader()
        for r in records:
            writer.writerow({name: getattr(r, name) for name in FIELDS})


def write_review_csv(records: Sequence[Record], path: Path) -> None:
    review = [r for r in records if r.confidence < 0.84 or r.column_name.startswith("Colonne ")]
    write_csv(review, path)


def iter_pdfs(inputs: Sequence[str]) -> Iterator[Path]:
    found: set[Path] = set()
    for raw in inputs:
        p = Path(raw)
        candidates = [p] if p.is_file() else sorted(p.rglob("*.pdf")) if p.is_dir() else []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in found:
                found.add(resolved)
                yield candidate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extrait les PDF DEFI vers SQLite et CSV.")
    p.add_argument("inputs", nargs="+", help="PDF ou dossiers contenant des PDF")
    p.add_argument("--sqlite", type=Path, default=Path("defi.sqlite"), help="Base SQLite de sortie")
    p.add_argument("--csv", type=Path, default=Path("defi.csv"), help="CSV de sortie")
    p.add_argument("--review-csv", type=Path, default=Path("defi-review.csv"), help="CSV des lignes à vérifier")
    p.add_argument("--include-empty", action="store_true", help="Conserver également les cellules vides")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pdfs = list(iter_pdfs(args.inputs))
    if not pdfs:
        print("Aucun PDF trouvé.", file=sys.stderr)
        return 2
    all_records: list[Record] = []
    seen_contents: set[tuple[str, int | None, str]] = set()
    for source in pdfs:
        school = re.split(r"[-_ ]defi[-_ ]", source.stem, flags=re.I)[0].casefold()
        with pdfplumber.open(source) as pdf:
            first_text = " ".join(page.extract_text() or "" for page in pdf.pages[:2])
        campaign = re.search(r"\bcampagne\s*[:\-]?\s*(20\d{2})\b", first_text, re.I)
        filename_years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", source.stem)
        year = int(campaign.group(1)) if campaign else int(filename_years[-1]) if filename_years else None
        identity = (school, year, hashlib.sha256(source.read_bytes()).hexdigest())
        if identity in seen_contents:
            print(f"Doublon de contenu ignoré : {source}")
            continue
        seen_contents.add(identity)
        print(f"Extraction : {source}")
        recs = extract_pdf(source, include_empty=args.include_empty)
        print(f"  {len(recs)} enregistrements")
        all_records.extend(recs)
    args.sqlite.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    write_sqlite(all_records, args.sqlite)
    write_csv(all_records, args.csv)
    write_review_csv(all_records, args.review_csv)
    print(f"Total : {len(all_records)} enregistrements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

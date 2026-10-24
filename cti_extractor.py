#!/usr/bin/env python3
"""Extraction structurée des fiches « Données certifiées » de la CTI.

Le script est conçu pour les PDF CTI générés par FPDF, contenant :
- des questions simples sous la forme « code / intitulé / valeur » ;
- des tableaux à bordures ;
- des en-têtes de colonnes hiérarchiques sur plusieurs lignes.

Sorties :
- une base SQLite structurée avec une table physique par section ;
- un catalogue des sections et des colonnes ;
- en option, un CSV plat et une base SQLite plate de diagnostic.

Exemple :
    python cti_extractor.py ./pdfs --sqlite cti.sqlite --csv cti.csv

Dépendance : pdfplumber >= 0.11
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pdfplumber


CODE_RE = re.compile(r"^([IVX]+(?:[.\-][A-Za-z0-9]+)+)")
CODE_ANY_RE = re.compile(r"\b([IVX]+(?:[.\-][A-Za-z0-9]+)+)\b")
MAJOR_SECTION_RE = re.compile(r"^([IVX]+)\.\s+(.+)$")
SUBSECTION_RE = re.compile(r"^([IVX]+(?:\.\d+|\.D\d+)+)\s*[-–]\s*(.+)$")

ROW_LABELS = {
    "hommes", "femmes", "total", "nb.", "nb", "nombre", "nombre intégrés",
    "avec prime", "sans prime", "adresse", "code postal", "ville",
    "nom du(des) test(s)", "niveau requis",
}

LIST_FIELD_LABELS = {
    "adresse", "code postal", "ville", "nom du(des) test(s)", "niveau requis",
}

YES_NO = {"oui", "non", "public", "privé", "prive"}


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
    if s.casefold() == "avec prim":
        s = "Avec prime"
    elif s.casefold() == "sans prim":
        s = "Sans prime"
    return s


def clean_code(value: str) -> str:
    return re.sub(r"\s+", "", value).rstrip("-.")


def remove_code_prefix(text: str, code: str) -> str:
    """Retire un code, même s'il est coupé par des espaces ou retours ligne."""
    s = clean_text(text)
    if not code:
        return re.sub(r"^[IVX]+(?:[.\-][A-Za-z0-9]+)+\s*", "", s).strip()
    flexible = r"\s*".join(re.escape(ch) for ch in code)
    return re.sub(r"^\s*" + flexible + r"\s*", "", s, count=1).strip()


def code_prefix(code: str) -> str:
    """Préfixe de contexte, par ex. II.1-D1-6 -> II.1-D1-."""
    m = re.match(r"^(.*[.\-])[^.\-]+$", code)
    return m.group(1) if m else code


def numeric_token(value: str) -> str | None:
    """Retourne un nombre exact, en tolérant un petit résidu de mise en page.

    Exemples acceptés : 42, 12.5, 5,013, 98.23 %, « 42750 e ».
    Les dates, téléphones et URL ne sont pas transformés.
    """
    s = clean_text(value)
    s = s.replace("\u00a0", " ").replace("\u202f", " ")
    s = re.sub(r"\s*(?:€|euros?)\s*$", "", s, flags=re.IGNORECASE)
    if not s or "/" in s or "@" in s or "://" in s:
        return None
    exact = re.fullmatch(r"[-+]?\d+(?:[ .]\d{3})*(?:[,.]\d+)?\s*%?", s)
    if exact:
        return s
    # Résidu très court provenant d'un mot coupé par une bordure : « 42750 e ».
    loose = re.fullmatch(r"\s*([-+]?\d+(?:[,.]\d+)?)\s*[A-Za-zéèêàùç]{0,2}\s*", s)
    return loose.group(1) if loose else None


def parse_number(value: str) -> float | None:
    token = numeric_token(value)
    if token is None:
        return None
    token = token.replace("%", "").replace(" ", "").replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def value_type(value: str) -> str:
    s = clean_text(value)
    if not s:
        return "blank"
    if parse_number(s) is not None:
        return "number"
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
        return "date"
    if "://" in s:
        return "url"
    if "@" in s and " " not in s:
        return "email"
    if s.lower() in YES_NO:
        return "boolean_or_category"
    return "text"


def looks_like_value(value: str) -> bool:
    s = clean_text(value)
    if not s:
        return False
    if parse_number(s) is not None:
        return True
    if s.lower() in YES_NO:
        return True
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
        return True
    return False


def dedupe_parts(parts: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        p = clean_text(part)
        key = p.casefold()
        if p and key not in seen:
            seen.add(key)
            result.append(p)
    return result


@dataclass
class Record:
    source_file: str
    source_page: int
    major_section_code: str = ""
    major_section_title: str = ""
    subsection_title: str = ""
    population_context: str = ""
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
    value_type: str = "text"
    extraction_method: str = ""
    confidence: float = 0.0
    raw_context: str = ""


@dataclass
class Anchor:
    code: str
    top: float
    bottom: float
    center: float
    line_indices: tuple[int, ...]


@dataclass
class HeaderSpan:
    x0: float
    x1: float
    headers: tuple[str, ...]


@dataclass
class HeaderContext:
    y_start: float
    y_end: float
    spans: tuple[HeaderSpan, ...]
    prefix: str = ""
    source_code: str = ""


def extract_lines(page: pdfplumber.page.Page) -> list[dict[str, Any]]:
    lines = page.extract_text_lines(layout=False, strip=True, return_chars=False)
    return sorted(lines, key=lambda x: (float(x["top"]), float(x["x0"])))


def extract_anchors(page: pdfplumber.page.Page) -> tuple[list[Anchor], list[dict[str, Any]]]:
    lines = extract_lines(page)
    anchors: list[Anchor] = []
    consumed: set[int] = set()

    for i, line in enumerate(lines):
        if i in consumed or float(line["x0"]) > 45:
            continue
        text = clean_text(line["text"])
        if not text:
            continue
        first = text.split()[0]
        candidate = first
        used = [i]

        # Codes coupés sur deux lignes : « II.1-D1- » puis « 7-1 ».
        if first.endswith("-"):
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j]
                if float(nxt["x0"]) > 45:
                    continue
                if float(nxt["top"]) - float(line["top"]) > 28:
                    break
                suffix = clean_text(nxt["text"]).split()[0]
                if re.fullmatch(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*", suffix):
                    candidate = first + suffix
                    used.append(j)
                    break

        m = CODE_RE.match(candidate)
        if not m:
            m = CODE_RE.match(text)
        if not m:
            continue
        code = clean_code(m.group(1))
        # Une section principale telle que « I. » n'est pas une question.
        if not re.search(r"\d", code):
            continue
        top = min(float(lines[k]["top"]) for k in used)
        bottom = max(float(lines[k]["bottom"]) for k in used)
        anchors.append(Anchor(code, top, bottom, (top + bottom) / 2, tuple(used)))
        consumed.update(used[1:])

    # Déduplication : certains codes peuvent apparaître deux fois au même endroit.
    unique: list[Anchor] = []
    for a in sorted(anchors, key=lambda x: x.center):
        if unique and a.code == unique[-1].code and abs(a.center - unique[-1].center) < 5:
            continue
        unique.append(a)
    return unique, lines


def anchor_regions(anchors: Sequence[Anchor], page_height: float) -> dict[str, tuple[float, float]]:
    regions: dict[str, tuple[float, float]] = {}
    for i, a in enumerate(anchors):
        start = (0.0 if a.center <= 50 else max(0.0, a.center - 38)) if i == 0 else (anchors[i - 1].center + a.center) / 2
        end = page_height - 28 if i == len(anchors) - 1 else (a.center + anchors[i + 1].center) / 2
        regions[a.code] = (max(0.0, start), min(page_height, end))
    return regions


def major_and_subsection(lines: Sequence[dict[str, Any]], y: float) -> tuple[str, str, str]:
    major_code = ""
    major_title = ""
    subsection = ""
    for line in lines:
        if float(line["top"]) > y:
            break
        text = clean_text(line["text"])
        m = MAJOR_SECTION_RE.match(text)
        if m and text.upper() == text and len(text) > 8:
            major_code, major_title = m.group(1), clean_text(m.group(2))
        sm = SUBSECTION_RE.match(text)
        if sm:
            subsection = clean_text(text)
    return major_code, major_title, subsection


def valid_tables(page: pdfplumber.page.Page) -> list[Any]:
    result = []
    for table in page.find_tables():
        x0, y0, x1, y1 = table.bbox
        if x0 < -1 or y0 < -1 or x1 > page.width + 1 or y1 > page.height + 1:
            continue
        if len(table.cells) < 2 or not table.rows or not table.columns:
            continue
        # Élimine les faux tableaux constitués par la bordure de page.
        if (x1 - x0) > page.width * 1.02 or (y1 - y0) > page.height * 1.02:
            continue
        result.append(table)
    return sorted(result, key=lambda t: (t.bbox[1], t.bbox[0]))


def table_column_intervals(table: Any, ncols: int) -> list[tuple[float, float]]:
    """Reconstruit les vraies colonnes à partir des arêtes de cellules.

    Les objets ``table.columns`` de pdfplumber peuvent reprendre la largeur
    d'une cellule fusionnée. Les arêtes uniques sont plus fiables.
    """
    xs: list[float] = []
    for cell in table.cells:
        xs.extend([float(cell[0]), float(cell[2])])
    xs = sorted(xs)
    unique: list[float] = []
    for x in xs:
        if not unique or abs(x - unique[-1]) > 0.8:
            unique.append(x)
    intervals = list(zip(unique[:-1], unique[1:]))
    if len(intervals) == ncols:
        return intervals
    # Repli : limites globales réparties selon les colonnes reconnues.
    if len(table.columns) == ncols:
        out = []
        for col in table.columns:
            x0, _, x1, _ = col.bbox
            out.append((float(x0), float(x1)))
        return out
    x0, _, x1, _ = table.bbox
    step = (x1 - x0) / max(1, ncols)
    return [(x0 + i * step, x0 + (i + 1) * step) for i in range(ncols)]


def table_matrix(table: Any) -> list[list[str]]:
    raw = table.extract()
    ncols = max((len(r) for r in raw), default=0)
    return [[clean_text(r[c] if c < len(r) else "") for c in range(ncols)] for r in raw]


def data_rows(matrix: Sequence[Sequence[str]]) -> list[int]:
    rows: list[int] = []
    for r, row in enumerate(matrix):
        nvals = sum(1 for c in row if looks_like_value(c))
        has_known_label = any(clean_text(c).casefold() in ROW_LABELS for c in row)
        if nvals > 0 and (has_known_label or nvals >= 2):
            rows.append(r)
    return rows


def filled_header_paths(matrix: Sequence[Sequence[str]], header_rows: Sequence[int], data_start: int) -> list[list[str]]:
    ncols = max((len(r) for r in matrix), default=0)
    paths: list[list[str]] = [[] for _ in range(ncols)]
    previous_paths: list[tuple[str, ...]] = [tuple() for _ in range(ncols)]

    for ridx in header_rows:
        row = list(matrix[ridx]) + [""] * (ncols - len(matrix[ridx]))
        filled = [""] * ncols
        current = ""
        for c in range(data_start, ncols):
            cell = clean_text(row[c])
            if cell:
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


def table_codes(matrix: Sequence[Sequence[str]]) -> list[str]:
    found: list[str] = []
    for row in matrix:
        for cell in row:
            compact = clean_text(cell).replace(" ", "")
            for m in CODE_ANY_RE.finditer(compact):
                code = clean_code(m.group(1))
                if re.search(r"\d", code) and code not in found:
                    found.append(code)
    return found


def anchor_for_table(table: Any, anchors: Sequence[Anchor], regions: dict[str, tuple[float, float]]) -> Anchor | None:
    _, y0, _, y1 = table.bbox
    center = (y0 + y1) / 2
    candidates = [a for a in anchors if regions[a.code][0] - 3 <= center <= regions[a.code][1] + 3]
    if not candidates:
        candidates = [a for a in anchors if y0 - 18 <= a.center <= y1 + 18]
    if not candidates:
        return None
    return min(candidates, key=lambda a: abs(a.center - center))


def words_in_box(page: pdfplumber.page.Page, x0: float, y0: float, x1: float, y1: float) -> list[dict[str, Any]]:
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    out = []
    for w in words:
        cx = (float(w["x0"]) + float(w["x1"])) / 2
        cy = (float(w["top"]) + float(w["bottom"])) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            out.append(w)
    return sorted(out, key=lambda w: (float(w["top"]), float(w["x0"])))


def group_words_by_line(words: Sequence[dict[str, Any]], tolerance: float = 3.0) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda w: (float(w["top"]), float(w["x0"]))):
        if not groups:
            groups.append([word])
            continue
        mean_top = sum(float(w["top"]) for w in groups[-1]) / len(groups[-1])
        if abs(float(word["top"]) - mean_top) <= tolerance:
            groups[-1].append(word)
        else:
            groups.append([word])
    for g in groups:
        g.sort(key=lambda w: float(w["x0"]))
    return groups


def line_text(words: Sequence[dict[str, Any]]) -> str:
    return clean_text(" ".join(str(w["text"]) for w in words))


ACADEMIC_YEAR_RE = re.compile(
    r"^(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}$"
)


def is_academic_year_marker(value: Any) -> bool:
    """Reconnaît un en-tête de campagne tel que 2025-2026."""
    return ACADEMIC_YEAR_RE.fullmatch(clean_text(value)) is not None


def scalar_value_split_x(
    words: Sequence[dict[str, Any]],
    default_split: float = 250.0,
) -> float:
    """Détecte géométriquement le début d'une valeur scalaire simple."""
    candidates: list[float] = []

    for group in group_words_by_line(words):
        ordered = sorted(group, key=lambda w: float(w["x0"]))

        for i in range(1, len(ordered)):
            gap = float(ordered[i]["x0"]) - float(ordered[i - 1]["x1"])
            if gap < 10.0:
                continue

            tail = line_text(ordered[i:])
            if looks_like_value(tail):
                candidates.append(float(ordered[i]["x0"]) - 2.0)

        if ordered:
            last = ordered[-1]
            last_text = clean_text(last.get("text", ""))
            if (
                looks_like_value(last_text)
                and float(last["x0"]) > default_split + 12
            ):
                candidates.append(float(last["x0"]) - 2.0)

    return max(candidates) if candidates else default_split


def title_for_anchor(page: pdfplumber.page.Page, anchor: Anchor, region: tuple[float, float], cutoff_x: float = 250) -> str:
    words = words_in_box(page, 52, region[0], max(55, cutoff_x), region[1])
    texts = []
    for group in group_words_by_line(words):
        t = line_text(group)
        t = re.sub(r"^[IVX]+(?:[.\-][A-Za-z0-9]+)+\s*", "", t)
        if t and not t.lower().startswith("page "):
            texts.append(t)
    return clean_text(" ".join(dedupe_parts(texts)))


def header_context_from_table(table: Any, matrix: Sequence[Sequence[str]], y_end: float, prefix: str = "", source_code: str = "") -> HeaderContext:
    ncols = max((len(r) for r in matrix), default=0)
    paths = filled_header_paths(matrix, range(len(matrix)), 0)
    intervals = table_column_intervals(table, ncols)
    spans: list[HeaderSpan] = []
    for c in range(min(ncols, len(paths), len(intervals))):
        p = tuple(dedupe_parts(paths[c]))
        if not p:
            continue
        x0, x1 = intervals[c]
        spans.append(HeaderSpan(float(x0), float(x1), p))
    return HeaderContext(float(table.bbox[3]), float(y_end), tuple(spans), prefix, source_code)


def context_for_anchor(anchor: Anchor, contexts: Sequence[HeaderContext]) -> HeaderContext | None:
    pref = code_prefix(anchor.code)
    specific = [c for c in contexts if c.prefix and pref.startswith(c.prefix)]
    if specific:
        return min(specific, key=lambda c: abs(anchor.center - c.y_start))
    generic = [c for c in contexts if not c.prefix and c.y_start <= anchor.center <= c.y_end]
    return generic[-1] if generic else None


def population_context_for_anchor(
    lines: Sequence[dict[str, Any]],
    y: float,
) -> str:
    """Contexte de cohorte placé au-dessus des tableaux VIII.D*.

    Exemples CTI :
    - Situation des diplômés de la dernière promotion en janvier après l'obtention du diplôme
    - Situation des diplômés de l'avant dernière promotion en janvier, plus d'un an après la sortie
    """
    context = ""
    for line in lines:
        if float(line["top"]) > y:
            break
        text = clean_text(line["text"])
        folded = text.casefold()
        if "situation des diplômés" in folded or "situation des diplomes" in folded:
            context = text
    return context


def record_base(source: Path, page_no: int, lines: Sequence[dict[str, Any]], y: float, code: str, title: str) -> dict[str, Any]:
    major_code, major_title, subsection = major_and_subsection(lines, y)
    return {
        "source_file": source.name,
        "source_page": page_no,
        "major_section_code": major_code,
        "major_section_title": major_title,
        "subsection_title": subsection,
        "population_context": population_context_for_anchor(lines, y),
        "section_code": code,
        "section_title": clean_text(title),
    }



SALARY_SECTION_RE = re.compile(r"^VIII[-.]D\d+[-.]7$", re.IGNORECASE)


def normalized_section_code(value: str) -> str:
    return clean_code(clean_text(value).replace(".", "-"))


def is_salary_detail_section(value: str) -> bool:
    return SALARY_SECTION_RE.fullmatch(normalized_section_code(value)) is not None


def parse_money_value(value: Any) -> float | None:
    """Convertit une cellule de salaire sans confondre les répondants."""
    text = clean_text(value)
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = re.sub(r"\s*(?:€|euros?|e)\s*$", "", text, flags=re.IGNORECASE)
    number = parse_number(text)

    if number is None:
        return None

    if 10_000 <= number <= 200_000:
        return float(number)

    return None


def parse_response_count(value: Any) -> float | None:
    text = clean_text(value)
    number = parse_number(text)

    if number is None:
        return None

    if 0 <= number <= 10_000 and abs(number - round(number)) < 1e-9:
        return float(round(number))

    return None


def salary_numeric_text_from_cell(
    page: pdfplumber.page.Page,
    table: Any,
    matrix_row: int,
    matrix_column: int,
) -> str:
    """Relit une cellule de salaire à partir des glyphes du PDF.

    Certains PDF CTI font légèrement déborder un nombre sur une bordure de
    cellule. ``table.extract()`` peut alors dupliquer un glyphe adjacent et
    transformer par exemple ``40400`` en ``400400``. Cette lecture géométrique
    réaffecte les glyphes selon leur centre et élimine les quasi-doublons.
    """
    try:
        cell = table.rows[matrix_row].cells[matrix_column]
    except (AttributeError, IndexError, TypeError):
        return ""

    if not cell:
        return ""

    x0, top, x1, bottom = map(float, cell)
    chars: list[dict[str, Any]] = []

    for char in page.chars:
        glyph = str(char.get("text", ""))
        if not glyph or not all(ch.isdigit() or ch in ".," for ch in glyph):
            continue

        cx = (float(char["x0"]) + float(char["x1"])) / 2
        cy = (float(char["top"]) + float(char["bottom"])) / 2
        if x0 <= cx < x1 and top <= cy < bottom:
            chars.append(char)

    if not chars:
        return ""

    # Une cellule de salaire ne contient normalement qu'une ligne numérique.
    line_groups: list[list[dict[str, Any]]] = []
    for char in sorted(chars, key=lambda c: (float(c["top"]), float(c["x0"]))):
        cy = (float(char["top"]) + float(char["bottom"])) / 2
        for group in line_groups:
            ref = group[0]
            ref_cy = (float(ref["top"]) + float(ref["bottom"])) / 2
            if abs(cy - ref_cy) <= 2.0:
                group.append(char)
                break
        else:
            line_groups.append([char])

    chars = max(
        line_groups,
        key=lambda group: sum(str(c.get("text", "")).isdigit() for c in group),
    )
    chars = sorted(chars, key=lambda c: float(c["x0"]))

    if len(chars) <= 1:
        return "".join(str(c.get("text", "")) for c in chars)

    widths = [max(0.1, float(c["x1"]) - float(c["x0"])) for c in chars]
    typical_width = sorted(widths)[len(widths) // 2]

    clusters: list[list[dict[str, Any]]] = []
    for char in chars:
        if not clusters:
            clusters.append([char])
            continue
        prev = clusters[-1][-1]
        prev_cx = (float(prev["x0"]) + float(prev["x1"])) / 2
        cx = (float(char["x0"]) + float(char["x1"])) / 2
        if abs(cx - prev_cx) < typical_width * 0.35:
            clusters[-1].append(char)
        else:
            clusters.append([char])

    if all(len(cluster) == 1 for cluster in clusters):
        return "".join(str(cluster[0].get("text", "")) for cluster in clusters)

    import itertools

    best_text = ""
    best_score: float | None = None
    cell_center = (x0 + x1) / 2

    for choice in itertools.product(*clusters):
        selected = sorted(choice, key=lambda c: float(c["x0"]))
        candidate = "".join(str(c.get("text", "")) for c in selected)
        if parse_money_value(candidate) is None:
            continue

        centers = [
            (float(c["x0"]) + float(c["x1"])) / 2
            for c in selected
        ]
        gaps = [b - a for a, b in zip(centers, centers[1:])]
        if gaps:
            mean_gap = sum(gaps) / len(gaps)
            spacing_error = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
        else:
            spacing_error = 0.0

        sequence_center = (centers[0] + centers[-1]) / 2
        score = spacing_error + 0.02 * abs(sequence_center - cell_center)

        if best_score is None or score < best_score:
            best_score = score
            best_text = candidate

    return best_text


def salary_block_starts(matrix: Sequence[Sequence[str]]) -> list[int]:
    if len(matrix) < 2:
        return []

    row = list(matrix[1])
    starts = [
        index
        for index, cell in enumerate(row)
        if clean_text(cell).casefold().startswith("avec prime")
    ]

    # Le format CTI standard possède deux blocs de 7 colonnes. Certaines
    # cellules fusionnées peuvent faire disparaître le second libellé.
    if not starts and len(row) >= 7:
        starts = [0]

    if len(row) >= 14 and 7 not in starts:
        starts.append(7)

    return sorted(set(starts))


def make_salary_record(
    *,
    row_base: dict[str, Any],
    table: Any,
    matrix_row: int,
    matrix_column: int,
    row_name: str,
    track: str,
    metric: str,
    population: str,
    value: float,
    confidence: float,
    correction: str = "",
) -> Record:
    headers = [track, metric, population]
    context = {
        "table_bbox": table.bbox,
        "row": matrix_row,
        "column": matrix_column,
        "salary_parser": "cti_salary_v2",
        "semantic_role": {
            "track": track,
            "metric": metric,
            "population": population,
        },
    }

    if correction:
        context["correction"] = correction

    return Record(
        **row_base,
        row_name=row_name,
        column_group=" > ".join(headers[:-1]),
        column_name=headers[-1],
        column_path=" > ".join(headers),
        column_headers_json=json.dumps(headers, ensure_ascii=False),
        value_text=str(int(value)) if float(value).is_integer() else str(value),
        value_num=float(value),
        value_type="number",
        extraction_method=(
            "pdfplumber_salary_reclassified"
            if correction
            else "pdfplumber_salary_table"
        ),
        confidence=confidence,
        raw_context=json.dumps(context, ensure_ascii=False),
    )


def make_salary_schema_record(
    *,
    row_base: dict[str, Any],
    table: Any,
    matrix_row: int,
    matrix_column: int,
    row_name: str,
    track: str,
    metric: str,
    population: str,
) -> Record:
    """Déclare une colonne de salaire/réponses même si la cellule est vide."""
    headers = [track, metric, population]
    context = {
        "table_bbox": table.bbox,
        "row": matrix_row,
        "column": matrix_column,
        "salary_parser": "cti_salary_v2",
        "schema_only": True,
        "semantic_role": {
            "track": track,
            "metric": metric,
            "population": population,
        },
    }
    return Record(
        **row_base,
        row_name=row_name,
        column_group=" > ".join(headers[:-1]),
        column_name=headers[-1],
        column_path=" > ".join(headers),
        column_headers_json=json.dumps(headers, ensure_ascii=False),
        value_text="",
        value_num=None,
        value_type="blank",
        extraction_method="pdfplumber_salary_schema",
        confidence=0.90,
        raw_context=json.dumps(context, ensure_ascii=False),
    )


def is_salary_table_candidate(
    matrix: Sequence[Sequence[str]],
    anchor: Anchor | None,
) -> bool:
    """Reconnaît VIII-D*-7 même lorsque toutes les cellules chiffrées sont vides."""
    codes = table_codes(matrix)
    code = codes[0] if codes else (anchor.code if anchor else "")
    if not is_salary_detail_section(code):
        return False
    flat = " ".join(clean_text(cell) for row in matrix for cell in row).casefold()
    return any(
        marker in flat
        for marker in ("avec prime", "avec prim", "sans prime", "sans prim")
    )


def parse_salary_table_records(
    source: Path,
    page_no: int,
    page: pdfplumber.page.Page,
    lines: Sequence[dict[str, Any]],
    table: Any,
    matrix: Sequence[Sequence[str]],
    anchor: Anchor | None,
    region: tuple[float, float] | None,
    include_empty: bool,
) -> list[Record] | None:
    """Parse les sections VIII-D*-7 selon leur géométrie CTI stable.

    Le tableau contient, pour FISE puis FISA+FISEA :
    - les salaires médians H / F / Total ;
    - un séparateur « Nb. rép. » ;
    - les nombres de réponses H / F.

    Cette fonction évite le décalage historique où les salaires F/Total
    étaient rangés sous « Nombre de réponses ».
    """
    codes = table_codes(matrix)
    code = codes[0] if codes else (anchor.code if anchor else "")

    if not is_salary_detail_section(code):
        return None

    flat = " ".join(clean_text(cell) for row in matrix for cell in row).casefold()
    # pdfplumber coupe parfois « prime » sur une bordure :
    # « Avec\nprim » + « e ». Le préfixe suffit à reconnaître
    # sans ambiguïté le tableau spécialisé de salaire VIII-D*-7.
    if not any(
        marker in flat
        for marker in ("avec prime", "avec prim", "sans prime", "sans prim")
    ):
        return None

    if len(matrix) < 2:
        return None

    ncols = max((len(row) for row in matrix), default=0)
    if ncols < 6:
        return None

    title = ""
    if anchor and region:
        cutoff = max(55, float(table.bbox[0]) - 4)
        title = title_for_anchor(page, anchor, region, cutoff)

    if not title:
        title = "Salaire annuel brut médian en France, avec et sans prime"

    y = anchor.center if anchor else float(table.bbox[1])
    row_base = record_base(source, page_no, lines, y, code, title)
    records: list[Record] = []
    starts = salary_block_starts(matrix)

    if not starts:
        return None

    header = list(matrix[0]) + [""] * (ncols - len(matrix[0]))
    tracks = ["FISE", "FISA + FISEA"]

    for matrix_row in range(1, len(matrix)):
        row = list(matrix[matrix_row]) + [""] * (ncols - len(matrix[matrix_row]))

        for block_index, start in enumerate(starts):
            end = starts[block_index + 1] if block_index + 1 < len(starts) else ncols
            track = tracks[block_index] if block_index < len(tracks) else f"Voie {block_index + 1}"
            row_name = clean_text(row[start])

            if row_name.casefold().startswith("avec prim"):
                row_name = "Avec prime"
            elif row_name.casefold().startswith("sans prim"):
                row_name = "Sans prime"

            if not row_name:
                continue

            marker = None
            for column in range(start + 1, end):
                folded = clean_text(row[column]).casefold().replace(".", "")
                if "nb" in folded and ("rep" in folded or "rép" in folded):
                    marker = column
                    break

            if marker is None:
                continue

            salary_columns = list(range(start + 1, marker))
            response_columns = list(range(marker + 1, end))
            salary_values: dict[str, float] = {}
            response_values: dict[str, float] = {}
            raw_salary_numbers: dict[str, tuple[float, int]] = {}

            fallback_salary_labels = ["H", "F", "Total"]
            fallback_response_labels = ["H", "F", "Total"]

            for offset, column in enumerate(salary_columns):
                label = clean_text(header[column]) or (
                    fallback_salary_labels[offset]
                    if offset < len(fallback_salary_labels)
                    else str(offset + 1)
                )
                raw_text = clean_text(row[column])
                geometric_text = salary_numeric_text_from_cell(
                    page, table, matrix_row, column
                )

                # La lecture géométrique est prioritaire lorsqu'elle produit
                # un salaire plausible ; sinon on garde table.extract().
                value = parse_money_value(geometric_text)
                correction = ""
                if value is not None:
                    if geometric_text and parse_money_value(raw_text) != value:
                        correction = "salary_cell_geometry_repair"
                    numeric_source = geometric_text
                else:
                    value = parse_money_value(raw_text)
                    numeric_source = raw_text

                raw_number = parse_number(numeric_source)
                if raw_number is None:
                    raw_number = parse_number(raw_text)
                if raw_number is not None:
                    raw_salary_numbers[label] = (float(raw_number), column)

                if value is not None:
                    salary_values[label] = value
                    records.append(
                        make_salary_record(
                            row_base=row_base,
                            table=table,
                            matrix_row=matrix_row,
                            matrix_column=column,
                            row_name=row_name,
                            track=track,
                            metric="Salaire médian",
                            population=label,
                            value=value,
                            confidence=0.99 if correction else 0.995,
                            correction=correction,
                        )
                    )
                else:
                    # Même vide, l'en-tête définit une vraie colonne métier.
                    records.append(
                        make_salary_schema_record(
                            row_base=row_base,
                            table=table,
                            matrix_row=matrix_row,
                            matrix_column=column,
                            row_name=row_name,
                            track=track,
                            metric="Salaire médian",
                            population=label,
                        )
                    )

            for offset, column in enumerate(response_columns):
                label = clean_text(header[column]) or (
                    fallback_response_labels[offset]
                    if offset < len(fallback_response_labels)
                    else str(offset + 1)
                )
                value = parse_response_count(row[column])
                if value is None:
                    records.append(
                        make_salary_schema_record(
                            row_base=row_base,
                            table=table,
                            matrix_row=matrix_row,
                            matrix_column=column,
                            row_name=row_name,
                            track=track,
                            metric="Nombre de réponses",
                            population=label,
                        )
                    )
                    continue
                response_values[label] = value
                records.append(
                    make_salary_record(
                        row_base=row_base,
                        table=table,
                        matrix_row=matrix_row,
                        matrix_column=column,
                        row_name=row_name,
                        track=track,
                        metric="Nombre de réponses",
                        population=label,
                        value=value,
                        confidence=0.995,
                    )
                )

            # Certains PDF placent le total des répondants dans la cellule
            # « salaire total ». On ne le traite jamais comme un salaire.
            raw_total = raw_salary_numbers.get("Total")
            response_sum = None
            if "H" in response_values and "F" in response_values:
                response_sum = response_values["H"] + response_values["F"]

            if (
                raw_total is not None
                and "Total" not in salary_values
                and response_sum is not None
                and abs(raw_total[0] - response_sum) < 1e-9
            ):
                records.append(
                    make_salary_record(
                        row_base=row_base,
                        table=table,
                        matrix_row=matrix_row,
                        matrix_column=raw_total[1],
                        row_name=row_name,
                        track=track,
                        metric="Nombre de réponses",
                        population="Total",
                        value=response_sum,
                        confidence=0.99,
                        correction=(
                            "total_repondants_reclasse_depuis_cellule_salaire_total"
                        ),
                    )
                )
            elif response_sum is not None and "Total" not in response_values:
                # Total dérivé, clairement identifié dans la provenance.
                records.append(
                    make_salary_record(
                        row_base=row_base,
                        table=table,
                        matrix_row=matrix_row,
                        matrix_column=marker,
                        row_name=row_name,
                        track=track,
                        metric="Nombre de réponses",
                        population="Total",
                        value=response_sum,
                        confidence=0.97,
                        correction="total_repondants_calcule_h_plus_f",
                    )
                )
            elif "Total" not in response_values:
                # Le total fait partie du schéma attendu même si H/F sont vides.
                records.append(
                    make_salary_schema_record(
                        row_base=row_base,
                        table=table,
                        matrix_row=matrix_row,
                        matrix_column=marker,
                        row_name=row_name,
                        track=track,
                        metric="Nombre de réponses",
                        population="Total",
                    )
                )

    if records or include_empty:
        return records

    return None



def infer_empty_structured_table_layout(
    matrix: Sequence[Sequence[str]],
) -> dict[str, Any] | None:
    """Reconnaît génériquement un tableau structuré dont les cellules métier sont vides.

    Le schéma est déduit des en-têtes et des libellés de lignes, jamais des valeurs.
    Cela couvre notamment les tableaux H/F/Total, FISE/FISA, durées, pays, etc.
    """
    if len(matrix) < 2:
        return None

    ncols = max((len(row) for row in matrix), default=0)
    if ncols < 2:
        return None

    padded = [list(row) + [""] * (ncols - len(row)) for row in matrix]

    # Ignore uniquement les lignes totalement vides en fin de matrice.
    while padded and not any(clean_text(cell) for cell in padded[-1]):
        padded.pop()
    if len(padded) < 2:
        return None

    best: dict[str, Any] | None = None

    # data_start sépare les colonnes descriptives (à gauche) des colonnes métier.
    for data_start in range(1, ncols):
        idx = len(padded) - 1
        data_rows_rev: list[int] = []

        # Un tableau vide possède généralement en fin de tableau des lignes dont
        # le libellé est à gauche et dont toutes les cellules métier sont vides.
        while idx >= 0:
            row = padded[idx]
            left_nonempty = any(clean_text(cell) for cell in row[:data_start])
            right_nonempty = any(clean_text(cell) for cell in row[data_start:])
            if left_nonempty and not right_nonempty:
                data_rows_rev.append(idx)
                idx -= 1
                continue
            break

        if not data_rows_rev:
            continue

        first_data = idx + 1
        if first_data <= 0:
            continue

        header_rows = list(range(first_data))
        paths = filled_header_paths(padded, header_rows, data_start)
        business_columns = [
            column
            for column in range(data_start, ncols)
            if column < len(paths) and paths[column]
        ]
        if not business_columns:
            continue

        # Il faut qu'au moins un vrai en-tête se trouve au-dessus des colonnes
        # métier ; sinon il peut s'agir d'un simple bloc de texte.
        if not any(
            any(clean_text(padded[r][c]) for c in business_columns)
            for r in header_rows
        ):
            continue

        data_rows = list(reversed(data_rows_rev))
        row_labels = [
            clean_text(" | ".join(
                cell for cell in padded[r][:data_start] if clean_text(cell)
            ))
            for r in data_rows
        ]
        row_labels = [label for label in row_labels if label]
        if not row_labels:
            continue

        score = len(data_rows) * 1000 + len(business_columns) * 10 - data_start
        candidate = {
            "matrix": padded,
            "ncols": ncols,
            "data_start": data_start,
            "first_data": first_data,
            "header_rows": header_rows,
            "data_rows": data_rows,
            "paths": paths,
            "business_columns": business_columns,
            "row_labels": row_labels,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = candidate

    return best


def is_empty_structured_table_candidate(
    matrix: Sequence[Sequence[str]],
) -> bool:
    return infer_empty_structured_table_layout(matrix) is not None


def structured_empty_table_expects_numeric(
    headers_by_column: Sequence[Sequence[str]],
    row_labels: Sequence[str],
) -> bool:
    """Déduit si un tableau vide contient normalement des mesures numériques."""
    combined = fold_text(
        " | ".join(
            [" > ".join(headers) for headers in headers_by_column]
            + list(row_labels)
        )
    )
    numeric_markers = (
        "homme", "hommes", "femme", "femmes", "total",
        "nombre", "effectif", "pourcentage", "taux", "salaire",
        "fise", "fisa", "fisea", "nb rep", "reponse",
    )
    return any(marker in combined for marker in numeric_markers)


def parse_empty_structured_table_schema(
    source: Path,
    page_no: int,
    page: pdfplumber.page.Page,
    lines: Sequence[dict[str, Any]],
    table: Any,
    matrix: Sequence[Sequence[str]],
    anchor: Anchor | None,
    region: tuple[float, float] | None,
) -> list[Record]:
    """Crée uniquement le schéma d'un tableau métier entièrement vide.

    Aucune fausse ligne n'est créée : les Records sont marqués schema_only et
    servent uniquement à créer les colonnes SQLite.
    """
    layout = infer_empty_structured_table_layout(matrix)
    if layout is None:
        return []

    padded = layout["matrix"]
    data_start = layout["data_start"]
    header_rows = layout["header_rows"]
    paths = layout["paths"]
    row_labels = layout["row_labels"]

    code = anchor.code if anchor else (table_codes(matrix)[0] if table_codes(matrix) else "")
    if not code:
        return []

    left_header_parts: list[str] = []
    for r in header_rows:
        for c in range(data_start):
            value = clean_text(padded[r][c])
            if value and not CODE_ANY_RE.search(value.replace(" ", "")):
                left_header_parts.append(value)

    left_header_parts = dedupe_parts(left_header_parts)
    table_title = left_header_parts[0] if left_header_parts else ""
    section_title = clean_text(" | ".join(left_header_parts))
    if not section_title and anchor and region:
        section_title = title_for_anchor(
            page, anchor, region, cutoff_x=max(55.0, float(table.bbox[0]) - 2.0)
        )

    y = anchor.center if anchor else float(table.bbox[1])
    base = record_base(source, page_no, lines, y, code, section_title)
    records: list[Record] = []
    headers_by_column = [
        list(paths[column])
        for column in layout["business_columns"]
        if column < len(paths)
    ]
    schema_numeric_expected = structured_empty_table_expects_numeric(
        headers_by_column, row_labels
    )

    for column in layout["business_columns"]:
        headers = list(paths[column]) if column < len(paths) else []
        headers = dedupe_parts(headers)
        if not headers:
            continue
        records.append(Record(
            **base,
            table_title=table_title,
            column_group=" > ".join(headers[:-1]),
            column_name=headers[-1],
            column_path=" > ".join(headers),
            column_headers_json=json.dumps(headers, ensure_ascii=False),
            value_text="",
            value_num=None,
            value_type="blank",
            extraction_method="pdfplumber_table_schema",
            confidence=0.94,
            raw_context=json.dumps(
                {
                    "table_bbox": table.bbox,
                    "row": layout["first_data"],
                    "column": column,
                    "schema_only": True,
                    "structured_empty_table": True,
                    "row_labels": row_labels,
                    "data_start": data_start,
                    "schema_numeric_expected": schema_numeric_expected,
                },
                ensure_ascii=False,
            ),
        ))

    return records


STRUCTURAL_SCALAR_WORDS = {
    "h", "f", "homme", "hommes", "femme", "femmes", "total",
    "fise", "fisa", "fisea", "fc", "vae", "nombre", "nb",
    "rep", "reponse", "reponses", "avec", "sans", "prime",
}

def looks_like_structural_scalar_artifact(value: Any) -> bool:
    """Détecte un faux scalaire composé uniquement d'en-têtes de tableau."""
    folded = fold_text(value)
    words = re.findall(r"[a-z0-9]+", folded)
    if len(words) < 2:
        return False
    return all(word in STRUCTURAL_SCALAR_WORDS for word in words)

def infer_count_table_layout(
    matrix: Sequence[Sequence[str]],
    context: HeaderContext | None = None,
) -> dict[str, Any] | None:
    """Lit les blocs Nombre/Réponses par leurs séparateurs, jamais par leurs valeurs.

    Les limites d'un en-tête externe simple ne sont pas celles des blocs
    locaux H/F/Total. Les voies et sous-groupes vides conservent leur place.
    """
    for r, row in enumerate(matrix):
        markers = []
        for c, value in enumerate(row):
            marker = re.sub(r"[\s.]", "", clean_text(value).casefold())
            if marker in {"nb", "nbrép", "nbrep", "nombre", "nbréponses", "nbreponses"}:
                markers.append((c, "Nombre de réponses" if "rep" in marker or "rép" in marker else "Nombre"))
        if r == 0 or len(markers) < 2 or not any(role == "Nombre de réponses" for _, role in markers):
            continue
        if context and len(context.spans) != len(markers):
            continue
        paths = {}
        for index, (start, role) in enumerate(markers):
            end = markers[index + 1][0] if index + 1 < len(markers) else len(row)
            prefix = list(context.spans[index].headers) if context else [role]
            if context and prefix[-1].casefold() != role.casefold():
                break
            for c in range(start + 1, end):
                headers = dedupe_parts(clean_text(matrix[h][c]) for h in range(r) if c < len(matrix[h]))
                subgroup = headers[-1] if headers else ""
                if subgroup.casefold() not in {"h", "f", "hommes", "femmes", "total"}:
                    break
                paths[c] = [*prefix, subgroup]
            else:
                continue
            break
        else:
            if paths:
                return {"paths": paths, "data_rows": list(range(r, len(matrix)))}
    return None


def parse_table_records(
    source: Path,
    page_no: int,
    page: pdfplumber.page.Page,
    lines: Sequence[dict[str, Any]],
    table: Any,
    matrix: Sequence[Sequence[str]],
    anchor: Anchor | None,
    region: tuple[float, float] | None,
    external_context: HeaderContext | None,
    include_empty: bool,
) -> list[Record]:
    salary_records = parse_salary_table_records(
        source=source,
        page_no=page_no,
        page=page,
        lines=lines,
        table=table,
        matrix=matrix,
        anchor=anchor,
        region=region,
        include_empty=include_empty,
    )
    if salary_records is not None:
        return salary_records

    count_layout = infer_count_table_layout(matrix, external_context) if external_context else None
    drows = count_layout["data_rows"] if count_layout else data_rows(matrix)
    if not drows:
        schema_records = parse_empty_structured_table_schema(
            source=source,
            page_no=page_no,
            page=page,
            lines=lines,
            table=table,
            matrix=matrix,
            anchor=anchor,
            region=region,
        )
        if schema_records:
            return schema_records
        return []
    first_data = min(drows)
    ncols = max(len(r) for r in matrix)
    # Première colonne contenant une valeur sur les lignes de données.
    starts = []
    for r in drows:
        for c, cell in enumerate(matrix[r]):
            if looks_like_value(cell):
                starts.append(c)
                break
    data_start = (min(count_layout["paths"]) if count_layout else
                  min(starts) if starts else 1)
    header_rows = list(range(first_data))
    paths = filled_header_paths(matrix, header_rows, data_start)

    codes = table_codes(matrix)
    code = codes[0] if codes else (anchor.code if anchor else "")
    title = ""
    if anchor and region and table.bbox[0] > 90:
        cutoff = max(55, table.bbox[0] - 4)
        title = title_for_anchor(page, anchor, region, cutoff)

    # Titre présent dans les colonnes descriptives du tableau.
    first_row = matrix[first_data]
    pre_parts = [clean_text(x) for x in first_row[:data_start] if clean_text(x)]
    pre_parts_no_code = [remove_code_prefix(x, code) for x in pre_parts]
    pre_parts_no_code = [x for x in pre_parts_no_code if x and x.casefold() not in ROW_LABELS]
    if not title and pre_parts_no_code:
        title = clean_text(" ".join(pre_parts_no_code[:-1] or pre_parts_no_code))

    # Titre de tableau : cellules d'en-tête situées avant les colonnes de données.
    table_title_parts: list[str] = []
    for r in header_rows:
        for c in range(min(data_start, len(matrix[r]))):
            cell = clean_text(matrix[r][c])
            if cell and not CODE_ANY_RE.search(cell.replace(" ", "")):
                table_title_parts.append(cell)
    table_title = clean_text(" | ".join(dedupe_parts(table_title_parts)))

    records: list[Record] = []
    intervals = table_column_intervals(table, ncols)
    current_code = code
    current_title = title

    # Les sous-tableaux emploi/salaire ont des groupes séparés par « Nb. rép. ».
    # Les séquences de cellules numériques sont appariées, dans l'ordre, aux
    # chemins de l'en-tête externe (FISE/Nombre, FISE/Réponses, etc.).
    run_external_headers: dict[int, list[str]] = {}
    if external_context and drows and not count_layout:
        probe = list(matrix[drows[0]]) + [""] * (ncols - len(matrix[drows[0]]))
        runs: list[list[int]] = []
        current: list[int] = []
        for c in range(data_start, ncols):
            if looks_like_value(probe[c]):
                current.append(c)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        ext_paths = [list(s.headers) for s in external_context.spans]
        if len(runs) == len(ext_paths):
            for cols, hp in zip(runs, ext_paths):
                for c in cols:
                    run_external_headers[c] = hp

    for r in drows:
        row = list(matrix[r]) + [""] * (ncols - len(matrix[r]))
        row_codes = table_codes([row])
        if row_codes:
            current_code = row_codes[0]
        pre = [clean_text(x) for x in row[:data_start] if clean_text(x)]
        pre_clean = []
        for p in pre:
            p2 = remove_code_prefix(p, current_code)
            if p2:
                pre_clean.append(p2)

        row_name = ""
        row_group = ""
        if row_codes:
            if len(pre_clean) >= 2:
                candidate = pre_clean[-1]
                if candidate.casefold() in ROW_LABELS or len(candidate.split()) <= 4:
                    row_name = candidate
                    current_title = clean_text(" ".join(pre_clean[:-1])) or current_title
                else:
                    current_title = clean_text(" ".join(pre_clean))
            elif len(pre_clean) == 1:
                candidate = pre_clean[0]
                if len(candidate.split()) > 4 or candidate.lower().startswith(("nombre ", "taux ", "durée ", "pourcentage ", "salaire ")):
                    current_title = candidate
                else:
                    row_name = candidate
        else:
            row_name = pre_clean[-1] if pre_clean else ""
            row_group = clean_text(" | ".join(pre_clean[:-1]))
        if row_group and current_title and row_group.casefold() in current_title.casefold():
            row_group = ""

        row_base = record_base(source, page_no, lines, anchor.center if anchor else table.bbox[1], current_code, current_title)

        row_records = []
        for c in (count_layout["paths"] if count_layout else range(data_start, ncols)):
            raw_value = clean_text(row[c])
            if not raw_value and not include_empty and not count_layout:
                continue
            if raw_value and not looks_like_value(raw_value) and not include_empty:
                continue
            token = numeric_token(raw_value)
            value = token if token is not None else raw_value

            header_parts = (count_layout["paths"][c] if count_layout
                            else dedupe_parts(paths[c] if c < len(paths) else []))
            # Ajoute les en-têtes externes (FISE / nombre de réponses, etc.).
            if count_layout:
                pass
            elif c in run_external_headers:
                header_parts = dedupe_parts([*run_external_headers[c], *header_parts])
            elif c < len(intervals) and external_context:
                x0, x1 = intervals[c]
                cx = (float(x0) + float(x1)) / 2
                # En cas de chevauchement, préfère la plage la plus étroite.
                matches = [span for span in external_context.spans if span.x0 <= cx <= span.x1]
                if matches:
                    span = min(matches, key=lambda sp: sp.x1 - sp.x0)
                    header_parts = dedupe_parts([*span.headers, *header_parts])
            if not header_parts:
                header_parts = ["Valeur"]
            column_name = header_parts[-1]
            column_group = " > ".join(header_parts[:-1])
            path = " > ".join(header_parts)

            rec = Record(
                **row_base,
                table_title=table_title,
                row_group=row_group,
                row_name=row_name,
                column_group=column_group,
                column_name=column_name,
                column_path=path,
                column_headers_json=json.dumps(header_parts, ensure_ascii=False),
                value_text=value,
                value_num=parse_number(value),
                value_type=value_type(value),
                extraction_method=("pdfplumber_count_blocks" if count_layout else "pdfplumber_table"),
                confidence=(0.90 if not value else
                            0.82 if external_context and not count_layout and c not in run_external_headers
                            else 0.96),
                raw_context=json.dumps({"table_bbox": table.bbox, "row": r, "column": c,
                                        "schema_only": not bool(raw_value),
                                        "schema_numeric_expected": bool(count_layout)}, ensure_ascii=False),
            )
            records.append(rec)
            row_records.append(rec)

        if count_layout:
            response_groups: dict[str, dict[str, Record]] = defaultdict(dict)
            for rec in row_records:
                if "Nombre de réponses" in rec.column_group:
                    response_groups[rec.column_group][rec.column_name.casefold()] = rec
            for index, (group, components) in enumerate(response_groups.items()):
                if "total" in components:
                    continue
                h = components.get("h") or components.get("hommes")
                f = components.get("f") or components.get("femmes")
                if h is None or f is None:
                    continue
                total = h.value_num + f.value_num if h.value_num is not None and f.value_num is not None else None
                context = parse_raw_context(h.raw_context)
                context.update(column=ncols + index, schema_only=total is None,
                               correction="total_repondants_calcule_h_plus_f",
                               derived_from_columns=[parse_raw_context(v.raw_context)["column"] for v in (h, f)])
                records.append(replace(
                    h, column_name="Total", column_path=group + " > Total",
                    column_headers_json=json.dumps([*group.split(" > "), "Total"], ensure_ascii=False),
                    value_num=total, value_text="" if total is None else str(total),
                    value_type="blank" if total is None else "number",
                    extraction_method="pdfplumber_count_derived_total",
                    raw_context=json.dumps(context, ensure_ascii=False),
                ))
    return records


def context_span_expects_numeric(span: HeaderSpan) -> bool:
    """Déduit depuis l'en-tête si une colonne attend une valeur numérique."""
    label = " > ".join(span.headers)
    folded = fold_text(label)
    return any(
        token in folded
        for token in (
            "nombre", "effectif", "salaire", "pourcentage", "taux",
            "median", "moyenne", "total", "nb rep",
        )
    )


def parse_context_layout(
    source: Path,
    page_no: int,
    page: pdfplumber.page.Page,
    lines: Sequence[dict[str, Any]],
    anchor: Anchor,
    region: tuple[float, float],
    context: HeaderContext,
) -> list[Record]:
    """Extrait une rubrique placée sous un en-tête de colonnes partagé.

    Point important : l'en-tête externe décrit le *schéma* même lorsqu'une
    école n'a renseigné aucune valeur. Dans ce cas on produit des cellules
    ``schema_only`` vides. Elles servent à construire les colonnes SQLite,
    mais ne sont pas insérées comme lignes de données sauf si --include-empty
    est explicitement demandé.

    Cela évite aussi de retomber sur le parseur scalaire, qui pouvait prendre
    un intitulé long de rubrique pour une valeur métier.
    """
    if not context.spans:
        return []
    data_x0 = min(s.x0 for s in context.spans)
    words = words_in_box(page, 52, region[0], page.width - 3, region[1])
    groups = group_words_by_line(words)
    value_lines: list[dict[str, Any]] = []

    # Une zone intermédiaire existe dans les mini-tableaux II.1 (Nom du test / Niveau requis).
    sub_split = 200.0 if data_x0 > 300 else data_x0
    all_main: list[str] = []
    any_sub = False

    for group in groups:
        y = sum((float(w["top"]) + float(w["bottom"])) / 2 for w in group) / len(group)
        main_words = [w for w in group if 52 <= (float(w["x0"]) + float(w["x1"])) / 2 < sub_split]
        sub_words = [w for w in group if sub_split <= (float(w["x0"]) + float(w["x1"])) / 2 < data_x0]
        vals: list[str] = []
        for span in context.spans:
            sw = [
                w
                for w in group
                if span.x0
                <= (float(w["x0"]) + float(w["x1"])) / 2
                <= span.x1
            ]
            raw = line_text(sw)
            # Les fins d'en-têtes multi-lignes (par ex. « l'enquête »)
            # peuvent géométriquement tomber dans la région de la première
            # rubrique. Dans une colonne numérique, un texte non numérique
            # n'est jamais une valeur et doit être ignoré.
            if raw and context_span_expects_numeric(span) and parse_number(raw) is None:
                raw = ""
            vals.append(raw)
        main = re.sub(r"^[IVX]+(?:[.\-][A-Za-z0-9]+)+\s*", "", line_text(main_words)).strip()
        sub = line_text(sub_words)
        if main:
            all_main.append(main)
        if sub:
            any_sub = True
        if any(vals):
            value_lines.append({"y": y, "main": main, "sub": sub, "values": vals})

    if not value_lines:
        # La ligne est vide mais les colonnes sont connues grâce à l'en-tête
        # partagé. Le titre reste une métadonnée de section, jamais une colonne.
        section_title = title_for_anchor(
            page, anchor, region, cutoff_x=max(55.0, data_x0 - 2.0)
        )
        base = record_base(
            source, page_no, lines, anchor.center, anchor.code, section_title
        )
        records: list[Record] = []
        for span_index, span in enumerate(context.spans):
            headers = list(span.headers) or ["Valeur"]
            records.append(Record(
                **base,
                column_group=" > ".join(headers[:-1]),
                column_name=headers[-1],
                column_path=" > ".join(headers),
                column_headers_json=json.dumps(headers, ensure_ascii=False),
                value_text="",
                value_num=None,
                value_type="blank",
                extraction_method="pdfplumber_layout_schema",
                confidence=0.90,
                raw_context=json.dumps(
                    {
                        "region": region,
                        "y": anchor.center,
                        "schema_only": True,
                        "span_index": span_index,
                    },
                    ensure_ascii=False,
                ),
            ))
        return records

    # Rattache les lignes de libellé sans valeur au point de données le plus proche.
    all_line_data = []
    for group in groups:
        y = sum((float(w["top"]) + float(w["bottom"])) / 2 for w in group) / len(group)
        main_words = [w for w in group if 52 <= (float(w["x0"]) + float(w["x1"])) / 2 < sub_split]
        sub_words = [w for w in group if sub_split <= (float(w["x0"]) + float(w["x1"])) / 2 < data_x0]
        main = re.sub(r"^[IVX]+(?:[.\-][A-Za-z0-9]+)+\s*", "", line_text(main_words)).strip()
        sub = line_text(sub_words)
        if main or sub:
            all_line_data.append((y, main, sub))

    for y, main, sub in all_line_data:
        nearest = min(range(len(value_lines)), key=lambda i: abs(value_lines[i]["y"] - y))
        if abs(value_lines[nearest]["y"] - y) <= 18:
            if main and main not in value_lines[nearest]["main"]:
                value_lines[nearest]["main"] = clean_text(value_lines[nearest]["main"] + " " + main)
            if sub and sub not in value_lines[nearest]["sub"]:
                value_lines[nearest]["sub"] = clean_text(value_lines[nearest]["sub"] + " " + sub)

    sub_count = sum(1 for v in value_lines if clean_text(v["sub"]))
    known_sub = any(clean_text(v["sub"]).casefold().strip(":") in LIST_FIELD_LABELS for v in value_lines)
    use_sub = known_sub or sub_count >= 2
    if not use_sub:
        for item in value_lines:
            item["main"] = clean_text(item["main"] + " " + item["sub"])
            item["sub"] = ""
    if use_sub:
        section_title = clean_text(" ".join(dedupe_parts(all_main)))
    elif len(value_lines) == 1:
        # Pour une rubrique à une seule ligne, la zone descriptive située à
        # gauche des colonnes est plus fiable que la reconstruction mot à mot.
        section_title = title_for_anchor(
            page, anchor, region, cutoff_x=max(55.0, data_x0 - 2.0)
        ) or value_lines[0]["main"]
    else:
        section_title = clean_text(" / ".join(dedupe_parts(v["main"] for v in value_lines)))

    base = record_base(source, page_no, lines, anchor.center, anchor.code, section_title)
    records: list[Record] = []
    for item in value_lines:
        row_name = item["sub"] if use_sub else (item["main"] if len(value_lines) > 1 else "")
        for span_index, (span, raw) in enumerate(zip(context.spans, item["values"])):
            headers = list(span.headers) or ["Valeur"]
            cleaned_raw = clean_text(raw)
            value = numeric_token(cleaned_raw) or cleaned_raw
            schema_only = not bool(cleaned_raw)
            records.append(Record(
                **base,
                row_name=row_name,
                column_group=" > ".join(headers[:-1]),
                column_name=headers[-1],
                column_path=" > ".join(headers),
                column_headers_json=json.dumps(headers, ensure_ascii=False),
                value_text=value,
                value_num=parse_number(value),
                value_type=value_type(value),
                extraction_method=(
                    "pdfplumber_layout_schema"
                    if schema_only
                    else "pdfplumber_layout_columns"
                ),
                confidence=0.90 if schema_only else 0.91,
                raw_context=json.dumps(
                    {
                        "region": region,
                        "y": item["y"],
                        "schema_only": schema_only,
                        "span_index": span_index,
                    },
                    ensure_ascii=False,
                ),
            ))
    return records


def scalar_region_from_edges(page: pdfplumber.page.Page, anchor: Anchor, fallback: tuple[float, float]) -> tuple[float, float]:
    """Utilise les séparateurs horizontaux du formulaire pour les lignes simples."""
    ys: list[float] = []
    for edge in page.edges:
        if edge.get("orientation") != "h":
            continue
        width = abs(float(edge.get("x1", 0)) - float(edge.get("x0", 0)))
        if width < page.width * 0.45:
            continue
        ys.append(float(edge["top"]))
    ys = sorted(set(round(y, 2) for y in ys))
    above = [y for y in ys if y < anchor.center - 1]
    below = [y for y in ys if y > anchor.center + 1]
    if above and below:
        lo, hi = max(above), min(below)
        if lo < 10 and fallback[0] > 20:
            return fallback
        if 12 <= hi - lo <= 180:
            return (max(0.0, lo + 1), min(page.height, hi - 1))
    return fallback


def parse_scalar_layout(
    source: Path,
    page_no: int,
    page: pdfplumber.page.Page,
    lines: Sequence[dict[str, Any]],
    anchor: Anchor,
    region: tuple[float, float],
) -> list[Record]:
    words = words_in_box(page, 52, region[0], page.width - 3, region[1])
    if not words:
        return []

    # Un millésime de campagne peut apparaître au-dessus de la cellule valeur.
    # S'il n'est pas aligné avec la ligne de la question, il s'agit d'un en-tête
    # de formulaire et non du contenu métier.
    filtered_words = []
    for word in words:
        word_text = clean_text(word.get("text", ""))
        cy = (float(word["top"]) + float(word["bottom"])) / 2
        if (
            is_academic_year_marker(word_text)
            and abs(cy - anchor.center) > 6.0
        ):
            continue
        filtered_words.append(word)
    words = filtered_words

    if not words:
        return []

    # Détecte d'abord les sous-champs explicites (Adresse / Code postal / Ville).
    nested_split = 195.0
    nested_right = [
        w for w in words
        if (float(w["x0"]) + float(w["x1"])) / 2 >= nested_split
    ]
    nested_lines_probe = [
        line_text(g).casefold().strip(":")
        for g in group_words_by_line(nested_right)
    ]
    has_nested_fields = any(t in LIST_FIELD_LABELS for t in nested_lines_probe)

    # Pour une valeur scalaire simple, la séparation n'est plus figée à x=250 :
    # on utilise la position réelle du segment de valeur lorsqu'il est détectable.
    split_x = (
        nested_split
        if has_nested_fields
        else scalar_value_split_x(words, default_split=250.0)
    )

    right = [
        w for w in words
        if (float(w["x0"]) + float(w["x1"])) / 2 >= split_x
    ]

    left = [
        w for w in words
        if 52 <= (float(w["x0"]) + float(w["x1"])) / 2 < split_x
    ]

    title_parts = []
    for group in group_words_by_line(left):
        t = re.sub(r"^[IVX]+(?:[.\-][A-Za-z0-9]+)+\s*", "", line_text(group)).strip()
        if t:
            title_parts.append(t)
    title = clean_text(" ".join(dedupe_parts(title_parts)))
    base = record_base(source, page_no, lines, anchor.center, anchor.code, title)

    if not right:
        return [Record(
            **base,
            value_text="",
            value_num=None,
            value_type="blank",
            extraction_method="pdfplumber_scalar",
            confidence=0.55,
            raw_context=json.dumps({"region": region}, ensure_ascii=False),
        )]

    right_lines = [(sum(float(w["top"]) for w in g) / len(g), line_text(g)) for g in group_words_by_line(right)]
    # Sous-champs explicites : Adresse / Code postal / Ville, etc.
    records: list[Record] = []
    current_label = ""
    current_values: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_values
        if current_label:
            value = clean_text(" ".join(current_values))
            records.append(Record(
                **base,
                row_name=current_label,
                column_name="Valeur",
                column_path="Valeur",
                column_headers_json='["Valeur"]',
                value_text=numeric_token(value) or value,
                value_num=parse_number(numeric_token(value) or value),
                value_type=value_type(numeric_token(value) or value),
                extraction_method="pdfplumber_scalar_subfield",
                confidence=0.90,
                raw_context=json.dumps({"region": region}, ensure_ascii=False),
            ))
        current_label = ""
        current_values = []

    for _, text in right_lines:
        key = text.casefold().strip(":")
        if key in LIST_FIELD_LABELS:
            flush()
            current_label = text.strip(":")
        elif current_label:
            current_values.append(text)
    flush()
    if records:
        return records

    value = clean_text(" ".join(text for _, text in right_lines))
    value = numeric_token(value) or value

    # Garde-fou générique : une suite telle que "Hommes Femmes Total" ou
    # "FISE FISA FISEA" est un en-tête de tableau, jamais une valeur scalaire.
    # Si un nouveau layout échappe au parseur de tableaux, on préfère ne créer
    # aucune fausse colonne plutôt que d'injecter cet artefact dans SQLite.
    if looks_like_structural_scalar_artifact(value):
        return []

    return [Record(
        **base,
        column_name="Valeur",
        column_path="Valeur",
        column_headers_json='["Valeur"]',
        value_text=value,
        value_num=parse_number(value),
        value_type=value_type(value),
        extraction_method="pdfplumber_scalar",
        confidence=0.88 if value else 0.58,
        raw_context=json.dumps({"region": region}, ensure_ascii=False),
    )]


def extract_pdf(source: Path, include_empty: bool = False) -> list[Record]:
    records: list[Record] = []
    with pdfplumber.open(source) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            anchors, lines = extract_anchors(page)
            regions = anchor_regions(anchors, page.height)
            tables = valid_tables(page)

            # Prépare les tableaux et les contextes d'en-têtes.
            infos = []
            for table in tables:
                matrix = table_matrix(table)
                drows = data_rows(matrix)
                anchor = anchor_for_table(table, anchors, regions)
                salary_candidate = is_salary_table_candidate(matrix, anchor)
                empty_structured_candidate = (
                    not drows
                    and not salary_candidate
                    and (is_empty_structured_table_candidate(matrix)
                         or infer_count_table_layout(matrix) is not None)
                )
                infos.append({
                    "table": table,
                    "matrix": matrix,
                    "drows": drows,
                    "anchor": anchor,
                    "salary_candidate": salary_candidate,
                    "empty_structured_candidate": empty_structured_candidate,
                })

            contexts: list[HeaderContext] = []
            # Un tableau salaire vide est un tableau métier, pas un simple en-tête.
            header_infos = [
                i for i in infos
                if (
                    not i["drows"]
                    and not i["salary_candidate"]
                    and not i["empty_structured_candidate"]
                )
            ]
            for pos, info in enumerate(header_infos):
                table = info["table"]
                matrix = info["matrix"]
                next_y = header_infos[pos + 1]["table"].bbox[1] if pos + 1 < len(header_infos) else page.height
                anchor = info["anchor"]
                pref = ""
                source_code = ""
                if anchor and any("voie" in clean_text(c).casefold() for row in matrix for c in row):
                    pref = code_prefix(anchor.code)
                    source_code = anchor.code
                contexts.append(header_context_from_table(table, matrix, next_y, pref, source_code))

            assigned_codes: set[str] = set()
            # Tableaux avec données.
            for info in infos:
                # Les VIII-D*-7 doivent être analysés même sans aucun chiffre :
                # leur géométrie suffit à définir le schéma H/F/Total.
                if (
                    not info["drows"]
                    and not info["salary_candidate"]
                    and not info["empty_structured_candidate"]
                ):
                    continue
                table = info["table"]
                anchor = info["anchor"]
                region = regions.get(anchor.code) if anchor else None
                context = context_for_anchor(anchor, contexts) if anchor else None
                recs = parse_table_records(
                    source, page_idx, page, lines, table, info["matrix"], anchor, region, context, include_empty
                )
                records.extend(recs)
                if recs:
                    assigned_codes.update(r.section_code for r in recs if r.section_code)
                    if anchor:
                        assigned_codes.add(anchor.code)

            # Les tableaux « Voie » sont à la fois des valeurs et des en-têtes pour les questions suivantes.
            for info in header_infos:
                anchor = info["anchor"]
                if not anchor or anchor.code in assigned_codes:
                    continue
                flat = [clean_text(c) for row in info["matrix"] for c in row if clean_text(c)]
                if not flat or "voie" not in title_for_anchor(page, anchor, regions[anchor.code], info["table"].bbox[0]).casefold():
                    continue
                title = title_for_anchor(page, anchor, regions[anchor.code], info["table"].bbox[0]) or "Voie"
                base = record_base(source, page_idx, lines, anchor.center, anchor.code, title)
                for n, value in enumerate(flat, start=1):
                    records.append(Record(
                        **base,
                        row_name=str(n),
                        column_name="Voie",
                        column_path="Voie",
                        column_headers_json='["Voie"]',
                        value_text=value,
                        value_type=value_type(value),
                        extraction_method="pdfplumber_header_values",
                        confidence=0.93,
                        raw_context=json.dumps({"table_bbox": info["table"].bbox}, ensure_ascii=False),
                    ))
                assigned_codes.add(anchor.code)

            # Questions simples et tableaux sans bordures.
            for anchor in anchors:
                if anchor.code in assigned_codes:
                    continue
                region = regions[anchor.code]
                context = context_for_anchor(anchor, contexts)
                recs = parse_context_layout(source, page_idx, page, lines, anchor, region, context) if context else []
                if not recs:
                    scalar_region = scalar_region_from_edges(page, anchor, region)
                    recs = parse_scalar_layout(source, page_idx, page, lines, anchor, scalar_region)
                records.extend(recs)

    return records


SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS cti_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    source_page INTEGER NOT NULL,
    major_section_code TEXT,
    major_section_title TEXT,
    subsection_title TEXT,
    population_context TEXT,
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
    value_type TEXT,
    extraction_method TEXT,
    confidence REAL,
    raw_context TEXT
);
CREATE INDEX IF NOT EXISTS idx_cti_code ON cti_values(section_code);
CREATE INDEX IF NOT EXISTS idx_cti_source ON cti_values(source_file, source_page);
CREATE INDEX IF NOT EXISTS idx_cti_dimensions ON cti_values(section_code, row_name, column_path);
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
            f"INSERT INTO cti_values ({','.join(FIELDS)}) VALUES ({placeholders})",
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
            writer.writerow(asdict(r))


def write_review_csv(records: Sequence[Record], path: Path) -> None:
    review = [r for r in records if r.confidence < 0.80 or not r.value_text]
    write_csv(review, path)


def iter_pdfs(inputs: Sequence[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for raw in inputs:
        p = Path(raw)
        candidates = sorted(p.rglob("*.pdf")) if p.is_dir() else [p]
        for c in candidates:
            c = c.resolve()
            if c.suffix.lower() == ".pdf" and c.exists() and c not in seen:
                seen.add(c)
                yield c



# =============================================================================
# BASE STRUCTURÉE PAR SECTION
# =============================================================================

import hashlib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import field


COMMON_SECTION_COLUMNS = {
    "id", "document_id", "school_key", "source_type", "year", "city",
    "school_name", "source_file", "source_page", "major_section_code",
    "major_section_title", "subsection_title", "population_context", "section_code",
    "section_title", "table_title", "row_group", "row_name",
    "source_row_key", "confidence_min", "confidence_max",
    "extraction_methods", "source_value_ids_json",
}

GENERIC_VALUE_LABELS = {"", "valeur", "value", "donnee", "donnée"}
MAX_IDENTIFIER_LENGTH = 96


@dataclass(frozen=True)
class DocumentInfo:
    id: int
    school_key: str
    source_type: str
    year: int | None
    city: str
    school_name: str
    source_file: str
    page_count: int


@dataclass
class SectionCell:
    raw_id: int
    document: DocumentInfo
    record: Record
    row_key: str = ""
    column_order: tuple[Any, ...] = field(default_factory=tuple)
    original_field_label: str = ""
    semantic_field_base: str = ""


@dataclass
class LogicalSectionRow:
    cells: list[SectionCell]


@dataclass
class SectionFieldSpec:
    sql_name: str
    sqlite_type: str
    value_role: str
    semantic_base: str
    occurrence_index: int
    original_labels: list[str]
    non_null_count: int
    numeric_count: int
    sample_values: list[str]
    unit: str
    measure_kind: str
    aggregation_hint: str
    description: str


@dataclass
class SectionFieldBinding:
    numeric_spec: SectionFieldSpec | None = None
    text_spec: SectionFieldSpec | None = None


@dataclass(frozen=True)
class ExtractionSummary:
    output_path: Path
    document_count: int
    section_count: int
    row_count: int
    record_count: int


def fold_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", clean_text(value))
    return "".join(
        char for char in normalized if not unicodedata.combining(char)
    ).casefold()


def short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def sql_slug(value: str, fallback: str = "valeur") -> str:
    normalized = unicodedata.normalize("NFKD", clean_text(value))
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    ).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    if not normalized:
        normalized = fallback
    if normalized[0].isdigit():
        normalized = "valeur_" + normalized
    if len(normalized) > MAX_IDENTIFIER_LENGTH:
        suffix = short_hash(normalized, 8)
        normalized = normalized[: MAX_IDENTIFIER_LENGTH - 9] + "_" + suffix
    return normalized


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def semantic_label(value: str) -> str:
    text_value = clean_text(value)
    text_value = re.sub(r"\b20\d{2}\s*[-–/]\s*20\d{2}\b", " année ", text_value)
    text_value = re.sub(r"\b20\d{2}\b", " année ", text_value)
    return sql_slug(text_value)


def detect_document_info(path: Path, document_id: int) -> DocumentInfo:
    stem = path.stem
    year_match = re.search(r"(20\d{2})", stem)
    year = int(year_match.group(1)) if year_match else None
    school_part = re.split(r"-cti-", stem, flags=re.IGNORECASE)[0]
    school_part = re.sub(r"\(\d+\)$", "", school_part)
    key = sql_slug(school_part, "inconnue").upper()
    city = school_part.replace("_", " ").replace("-", " ").strip()
    city = clean_text(city).title()

    try:
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            first_text = clean_text(pdf.pages[0].extract_text() or "")
    except Exception:
        page_count = 0
        first_text = ""

    campaign = re.search(r"\bcampagne\s*[:\-]?\s*(20\d{2})\b", first_text, re.I)
    if campaign:
        year = int(campaign.group(1))

    school_name = city
    match = re.search(
        r"Par la direction de\s*:\s*(.*?)\s+Académie\s*:",
        first_text,
        flags=re.IGNORECASE,
    )
    if match:
        school_name = clean_text(match.group(1))

    return DocumentInfo(
        id=document_id,
        school_key=key,
        source_type="CTI",
        year=year,
        city=city,
        school_name=school_name,
        source_file=path.name,
        page_count=page_count,
    )


def parse_raw_context(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def is_schema_only_record(record: Record) -> bool:
    """Vrai pour une cellule vide conservée uniquement pour décrire le schéma."""
    return bool(parse_raw_context(record.raw_context).get("schema_only"))


MISSING_TEXT_MARKERS = {"", "/", "-", "–", "—"}


def is_meaningful_text_value(value: Any) -> bool:
    return clean_text(value).strip().casefold() not in MISSING_TEXT_MARKERS


def record_has_business_value(record: Record) -> bool:
    return record.value_num is not None or is_meaningful_text_value(record.value_text)


def logical_row_has_business_value(row: LogicalSectionRow) -> bool:
    return any(record_has_business_value(cell.record) for cell in row.cells)


def rounded_sequence(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return ",".join(
        f"{float(item):.2f}" if isinstance(item, (int, float)) else clean_text(item)
        for item in value
    )


def assign_cell_position(cell: SectionCell) -> None:
    context = parse_raw_context(cell.record.raw_context)

    if "table_bbox" in context and "row" in context:
        identity = (
            "table:"
            + rounded_sequence(context.get("table_bbox"))
            + ":row:"
            + str(context.get("row"))
        )
        column = context.get("column", 0)
        try:
            column_order = int(column)
        except (TypeError, ValueError):
            column_order = 0
        cell.column_order = (0, column_order, cell.raw_id)
    elif "region" in context and "y" in context:
        identity = (
            "layout:"
            + rounded_sequence(context.get("region"))
            + ":y:"
            + clean_text(context.get("y"))
        )
        cell.column_order = (
            1,
            fold_text(cell.record.column_group),
            fold_text(cell.record.column_name),
            cell.raw_id,
        )
    elif "region" in context:
        identity = "region:" + rounded_sequence(context.get("region"))
        cell.column_order = (
            2,
            fold_text(cell.record.column_group),
            fold_text(cell.record.column_name),
            cell.raw_id,
        )
    else:
        identity = "fallback:" + "|".join(
            [
                cell.record.table_title,
                cell.record.row_group,
                cell.record.row_name,
                str(cell.raw_id),
            ]
        )
        cell.column_order = (3, cell.raw_id)

    cell.row_key = short_hash(
        f"{cell.document.id}|{cell.record.source_page}|"
        f"{cell.record.section_code}|{identity}",
        16,
    )


def field_label(cell: SectionCell, dominant_title: str) -> str:
    label = clean_text(cell.record.column_path) or clean_text(cell.record.column_name)
    if fold_text(label) in GENERIC_VALUE_LABELS:
        label = dominant_title or "valeur"
    context = clean_text(cell.record.section_title) or dominant_title
    # Des formulaires réutilisent l'en-tête « Nombre » pour une mesure dont
    # la nature est définie par la rubrique (notamment les salaires).
    kind = infer_field_semantics(label, context)[1]
    if "salaire" in fold_text(context) and "median" in fold_text(context):
        parts = [clean_text(part) for part in label.split(" > ")]
        if len(parts) == 2 and kind in {"median_salary", "response_count"}:
            label = " > ".join([parts[0], "Nombre de réponses" if kind == "response_count" else "Salaire médian", "Total"])
    return label


def dominant_nonempty(values: Iterable[str]) -> str:
    counter = Counter(clean_text(value) for value in values if clean_text(value))
    return counter.most_common(1)[0][0] if counter else ""


def group_section_rows(cells: Sequence[SectionCell]) -> list[LogicalSectionRow]:
    grouped: dict[tuple[Any, ...], list[SectionCell]] = defaultdict(list)
    for cell in cells:
        grouped[(cell.document.id, cell.record.source_page, cell.row_key)].append(cell)
    return [
        LogicalSectionRow(sorted(group, key=lambda item: item.column_order))
        for _, group in sorted(grouped.items(), key=lambda item: item[0])
    ]


def prepare_section_fields(rows: Sequence[LogicalSectionRow], dominant_title: str) -> None:
    for logical_row in rows:
        for cell in logical_row.cells:
            label = field_label(cell, dominant_title)
            cell.original_field_label = label
            cell.semantic_field_base = semantic_label(label)

    ambiguous: set[str] = set()
    for logical_row in rows:
        labels: dict[str, set[str]] = defaultdict(set)
        for cell in logical_row.cells:
            labels[cell.semantic_field_base].add(cell.original_field_label)
        ambiguous.update(base for base, values in labels.items() if len(values) > 1)

    for logical_row in rows:
        for cell in logical_row.cells:
            if cell.semantic_field_base in ambiguous:
                cell.semantic_field_base = sql_slug(cell.original_field_label)


def unique_column_name(base: str, used: set[str], identity: str) -> str:
    candidate = sql_slug(base)
    if candidate in COMMON_SECTION_COLUMNS:
        candidate = "donnee_" + candidate
    if candidate not in used:
        used.add(candidate)
        return candidate
    candidate = candidate + "_" + short_hash(identity, 8)
    index = 2
    final = candidate
    while final in used:
        final = f"{candidate}_{index}"
        index += 1
    used.add(final)
    return final


def infer_field_semantics(
    label: str, context: str = "", raw_values: Sequence[str] = (),
) -> tuple[str, str, str, str]:
    folded = fold_text(label)

    # L'unité des cellules sources prime sur un libellé générique (Hommes,
    # Femmes, Valeur...). Ne jamais inférer un pourcentage de sa seule valeur.
    numeric_values = [clean_text(value) for value in raw_values
                      if parse_number(value) is not None]
    percent_values = [value for value in numeric_values
                      if re.fullmatch(r"[-+]?\d+(?:[ .]\d{3})*(?:[,.]\d+)?\s*%", value)]
    if percent_values:
        if len(percent_values) != len(numeric_values):
            return (
                "", "mixed_units", "not_aggregatable",
                "Unités sources hétérogènes : pourcentages et nombres sans unité "
                "explicite. Ne pas agréger sans résoudre cette ambiguïté.",
            )
        return (
            "percent", "percentage", "weighted_average_if_counts_available",
            "Pourcentage attesté par le symbole % dans les cellules sources, "
            "exprimé sur 100. Non additionnable entre populations. Recalculer "
            "sur les effectifs ou pondérer par les dénominateurs correspondants ; "
            "sans ces effectifs, aucune proportion globale n'est calculable.",
        )

    # Le rôle de dénominateur prime sur le sujet de la rubrique : des
    # répondants à une question de salaire ne sont pas des montants en EUR.
    if ("nombre de reponses" in folded or "nb rep" in folded
            or re.search(r"\b(nombre|nb)\b.*\b(repondants?|repondu)\b", folded)):
        return (
            "count", "response_count", "sum",
            "Nombre de répondants associé à la mesure.",
        )
    measure_context = folded + " " + fold_text(context)
    if "salaire" in measure_context and "median" in measure_context:
        return (
            "EUR",
            "median_salary",
            "weighted_average_by_response_count",
            "Salaire médian annuel extrait du tableau CTI. Pour une synthèse, "
            "pondérer par le nombre de réponses de la même voie, du même "
            "sous-groupe et de la même catégorie de primes. Le résultat est "
            "une moyenne pondérée des médianes, pas une médiane globale.",
        )
    if "pourcentage" in folded or "taux" in folded or "%" in label:
        return (
            "percent",
            "percentage",
            "weighted_average_if_counts_available",
            "Pourcentage déclaré dans le formulaire.",
        )
    if "nombre" in folded or "effectif" in folded or "total" in folded:
        if re.search(r"\ba\s+\d+(?:\s*[-–]\s*\d+)?\s+mois\b", folded):
            return (
                "count", "count", "sum_within_horizon",
                "Effectif sommable uniquement à horizon d'observation identique. "
                "Les colonnes de périodes différentes décrivent des observations "
                "distinctes et ne sont pas des populations parallèles à additionner.",
            )
        return (
            "count",
            "count",
            "sum",
            "Effectif ou quantité déclarée.",
        )
    return ("", "value", "context_dependent", "Valeur extraite du formulaire CTI.")


def build_section_field_specs(
    rows: Sequence[LogicalSectionRow],
) -> tuple[list[SectionFieldSpec], dict[tuple[str, int], SectionFieldBinding]]:
    values: dict[tuple[str, int], list[SectionCell]] = defaultdict(list)
    labels: dict[tuple[str, int], set[str]] = defaultdict(set)

    for logical_row in rows:
        occurrences: dict[str, int] = defaultdict(int)
        for cell in logical_row.cells:
            base = cell.semantic_field_base
            occurrences[base] += 1
            key = (base, occurrences[base])
            values[key].append(cell)
            labels[key].add(cell.original_field_label)

    used = set(COMMON_SECTION_COLUMNS)
    specs: list[SectionFieldSpec] = []
    bindings: dict[tuple[str, int], SectionFieldBinding] = {}

    for (base, occurrence), cells in sorted(values.items()):
        numeric = [cell for cell in cells if cell.record.value_num is not None]
        textual = [
            cell for cell in cells
            if cell.record.value_num is None
            and is_meaningful_text_value(cell.record.value_text)
        ]
        original_labels = sorted(labels[(base, occurrence)])
        name_base = base if occurrence == 1 else f"{base}_{occurrence}"
        unit, kind, aggregation, description = infer_field_semantics(
            " | ".join(original_labels),
            " | ".join(sorted({cell.record.section_title for cell in cells if cell.record.section_title})),
            [cell.record.value_text for cell in numeric],
        )
        binding = SectionFieldBinding()

        # Une colonne peut être connue par l'en-tête même si toutes ses
        # cellules sont vides. Son type est alors déduit de la sémantique du
        # libellé plutôt que de fabriquer une colonne TEXT arbitraire.
        blank_only = not numeric and not textual
        schema_numeric_expected = blank_only and any(
            bool(parse_raw_context(cell.record.raw_context).get("schema_numeric_expected"))
            for cell in cells
        )
        inferred_numeric = blank_only and (
            schema_numeric_expected
            or kind in {"count", "percentage", "median_salary"}
            or unit in {"count", "percent", "EUR"}
        )

        if numeric or inferred_numeric:
            name = unique_column_name(
                name_base,
                used,
                f"{base}|{occurrence}|numeric|{original_labels}",
            )
            samples = []
            for cell in numeric:
                value = str(cell.record.value_num)
                if value not in samples:
                    samples.append(value)
                if len(samples) == 5:
                    break
            spec = SectionFieldSpec(
                sql_name=name,
                sqlite_type="REAL",
                value_role="numeric",
                semantic_base=base,
                occurrence_index=occurrence,
                original_labels=original_labels,
                non_null_count=len(numeric),
                numeric_count=len(numeric),
                sample_values=samples,
                unit=unit,
                measure_kind=kind,
                aggregation_hint=aggregation,
                description=description,
            )
            specs.append(spec)
            binding.numeric_spec = spec

        if textual or (blank_only and not inferred_numeric):
            text_base = name_base if not numeric else name_base + "_texte_brut"
            name = unique_column_name(
                text_base,
                used,
                f"{base}|{occurrence}|text|{original_labels}",
            )
            samples = []
            for cell in textual:
                value = clean_text(cell.record.value_text)
                if value and value not in samples:
                    samples.append(value)
                if len(samples) == 5:
                    break
            spec = SectionFieldSpec(
                sql_name=name,
                sqlite_type="TEXT",
                value_role="text",
                semantic_base=base,
                occurrence_index=occurrence,
                original_labels=original_labels,
                non_null_count=len(textual),
                numeric_count=0,
                sample_values=samples,
                unit=unit,
                measure_kind=kind,
                aggregation_hint=aggregation,
                description=description,
            )
            specs.append(spec)
            binding.text_spec = spec

        bindings[(base, occurrence)] = binding

    return specs, bindings


def make_section_table_name(code: str, used: set[str]) -> str:
    base = sql_slug("cti_" + code, "cti_section")
    candidate = base
    if candidate in used:
        candidate = base + "_" + short_hash(code, 8)
    used.add(candidate)
    return candidate


def create_section_database_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            school_key TEXT NOT NULL,
            source_type TEXT NOT NULL,
            year INTEGER,
            city TEXT,
            school_name TEXT,
            source_file TEXT NOT NULL UNIQUE,
            page_count INTEGER NOT NULL
        );

        CREATE TABLE section_catalog (
            table_name TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            section_code TEXT NOT NULL,
            major_section_code TEXT,
            major_section_title TEXT,
            section_title TEXT,
            population_context TEXT,
            section_titles_json TEXT NOT NULL,
            table_titles_json TEXT NOT NULL,
            row_granularity TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            dynamic_column_count INTEGER NOT NULL,
            description TEXT NOT NULL,
            search_text TEXT NOT NULL,
            UNIQUE(source_type, section_code)
        );

        CREATE TABLE column_catalog (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            sqlite_type TEXT NOT NULL,
            value_role TEXT NOT NULL,
            semantic_base TEXT NOT NULL,
            occurrence_index INTEGER NOT NULL,
            original_labels_json TEXT NOT NULL,
            non_null_count INTEGER NOT NULL,
            numeric_count INTEGER NOT NULL,
            sample_values_json TEXT NOT NULL,
            unit TEXT NOT NULL,
            measure_kind TEXT NOT NULL,
            aggregation_hint TEXT NOT NULL,
            description TEXT NOT NULL,
            PRIMARY KEY(table_name, column_name),
            FOREIGN KEY(table_name) REFERENCES section_catalog(table_name)
                ON DELETE CASCADE
        );

        CREATE TABLE extraction_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_page INTEGER NOT NULL,
            section_code TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            message TEXT NOT NULL,
            context_json TEXT NOT NULL
        );

        CREATE VIEW vanna_section_catalog AS
        SELECT
            table_name,
            source_type,
            section_code,
            section_title,
            population_context,
            description,
            search_text,
            row_count,
            dynamic_column_count
        FROM section_catalog;
        """
    )


def create_physical_section_table(
    connection: sqlite3.Connection,
    table_name: str,
    fields: Sequence[SectionFieldSpec],
) -> None:
    dynamic = "".join(
        f",\n            {quote_identifier(field.sql_name)} {field.sqlite_type}"
        for field in fields
    )
    connection.execute(
        f"""
        CREATE TABLE {quote_identifier(table_name)} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            school_key TEXT NOT NULL,
            source_type TEXT NOT NULL,
            year INTEGER,
            city TEXT,
            school_name TEXT,
            source_file TEXT NOT NULL,
            source_page INTEGER NOT NULL,
            major_section_code TEXT,
            major_section_title TEXT,
            subsection_title TEXT,
            population_context TEXT,
            section_code TEXT NOT NULL,
            section_title TEXT,
            table_title TEXT,
            row_group TEXT,
            row_name TEXT,
            source_row_key TEXT NOT NULL,
            confidence_min REAL,
            confidence_max REAL,
            extraction_methods TEXT,
            source_value_ids_json TEXT NOT NULL
            {dynamic},
            UNIQUE(document_id, source_page, source_row_key),
            FOREIGN KEY(document_id) REFERENCES documents(id)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        f"CREATE INDEX {quote_identifier('idx_' + table_name + '_school')} "
        f"ON {quote_identifier(table_name)}(school_key, year)"
    )
    connection.execute(
        f"CREATE INDEX {quote_identifier('idx_' + table_name + '_row')} "
        f"ON {quote_identifier(table_name)}(row_name)"
    )


def logical_row_payload(
    row: LogicalSectionRow,
    bindings: dict[tuple[str, int], SectionFieldBinding],
) -> dict[str, Any]:
    cells = row.cells
    first = cells[0]
    record = first.record
    document = first.document

    def dominant(attribute: str) -> str:
        return dominant_nonempty(getattr(cell.record, attribute) for cell in cells)

    confidences = [
        float(cell.record.confidence)
        for cell in cells
        if cell.record.confidence is not None
    ]
    payload: dict[str, Any] = {
        "document_id": document.id,
        "school_key": document.school_key,
        "source_type": document.source_type,
        "year": document.year,
        "city": document.city,
        "school_name": document.school_name,
        "source_file": document.source_file,
        "source_page": record.source_page,
        "major_section_code": dominant("major_section_code"),
        "major_section_title": dominant("major_section_title"),
        "subsection_title": dominant("subsection_title"),
        "population_context": dominant("population_context"),
        "section_code": record.section_code,
        "section_title": dominant("section_title"),
        "table_title": dominant("table_title"),
        "row_group": dominant("row_group"),
        "row_name": dominant("row_name"),
        "source_row_key": first.row_key,
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
        "extraction_methods": ", ".join(sorted({
            cell.record.extraction_method
            for cell in cells if cell.record.extraction_method
        })),
        "source_value_ids_json": json.dumps(
            [cell.raw_id for cell in cells], ensure_ascii=False
        ),
    }

    occurrences: dict[str, int] = defaultdict(int)
    for cell in cells:
        base = cell.semantic_field_base
        occurrences[base] += 1
        binding = bindings[(base, occurrences[base])]
        if cell.record.value_num is not None and binding.numeric_spec:
            payload[binding.numeric_spec.sql_name] = cell.record.value_num
        elif binding.text_spec:
            payload[binding.text_spec.sql_name] = cell.record.value_text

    return payload


def insert_extraction_issues(
    connection: sqlite3.Connection,
    cells: Sequence[SectionCell],
) -> None:
    rows = []
    for cell in cells:
        context = parse_raw_context(cell.record.raw_context)
        correction = clean_text(context.get("correction"))
        if correction:
            rows.append((
                cell.document.source_file,
                cell.record.source_page,
                cell.record.section_code,
                "deterministic_correction",
                correction,
                json.dumps(context, ensure_ascii=False),
            ))
    if rows:
        connection.executemany(
            """
            INSERT INTO extraction_issues(
                source_file, source_page, section_code,
                issue_type, message, context_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def build_section_database_from_records(
    *,
    documents: Sequence[DocumentInfo],
    records_by_document: dict[int, Sequence[Record]],
    output_path: Path,
    replace: bool = True,
    keep_empty: bool = False,
    verbose: bool = True,
) -> ExtractionSummary:
    if output_path.exists():
        if not replace:
            raise FileExistsError(f"{output_path} existe déjà.")
        output_path.unlink()

    all_cells: list[SectionCell] = []
    raw_id = 1
    document_by_id = {document.id: document for document in documents}

    for document_id, records in records_by_document.items():
        document = document_by_id[document_id]
        for record in records:
            if not clean_text(record.section_code):
                continue
            if (
                not keep_empty
                and not record_has_business_value(record)
                and not is_schema_only_record(record)
            ):
                continue
            cell = SectionCell(raw_id=raw_id, document=document, record=record)
            assign_cell_position(cell)
            all_cells.append(cell)
            raw_id += 1

    grouped_sections: dict[str, list[SectionCell]] = defaultdict(list)
    for cell in all_cells:
        grouped_sections[clean_text(cell.record.section_code)].append(cell)

    connection = sqlite3.connect(output_path, timeout=60)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")

    used_tables: set[str] = set()
    total_rows = 0

    try:
        create_section_database_schema(connection)
        connection.executemany(
            """
            INSERT INTO documents(
                id, school_key, source_type, year, city,
                school_name, source_file, page_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    document.id,
                    document.school_key,
                    document.source_type,
                    document.year,
                    document.city,
                    document.school_name,
                    document.source_file,
                    document.page_count,
                )
                for document in documents
            ],
        )

        for index, (section_code, cells) in enumerate(
            sorted(grouped_sections.items()), start=1
        ):
            rows = group_section_rows(cells)
            dominant_title = dominant_nonempty(
                cell.record.section_title for cell in cells
            )
            prepare_section_fields(rows, dominant_title)
            fields, bindings = build_section_field_specs(rows)
            data_rows = (
                rows
                if keep_empty
                else [row for row in rows if logical_row_has_business_value(row)]
            )
            table_name = make_section_table_name(section_code, used_tables)
            create_physical_section_table(connection, table_name, fields)

            metadata_columns = [
                "document_id", "school_key", "source_type", "year", "city",
                "school_name", "source_file", "source_page",
                "major_section_code", "major_section_title",
                "subsection_title", "population_context", "section_code", "section_title",
                "table_title", "row_group", "row_name", "source_row_key",
                "confidence_min", "confidence_max", "extraction_methods",
                "source_value_ids_json",
            ]
            columns = metadata_columns + [field.sql_name for field in fields]
            payloads = [logical_row_payload(row, bindings) for row in data_rows]
            column_sql = ", ".join(quote_identifier(column) for column in columns)
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {quote_identifier(table_name)} ({column_sql}) "
                f"VALUES ({placeholders})",
                [[payload.get(column) for column in columns] for payload in payloads],
            )

            section_titles = sorted({
                cell.record.section_title for cell in cells
                if clean_text(cell.record.section_title)
            })
            table_titles = sorted({
                cell.record.table_title for cell in cells
                if clean_text(cell.record.table_title)
            })
            major_code = dominant_nonempty(
                cell.record.major_section_code for cell in cells
            )
            major_title = dominant_nonempty(
                cell.record.major_section_title for cell in cells
            )
            population_context = dominant_nonempty(
                cell.record.population_context for cell in cells
            )
            column_summary = ", ".join(field.sql_name for field in fields)
            description = (
                f"Table {table_name} : données CTI du code {section_code}. "
                f"Titre principal : {dominant_title or 'non renseigné'}. "
                + (f"Contexte de population : {population_context}. " if population_context else "")
                + "Une ligne représente une ligne logique du formulaire pour "
                "une école et une campagne."
            )
            search_text = " | ".join(filter(None, [
                table_name, "CTI", section_code, major_code, major_title,
                dominant_title, population_context, " ; ".join(section_titles),
                " ; ".join(table_titles), column_summary,
            ]))
            connection.execute(
                """
                INSERT INTO section_catalog(
                    table_name, source_type, section_code,
                    major_section_code, major_section_title, section_title,
                    population_context,
                    section_titles_json, table_titles_json, row_granularity,
                    document_count, row_count, dynamic_column_count,
                    description, search_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table_name, "CTI", section_code, major_code, major_title,
                    dominant_title, population_context,
                    json.dumps(section_titles, ensure_ascii=False),
                    json.dumps(table_titles, ensure_ascii=False),
                    "une ligne logique du document",
                    len({cell.document.id for cell in cells}),
                    len(data_rows), len(fields), description, search_text,
                ),
            )
            connection.executemany(
                """
                INSERT INTO column_catalog(
                    table_name, column_name, sqlite_type, value_role,
                    semantic_base, occurrence_index, original_labels_json,
                    non_null_count, numeric_count, sample_values_json,
                    unit, measure_kind, aggregation_hint, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        table_name, field.sql_name, field.sqlite_type,
                        field.value_role, field.semantic_base,
                        field.occurrence_index,
                        json.dumps(field.original_labels, ensure_ascii=False),
                        field.non_null_count, field.numeric_count,
                        json.dumps(field.sample_values, ensure_ascii=False),
                        field.unit, field.measure_kind,
                        field.aggregation_hint, field.description,
                    )
                    for field in fields
                ],
            )
            insert_extraction_issues(connection, cells)
            total_rows += len(data_rows)

            if verbose:
                print(
                    f"[{index:>3}/{len(grouped_sections)}] "
                    f"CTI {section_code} -> {table_name} "
                    f"({len(data_rows)} lignes, {len(fields)} colonnes métier)",
                    file=sys.stderr,
                )

        connection.commit()
        connection.execute("PRAGMA optimize")
    except Exception:
        connection.rollback()
        connection.close()
        output_path.unlink(missing_ok=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass

    return ExtractionSummary(
        output_path=output_path,
        document_count=len(documents),
        section_count=len(grouped_sections),
        row_count=total_rows,
        record_count=len(all_cells),
    )


def normalize_inputs(inputs: Sequence[str | Path] | str | Path) -> list[str]:
    if isinstance(inputs, (str, Path)):
        return [str(inputs)]
    return [str(value) for value in inputs]


def extract_cti_to_sections(
    inputs: Sequence[str | Path] | str | Path,
    sqlite_path: str | Path = "cti_sections.sqlite",
    *,
    replace: bool = True,
    include_empty: bool = False,
    flat_csv_path: str | Path | None = None,
    review_csv_path: str | Path | None = None,
    flat_sqlite_path: str | Path | None = None,
    verbose: bool = True,
) -> ExtractionSummary:
    """API principale destinée à extracteur_unifie_cti_defi.py."""
    pdfs = list(iter_pdfs(normalize_inputs(inputs)))
    if not pdfs:
        raise FileNotFoundError("Aucun PDF CTI trouvé.")

    documents: list[DocumentInfo] = []
    records_by_document: dict[int, list[Record]] = {}
    all_records: list[Record] = []

    seen_contents: set[tuple[str, int | None, str]] = set()
    for pdf_path in pdfs:
        document_id = len(documents) + 1
        document = detect_document_info(pdf_path, document_id)
        identity = (document.school_key, document.year, hashlib.sha256(pdf_path.read_bytes()).hexdigest())
        if identity in seen_contents:
            if verbose:
                print(f"Doublon de contenu ignoré : {pdf_path}", file=sys.stderr)
            continue
        seen_contents.add(identity)
        if verbose:
            print(f"Extraction CTI : {pdf_path}", file=sys.stderr)
        records = extract_pdf(pdf_path, include_empty=include_empty)
        documents.append(document)
        records_by_document[document_id] = records
        all_records.extend(records)
        if verbose:
            print(f"  {len(records)} cellules extraites", file=sys.stderr)

    summary = build_section_database_from_records(
        documents=documents,
        records_by_document=records_by_document,
        output_path=Path(sqlite_path),
        replace=replace,
        keep_empty=include_empty,
        verbose=verbose,
    )

    if flat_csv_path is not None:
        write_csv(all_records, Path(flat_csv_path))
    if review_csv_path is not None:
        write_review_csv(all_records, Path(review_csv_path))
    if flat_sqlite_path is not None:
        write_sqlite(all_records, Path(flat_sqlite_path))

    return summary


# Alias volontairement larges pour les appels existants du script unifié.
def extract_cti(*args: Any, **kwargs: Any) -> ExtractionSummary:
    return extract_cti_to_sections(*args, **kwargs)


def extract_to_section_database(*args: Any, **kwargs: Any) -> ExtractionSummary:
    return extract_cti_to_sections(*args, **kwargs)


def run_extraction(*args: Any, **kwargs: Any) -> ExtractionSummary:
    return extract_cti_to_sections(*args, **kwargs)


def extract_to_sqlite(*args: Any, **kwargs: Any) -> ExtractionSummary:
    return extract_cti_to_sections(*args, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extrait les PDF CTI directement dans une base SQLite "
            "avec une table physique par section."
        )
    )
    parser.add_argument("inputs", nargs="+", help="PDF ou répertoire(s) CTI")
    parser.add_argument(
        "--sqlite",
        default="cti_sections.sqlite",
        help="Base structurée par section de sortie.",
    )
    parser.add_argument(
        "--flat-sqlite",
        help="Base plate cti_values facultative, uniquement pour diagnostic.",
    )
    parser.add_argument("--csv", help="CSV plat facultatif")
    parser.add_argument("--review-csv", help="CSV de contrôle facultatif")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Refuse d'écraser la base de sortie.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    summary = extract_cti_to_sections(
        args.inputs,
        args.sqlite,
        replace=not args.no_replace,
        include_empty=args.include_empty,
        flat_csv_path=args.csv,
        review_csv_path=args.review_csv,
        flat_sqlite_path=args.flat_sqlite,
        verbose=not args.quiet,
    )
    print(f"Base créée : {summary.output_path.resolve()}")
    print(f"Documents : {summary.document_count}")
    print(f"Sections : {summary.section_count}")
    print(f"Lignes structurées : {summary.row_count}")
    print(f"Cellules conservées : {summary.record_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

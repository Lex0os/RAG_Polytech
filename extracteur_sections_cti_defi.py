#!/usr/bin/env python3
"""Extraction directe des PDF CTI et DEFI vers une base SQLite par section.

Contrairement à l'ancienne chaîne :

    PDF -> data_values -> build_section_tables.py -> base par section

ce programme construit directement les tables de sections en mémoire :

    PDF -> cti_extractor/defi_extractor -> tables cti_* et defi_*

Le parseur CTI importé doit être la version contenant le traitement spécialisé
des sections de salaires VIII-D*-7. Les salaires et les nombres de réponses
sont alors distingués au moment de la lecture du PDF, avant la création des
colonnes SQL.

Dépendances locales :
- cti_extractor.py
- defi_extractor.py, uniquement si des PDF DEFI sont fournis
- pdfplumber >= 0.11

Exemple :
    python extracteur_sections_cti_defi.py ./pdfs \
        --sqlite cti_defi_sections.sqlite --replace
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pdfplumber

try:
    import cti_extractor
except ImportError as exc:
    raise SystemExit(
        "Impossible de charger cti_extractor.py. "
        "Place ce fichier à côté de l'extracteur unifié."
    ) from exc

try:
    import defi_extractor
except ImportError:
    defi_extractor = None


@dataclass(frozen=True)
class SourceDocument:
    id: int
    school_key: str
    source_type: str
    year: int | None
    city: str
    school_name: str
    source_file: str
    source_path: str
    page_count: int
    file_sha256: str
    source_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnifiedSummary:
    output_path: Path
    document_count: int
    section_count: int
    row_count: int
    record_count: int
    cti_document_count: int
    defi_document_count: int


COMMON_RECORD_FIELDS = tuple(cti_extractor.Record.__dataclass_fields__.keys())


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = (
        text.replace("\u00ad", "")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\ufffe", "")
        .replace("\r", "\n")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def fold_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    return "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    ).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def detect_source_type(path: Path, first_text: str) -> str:
    stem = fold_text(path.stem)

    if re.search(r"(?:^|[-_ .])defi(?:$|[-_ .])", stem):
        return "DEFI"

    if re.search(r"(?:^|[-_ .])cti(?:$|[-_ .])", stem):
        return "CTI"

    normalized = fold_text(first_text)

    if (
        "donnees des ecoles francaises d'ingenieurs" in normalized
        or "cdefi-cge-cti" in normalized
    ):
        return "DEFI"

    if (
        "commission des titres d'ingenieur" in normalized
        or "a la demande de la cti" in normalized
    ):
        return "CTI"

    raise ValueError(
        f"Type CTI/DEFI impossible à déterminer pour {path.name}."
    )


def campaign_year(path: Path, first_text: str) -> int | None:
    match = re.search(
        r"\b[Cc]ampagne\s*[:\-]?\s*(20\d{2})\b",
        first_text,
    )

    if match:
        return int(match.group(1))

    years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", path.stem)
    return int(years[-1]) if years else None


def filename_school_prefix(path: Path) -> str:
    stem = path.stem

    match = re.search(
        r"(?i)(?:[-_. ]+)(?:cti|defi)(?:[-_. ]+|$)",
        stem,
    )

    if match:
        prefix = stem[:match.start()]
    else:
        # Couvre des variantes comme « Lyon Données certifiées CDEFI ».
        prefix = re.split(
            r"(?i)\b(?:cdefi|cti|defi)\b",
            stem,
            maxsplit=1,
        )[0]

    prefix = re.sub(
        r"(?i)(?:[-_. ]+)(?:donnees?|données?)(?:[-_. ]+certifiees?)?\s*$",
        "",
        prefix,
    )
    prefix = re.sub(
        r"(?i)(?:[-_. ]+)donnees[-_. ]certifiees\s*$",
        "",
        prefix,
    )
    prefix = re.sub(r"\(\d+\)$", "", prefix)
    prefix = re.sub(r"[_ .]+", " ", prefix)
    return prefix.strip(" -_")


def school_key_from_prefix(prefix: str) -> str:
    key = cti_extractor.sql_slug(prefix, "inconnue").upper()

    aliases = {
        "ANNECY_CHAMBERY": "ANNECY_CHAMBERY",
        "PARIS_SACLAY": "SACLAY",
    }

    return aliases.get(key, key)


def title_case_school(prefix: str) -> str:
    return " ".join(
        token.capitalize()
        for token in re.split(r"[-_ ]+", prefix)
        if token
    )


def extract_school_name(
    source_type: str,
    first_text: str,
    city: str,
) -> str:
    if source_type == "CTI":
        match = re.search(
            r"Par la direction de\s*:\s*(.+?)"
            r"(?=\nAcadémie\s*:|\nA la demande de la CTI|$)",
            first_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    else:
        match = re.search(
            r"(?:Conférence des Directeurs des Écoles Françaises "
            r"d['’]Ingénieurs\s*/\s*)?(.+?)"
            r"(?=\n[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’ -]{1,40}\n"
            r"Données des Ecoles Françaises d['’]Ingénieurs|"
            r"\nDonnées des Ecoles Françaises d['’]Ingénieurs|$)",
            first_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    if not match:
        return city

    candidate = clean_text(match.group(1)).replace("\n", " ")
    candidate = re.sub(
        r"(?i)^Conférence des Directeurs des Écoles Françaises "
        r"d['’]Ingénieurs\s*/\s*",
        "",
        candidate,
    )
    return candidate or city


def read_document_metadata(
    path: Path,
    document_id: int,
) -> SourceDocument:
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        text_parts = [
            clean_text(pdf.pages[index].extract_text() or "")
            for index in range(min(2, page_count))
        ]

    first_text = "\n".join(text_parts)
    source_type = detect_source_type(path, first_text)
    year = campaign_year(path, first_text)
    prefix = filename_school_prefix(path)
    city = title_case_school(prefix)
    key = school_key_from_prefix(prefix)
    school_name = extract_school_name(source_type, first_text, city)

    return SourceDocument(
        id=document_id,
        school_key=key,
        source_type=source_type,
        year=year,
        city=city,
        school_name=school_name,
        source_file=path.name,
        source_path=str(path.resolve()),
        page_count=page_count,
        file_sha256=sha256_file(path),
    )


def normalize_record(record: Any, source_file: str) -> cti_extractor.Record:
    values: dict[str, Any] = {}

    for field in COMMON_RECORD_FIELDS:
        if field == "source_file":
            values[field] = source_file
            continue

        value = getattr(record, field, None)

        if field == "source_page":
            try:
                values[field] = int(value)
            except (TypeError, ValueError):
                values[field] = 0
        elif field == "value_num":
            values[field] = value
        elif field == "confidence":
            try:
                values[field] = float(value)
            except (TypeError, ValueError):
                values[field] = 0.0
        elif field == "column_headers_json":
            values[field] = clean_text(value) or "[]"
        else:
            values[field] = clean_text(value)

    return cti_extractor.Record(**values)


def extract_pdf_records(
    path: Path,
    metadata: SourceDocument,
    *,
    include_empty: bool,
) -> list[cti_extractor.Record]:
    if metadata.source_type == "CTI":
        extracted = cti_extractor.extract_pdf(
            path,
            include_empty=include_empty,
        )
    else:
        if defi_extractor is None:
            raise RuntimeError(
                "Un PDF DEFI a été détecté mais defi_extractor.py est absent. "
                "Place ce fichier à côté de l'extracteur unifié."
            )

        extracted = defi_extractor.extract_pdf(
            path,
            include_empty=include_empty,
        )

    return [
        normalize_record(record, metadata.source_file)
        for record in extracted
    ]


def iter_pdfs(inputs: Sequence[str | Path]) -> Iterator[Path]:
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw)

        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("*.pdf"))
        else:
            candidates = []

        for candidate in candidates:
            resolved = candidate.resolve()

            if (
                resolved.suffix.lower() == ".pdf"
                and resolved.exists()
                and resolved not in seen
            ):
                seen.add(resolved)
                yield resolved


def make_table_name(
    source_type: str,
    section_code: str,
    used_names: set[str],
) -> str:
    base = cti_extractor.sql_slug(
        f"{source_type}_{section_code}",
        f"{source_type.lower()}_section",
    )
    candidate = base

    if candidate in used_names:
        candidate = (
            base
            + "_"
            + cti_extractor.short_hash(
                f"{source_type}|{section_code}",
                8,
            )
        )

    used_names.add(candidate)
    return candidate


def create_documents_schema(connection: sqlite3.Connection) -> None:
    cti_extractor.create_section_database_schema(connection)

    # Colonnes supplémentaires utiles pour tracer le document d'origine.
    connection.execute(
        "ALTER TABLE documents ADD COLUMN source_path TEXT"
    )
    connection.execute(
        "ALTER TABLE documents ADD COLUMN file_sha256 TEXT"
    )
    connection.execute("""
        CREATE TABLE document_aliases (
            document_id INTEGER NOT NULL REFERENCES documents(id),
            source_file TEXT NOT NULL,
            source_path TEXT NOT NULL,
            PRIMARY KEY (document_id, source_path)
        )
    """)


def validate_no_structural_header_artifacts(connection: sqlite3.Connection) -> None:
    """Refuse une base contenant un faux scalaire issu d'en-têtes de tableau.

    Exemple historique : une colonne nommée avec tout le titre de la rubrique
    et la pseudo-valeur "Hommes Femmes Total". Mieux vaut interrompre la
    construction que livrer silencieusement une colonne métier fausse.
    """
    problems: list[str] = []
    rows = connection.execute(
        """
        SELECT table_name, column_name, sample_values_json, original_labels_json
        FROM column_catalog
        WHERE value_role = 'text'
        """
    ).fetchall()

    for table_name, column_name, samples_json, labels_json in rows:
        try:
            samples = json.loads(samples_json or "[]")
        except json.JSONDecodeError:
            samples = []
        if not isinstance(samples, list):
            samples = []

        bad_samples = [
            clean_text(sample)
            for sample in samples
            if cti_extractor.looks_like_structural_scalar_artifact(sample)
        ]
        if bad_samples:
            problems.append(
                f"{table_name}.{column_name}: {bad_samples!r}"
            )

    if problems:
        preview = "; ".join(problems[:12])
        suffix = "" if len(problems) <= 12 else f" (+{len(problems)-12} autres)"
        raise RuntimeError(
            "Validation du schéma échouée : des en-têtes de tableau ont été "
            "interprétés comme valeurs métier : " + preview + suffix
        )


def build_unified_section_database(
    *,
    documents: Sequence[SourceDocument],
    records_by_document: dict[int, Sequence[cti_extractor.Record]],
    output_path: Path,
    replace: bool,
    include_empty: bool,
    verbose: bool,
) -> UnifiedSummary:
    if output_path.exists():
        if not replace:
            raise FileExistsError(
                f"{output_path} existe déjà. Utilise --replace."
            )
        output_path.unlink()

    cti_documents = [
        document
        for document in documents
        if document.source_type == "CTI"
    ]
    defi_documents = [
        document
        for document in documents
        if document.source_type == "DEFI"
    ]

    document_info = {
        document.id: cti_extractor.DocumentInfo(
            id=document.id,
            school_key=document.school_key,
            source_type=document.source_type,
            year=document.year,
            city=document.city,
            school_name=document.school_name,
            source_file=document.source_file,
            page_count=document.page_count,
        )
        for document in documents
    }

    all_cells: list[cti_extractor.SectionCell] = []
    raw_id = 1

    for document_id, records in records_by_document.items():
        document = document_info[document_id]

        for record in records:
            if not clean_text(record.section_code):
                continue

            if (
                not include_empty
                and not cti_extractor.record_has_business_value(record)
                and not cti_extractor.is_schema_only_record(record)
            ):
                continue

            cell = cti_extractor.SectionCell(
                raw_id=raw_id,
                document=document,
                record=record,
            )
            cti_extractor.assign_cell_position(cell)
            all_cells.append(cell)
            raw_id += 1

    grouped_sections: dict[
        tuple[str, str],
        list[cti_extractor.SectionCell],
    ] = {}

    for cell in all_cells:
        key = (
            cell.document.source_type,
            clean_text(cell.record.section_code),
        )
        grouped_sections.setdefault(key, []).append(cell)

    connection = sqlite3.connect(output_path, timeout=60)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 60000")

    used_table_names: set[str] = set()
    total_rows = 0

    try:
        create_documents_schema(connection)

        connection.executemany(
            """
            INSERT INTO documents(
                id, school_key, source_type, year, city,
                school_name, source_file, page_count,
                source_path, file_sha256
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    document.source_path,
                    document.file_sha256,
                )
                for document in documents
            ],
        )
        connection.executemany(
            "INSERT INTO document_aliases(document_id, source_file, source_path) VALUES (?, ?, ?)",
            [(document.id, Path(path).name, path)
             for document in documents
             for path in (document.source_path, *document.source_aliases)],
        )

        for index, ((source_type, section_code), cells) in enumerate(
            sorted(grouped_sections.items()),
            start=1,
        ):
            rows = cti_extractor.group_section_rows(cells)
            dominant_title = cti_extractor.dominant_nonempty(
                cell.record.section_title
                for cell in cells
            )

            cti_extractor.prepare_section_fields(
                rows,
                dominant_title,
            )
            fields, bindings = (
                cti_extractor.build_section_field_specs(rows)
            )
            # Les cellules schema_only servent à créer les colonnes, mais
            # ne deviennent pas de fausses lignes de données par défaut.
            data_rows = (
                rows
                if include_empty
                else [
                    row
                    for row in rows
                    if cti_extractor.logical_row_has_business_value(row)
                ]
            )
            table_name = make_table_name(
                source_type,
                section_code,
                used_table_names,
            )

            cti_extractor.create_physical_section_table(
                connection,
                table_name,
                fields,
            )

            metadata_columns = [
                "document_id",
                "school_key",
                "source_type",
                "year",
                "city",
                "school_name",
                "source_file",
                "source_page",
                "major_section_code",
                "major_section_title",
                "subsection_title",
                "population_context",
                "section_code",
                "section_title",
                "table_title",
                "row_group",
                "row_name",
                "source_row_key",
                "confidence_min",
                "confidence_max",
                "extraction_methods",
                "source_value_ids_json",
            ]
            columns = metadata_columns + [
                field.sql_name
                for field in fields
            ]
            payloads = [
                cti_extractor.logical_row_payload(row, bindings)
                for row in data_rows
            ]
            column_sql = ", ".join(
                cti_extractor.quote_identifier(column)
                for column in columns
            )
            placeholders = ", ".join("?" for _ in columns)

            connection.executemany(
                f"""
                INSERT INTO
                    {cti_extractor.quote_identifier(table_name)}
                    ({column_sql})
                VALUES ({placeholders})
                """,
                [
                    [payload.get(column) for column in columns]
                    for payload in payloads
                ],
            )

            section_titles = sorted(
                {
                    clean_text(cell.record.section_title)
                    for cell in cells
                    if clean_text(cell.record.section_title)
                }
            )
            table_titles = sorted(
                {
                    clean_text(cell.record.table_title)
                    for cell in cells
                    if clean_text(cell.record.table_title)
                }
            )
            major_code = cti_extractor.dominant_nonempty(
                cell.record.major_section_code
                for cell in cells
            )
            major_title = cti_extractor.dominant_nonempty(
                cell.record.major_section_title
                for cell in cells
            )
            population_context = cti_extractor.dominant_nonempty(
                cell.record.population_context
                for cell in cells
            )
            column_summary = ", ".join(
                field.sql_name
                for field in fields
            )
            description = (
                f"Table {table_name} : données {source_type} du code "
                f"{section_code}. Titre principal : "
                f"{dominant_title or 'non renseigné'}. "
                "Une ligne représente une ligne logique du formulaire "
                "pour une école et une campagne."
            )
            search_text = " | ".join(
                filter(
                    None,
                    [
                        table_name,
                        source_type,
                        section_code,
                        major_code,
                        major_title,
                        dominant_title,
                        population_context,
                        " ; ".join(section_titles),
                        " ; ".join(table_titles),
                        column_summary,
                    ],
                )
            )

            connection.execute(
                """
                INSERT INTO section_catalog(
                    table_name,
                    source_type,
                    section_code,
                    major_section_code,
                    major_section_title,
                    section_title,
                    population_context,
                    section_titles_json,
                    table_titles_json,
                    row_granularity,
                    document_count,
                    row_count,
                    dynamic_column_count,
                    description,
                    search_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table_name,
                    source_type,
                    section_code,
                    major_code,
                    major_title,
                    dominant_title,
                    population_context,
                    json.dumps(
                        section_titles,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        table_titles,
                        ensure_ascii=False,
                    ),
                    "une ligne logique du document",
                    len(
                        {
                            cell.document.id
                            for cell in cells
                        }
                    ),
                    len(rows),
                    len(fields),
                    description,
                    search_text,
                ),
            )

            connection.executemany(
                """
                INSERT INTO column_catalog(
                    table_name,
                    column_name,
                    sqlite_type,
                    value_role,
                    semantic_base,
                    occurrence_index,
                    original_labels_json,
                    non_null_count,
                    numeric_count,
                    sample_values_json,
                    unit,
                    measure_kind,
                    aggregation_hint,
                    description
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        table_name,
                        field.sql_name,
                        field.sqlite_type,
                        field.value_role,
                        field.semantic_base,
                        field.occurrence_index,
                        json.dumps(
                            field.original_labels,
                            ensure_ascii=False,
                        ),
                        field.non_null_count,
                        field.numeric_count,
                        json.dumps(
                            field.sample_values,
                            ensure_ascii=False,
                        ),
                        field.unit,
                        field.measure_kind,
                        field.aggregation_hint,
                        field.description,
                    )
                    for field in fields
                ],
            )

            cti_extractor.insert_extraction_issues(
                connection,
                cells,
            )
            total_rows += len(rows)

            if verbose:
                print(
                    f"[{index:>3}/{len(grouped_sections)}] "
                    f"{source_type} {section_code} -> {table_name} "
                    f"({len(rows)} lignes, "
                    f"{len(fields)} colonnes métier)",
                    file=sys.stderr,
                )

        # Garde-fou global : aucune table ne doit contenir les anciens faux
        # scalaires de type "Hommes Femmes Total". Si un nouveau layout non
        # reconnu apparaît, la construction échoue au lieu de produire une base
        # silencieusement incorrecte.
        validate_no_structural_header_artifacts(connection)

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

    return UnifiedSummary(
        output_path=output_path,
        document_count=len(documents),
        section_count=len(grouped_sections),
        row_count=total_rows,
        record_count=len(all_cells),
        cti_document_count=len(cti_documents),
        defi_document_count=len(defi_documents),
    )


def export_review_csv(
    *,
    documents: Sequence[SourceDocument],
    records_by_document: dict[int, Sequence[cti_extractor.Record]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document_by_id = {
        document.id: document
        for document in documents
    }
    fields = [
        "source_type",
        "year",
        "school_key",
        "source_file",
        "source_page",
        "section_code",
        "section_title",
        "row_name",
        "column_path",
        "value_text",
        "value_num",
        "confidence",
        "extraction_method",
        "raw_context",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter=";",
        )
        writer.writeheader()

        for document_id, records in records_by_document.items():
            document = document_by_id[document_id]
            threshold = (
                0.80
                if document.source_type == "CTI"
                else 0.84
            )

            for record in records:
                context = clean_text(record.raw_context)
                correction = (
                    "correction"
                    in fold_text(context)
                    or "salary_parser"
                    in fold_text(context)
                )

                if (
                    record.confidence >= threshold
                    and clean_text(record.value_text)
                    and not correction
                ):
                    continue

                writer.writerow(
                    {
                        "source_type": document.source_type,
                        "year": document.year,
                        "school_key": document.school_key,
                        "source_file": document.source_file,
                        "source_page": record.source_page,
                        "section_code": record.section_code,
                        "section_title": record.section_title,
                        "row_name": record.row_name,
                        "column_path": record.column_path,
                        "value_text": record.value_text,
                        "value_num": record.value_num,
                        "confidence": record.confidence,
                        "extraction_method": record.extraction_method,
                        "raw_context": record.raw_context,
                    }
                )



CHAIR_AUDIT_COLUMNS = (
    (
        "periode_de_la_chaire_texte_brut",
        "TEXT",
        "text",
        "periode_de_la_chaire_texte_brut",
        "Copie fidèle de la cellule de période du PDF.",
    ),
    (
        "annee_de_debut_de_la_chaire",
        "REAL",
        "numeric",
        "annee_de_debut_de_la_chaire",
        "Année de début dérivée de la cellule brute.",
    ),
    (
        "annee_de_fin_de_la_chaire",
        "REAL",
        "numeric",
        "annee_de_fin_de_la_chaire",
        "Année de fin dérivée de la cellule brute.",
    ),
    (
        "date_de_debut_de_la_chaire",
        "TEXT",
        "text",
        "date_de_debut_de_la_chaire",
        "Date de début normalisée au format ISO YYYY-MM-DD.",
    ),
    (
        "date_de_fin_de_la_chaire",
        "TEXT",
        "text",
        "date_de_fin_de_la_chaire",
        "Date de fin normalisée au format ISO YYYY-MM-DD.",
    ),
    (
        "statut_d_analyse_de_la_periode",
        "TEXT",
        "text",
        "statut_d_analyse_de_la_periode",
        "Statut déterministe du parsing de la période.",
    ),
    (
        "confiance_d_analyse_de_la_periode",
        "REAL",
        "numeric",
        "confiance_d_analyse_de_la_periode",
        "Confiance déterministe du parsing, entre 0 et 1.",
    ),
    (
        "note_d_analyse_de_la_periode",
        "TEXT",
        "text",
        "note_d_analyse_de_la_periode",
        "Explication de la normalisation ou de l'ambiguïté.",
    ),
)


def ensure_chair_audit_schema(
    database_path: Path,
) -> None:
    """Garantit un schéma stable pour ``defi_v_1``.

    Certaines colonnes, notamment les dates exactes, peuvent être entièrement
    vides dans un lot de PDF. Le constructeur dynamique ne les créerait alors
    pas. Cette étape ajoute uniquement les colonnes absentes, sans modifier
    les valeurs extraites.
    """
    if not database_path.exists():
        return

    with sqlite3.connect(database_path) as connection:
        table_exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'defi_v_1'
            """
        ).fetchone()

        if table_exists is None:
            return

        existing = {
            str(row[1])
            for row in connection.execute(
                'PRAGMA table_info("defi_v_1")'
            )
        }
        added = 0

        for (
            column_name,
            sqlite_type,
            value_role,
            semantic_base,
            description,
        ) in CHAIR_AUDIT_COLUMNS:
            if column_name not in existing:
                connection.execute(
                    f"""
                    ALTER TABLE "defi_v_1"
                    ADD COLUMN {cti_extractor.quote_identifier(column_name)}
                    {sqlite_type}
                    """
                )
                existing.add(column_name)
                added += 1

            connection.execute(
                """
                INSERT OR IGNORE INTO column_catalog(
                    table_name,
                    column_name,
                    sqlite_type,
                    value_role,
                    semantic_base,
                    occurrence_index,
                    original_labels_json,
                    non_null_count,
                    numeric_count,
                    sample_values_json,
                    unit,
                    measure_kind,
                    aggregation_hint,
                    description
                )
                VALUES (?, ?, ?, ?, ?, 1, ?, 0, 0, '[]', '', ?, ?, ?)
                """,
                (
                    "defi_v_1",
                    column_name,
                    sqlite_type,
                    value_role,
                    semantic_base,
                    json.dumps(
                        [column_name.replace("_", " ")],
                        ensure_ascii=False,
                    ),
                    (
                        "year"
                        if "annee_" in column_name
                        else (
                            "confidence"
                            if column_name.startswith("confiance_")
                            else "text"
                        )
                    ),
                    (
                        "do_not_aggregate"
                        if sqlite_type == "TEXT"
                        else "context_only"
                    ),
                    description,
                ),
            )

        if added:
            connection.execute(
                """
                UPDATE section_catalog
                SET dynamic_column_count = dynamic_column_count + ?
                WHERE table_name = 'defi_v_1'
                """,
                (added,),
            )

        connection.commit()

def extract_to_sections(
    *,
    inputs: Sequence[str | Path],
    sqlite_path: Path,
    replace: bool,
    include_empty: bool,
    review_csv_path: Path | None,
    verbose: bool,
) -> UnifiedSummary:
    pdfs = list(iter_pdfs(inputs))

    if not pdfs:
        raise FileNotFoundError("Aucun PDF CTI ou DEFI trouvé.")

    documents: list[SourceDocument] = []
    records_by_document: dict[
        int,
        list[cti_extractor.Record],
    ] = {}

    seen_contents: dict[tuple[str, str, int | None, str], int] = {}
    for pdf_path in pdfs:
        document_id = len(documents) + 1
        metadata = read_document_metadata(
            pdf_path,
            document_id,
        )
        identity = (metadata.school_key, metadata.source_type, metadata.year, metadata.file_sha256)
        if identity in seen_contents:
            index = seen_contents[identity]
            original = documents[index]
            documents[index] = dataclass_replace(original, source_aliases=(*original.source_aliases, metadata.source_path))
            if verbose:
                print(f"Doublon de contenu ignoré : {pdf_path.name} (conservé : {original.source_file})", file=sys.stderr)
            continue
        seen_contents[identity] = len(documents)

        if verbose:
            print(
                f"Extraction {metadata.source_type} : {pdf_path}",
                file=sys.stderr,
            )

        records = extract_pdf_records(
            pdf_path,
            metadata,
            include_empty=include_empty,
        )

        if verbose:
            print(
                f"  {len(records)} cellules extraites",
                file=sys.stderr,
            )

        documents.append(metadata)
        records_by_document[document_id] = records

    summary = build_unified_section_database(
        documents=documents,
        records_by_document=records_by_document,
        output_path=sqlite_path,
        replace=replace,
        include_empty=include_empty,
        verbose=verbose,
    )

    ensure_chair_audit_schema(sqlite_path)

    if review_csv_path is not None:
        export_review_csv(
            documents=documents,
            records_by_document=records_by_document,
            path=review_csv_path,
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrait directement les PDF CTI et DEFI dans une base SQLite "
            "avec une table physique par couple type/source et section."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF ou dossiers contenant des PDF CTI et/ou DEFI.",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path("cti_defi_sections.sqlite"),
        help="Base SQLite structurée de sortie.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Remplace la base de sortie si elle existe.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Conserve les cellules vides.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        help=(
            "CSV facultatif des extractions faibles ou corrigées, "
            "notamment les reclassements de salaires."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Masque le détail de chaque section créée.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        summary = extract_to_sections(
            inputs=args.inputs,
            sqlite_path=args.sqlite,
            replace=args.replace,
            include_empty=args.include_empty,
            review_csv_path=args.review_csv,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(
            f"ERREUR : {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Base créée : {summary.output_path.resolve()}")
    print(
        "Documents : "
        f"{summary.document_count} "
        f"({summary.cti_document_count} CTI, "
        f"{summary.defi_document_count} DEFI)"
    )
    print(f"Sections : {summary.section_count}")
    print(f"Lignes structurées : {summary.row_count}")
    print(f"Cellules conservées : {summary.record_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

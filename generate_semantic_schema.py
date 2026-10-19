import json
import sqlite3
from pathlib import Path


TECHNICAL_COLUMNS = {
    "id",
    "document_id",
    "source_type",
    "city",
    "school_name",
    "source_file",
    "source_page",
    "major_section_code",
    "major_section_title",
    "subsection_title",
    "section_code",
    "section_title",
    "table_title",
    "source_row_key",
    "confidence_min",
    "confidence_max",
    "extraction_methods",
    "source_value_ids_json",
}

# Évite de gonfler inutilement le contexte du LLM sur les tables ayant
# énormément de valeurs différentes dans row_name/row_group.
# Les valeurs les plus fréquentes sont conservées en priorité.
MAX_DIMENSION_VALUES = 30


def load_json_list(value: str) -> list[str]:
    try:
        result = json.loads(value or "[]")
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def quote_identifier(identifier: str) -> str:
    """Quote proprement un identifiant SQLite."""
    return '"' + identifier.replace('"', '""') + '"'


def compact_text(value: object) -> str:
    """Nettoie une valeur destinée au semantic_schema."""
    return " ".join(str(value or "").split()).strip()


def dimension_values(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    limit: int = MAX_DIMENSION_VALUES,
) -> tuple[list[str], int]:
    """Retourne les valeurs observées d'une dimension à faible cardinalité.

    Les valeurs les plus fréquentes sont prioritaires afin de privilégier les
    libellés canoniques présents dans plusieurs documents/écoles.
    """
    table_sql = quote_identifier(table_name)
    column_sql = quote_identifier(column_name)

    # Certaines tables particulières peuvent ne pas posséder la dimension.
    physical_columns = {
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({table_sql})"
        ).fetchall()
    }
    if column_name not in physical_columns:
        return [], 0

    rows = connection.execute(
        f"""
        SELECT {column_sql} AS value, COUNT(*) AS occurrences
        FROM {table_sql}
        WHERE {column_sql} IS NOT NULL
          AND TRIM(CAST({column_sql} AS TEXT)) <> ''
        GROUP BY {column_sql}
        ORDER BY occurrences DESC, CAST({column_sql} AS TEXT) COLLATE NOCASE
        """
    ).fetchall()

    values = []
    for row in rows:
        value = compact_text(row["value"])
        if value and value not in values:
            values.append(value)

    total = len(values)
    if limit and limit > 0:
        values = values[:limit]

    return values, total


def append_dimension(
    block: list[str],
    *,
    name: str,
    description: str,
    values: list[str],
    total_values: int,
) -> None:
    block.append(f"- {name} TEXT : {description}")

    if not values:
        return

    if total_values > len(values):
        block.append(
            f"  valeurs observées ({len(values)}/{total_values}): "
            + " | ".join(values)
        )
    else:
        block.append(
            "  valeurs observées: " + " | ".join(values)
        )


def generate_semantic_schema(
    database_path: str,
    output_path: str = "semantic_schema.txt",
) -> None:

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    sections = connection.execute(
        """
        SELECT
            table_name,
            source_type,
            section_code,
            major_section_title,
            section_title,
            population_context,
            row_granularity
        FROM section_catalog
        """
    ).fetchall()

    blocks = []

    for section in sections:
        table_name = section["table_name"]

        columns = connection.execute(
            """
            SELECT
                column_name,
                sqlite_type,
                original_labels_json,
                unit,
                measure_kind,
                aggregation_hint,
                description
            FROM column_catalog
            WHERE table_name = ?
            ORDER BY occurrence_index, column_name
            """,
            (table_name,),
        ).fetchall()

        row_names, row_name_count = dimension_values(
            connection,
            table_name,
            "row_name",
        )
        row_groups, row_group_count = dimension_values(
            connection,
            table_name,
            "row_group",
        )

        block = [
            f"TABLE: {table_name}",
            f"CODE: {section['section_code']}",
        ]

        if section["major_section_title"]:
            block.append(
                f"SECTION: {section['major_section_title']}"
            )

        block.append(
            f"SUJET: {section['section_title']}"
        )

        population_context = compact_text(
            section["population_context"]
        )

        if population_context:
            block.append(
                f"CONTEXTE: {population_context}"
            )

        block.extend(
            [
                "",
                "DIMENSIONS:",
                "- school_key TEXT : identifiant de l'école",
                "- year INTEGER : année / campagne",
            ]
        )

        append_dimension(
            block,
            name="row_name",
            description="sous-catégorie éventuelle de la rubrique",
            values=row_names,
            total_values=row_name_count,
        )
        append_dimension(
            block,
            name="row_group",
            description="groupe éventuel de la sous-catégorie",
            values=row_groups,
            total_values=row_group_count,
        )

        block.extend(["", "COLONNES METIER:"])

        for column in columns:
            if column["column_name"] in TECHNICAL_COLUMNS:
                continue

            labels = load_json_list(
                column["original_labels_json"]
            )

            label = (
                " > ".join(labels)
                if labels
                else column["column_name"]
            )

            block.append(
                f"- {column['column_name']} "
                f"{column['sqlite_type']}"
            )

            block.append(
                f"  libellé: {label}"
            )

            if column["measure_kind"]:
                block.append(
                    f"  type: {column['measure_kind']}"
                )

            if column["unit"]:
                block.append(
                    f"  unité: {column['unit']}"
                )

            if column["aggregation_hint"]:
                block.append(
                    f"  agrégation: "
                    f"{column['aggregation_hint']}"
                )

            if column["description"]:
                block.append(
                    f"  description: {column['description']}"
                )

            block.append("")

        blocks.append("\n".join(block))

    connection.close()

    Path(output_path).write_text(
        "\n"
        .join(blocks),
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate_semantic_schema(
        "cti_defi_sections.sqlite",
        "semantic_schema.txt",
    )

# coding: utf-8
"""
Render kolekcji analiz do HTML (jeden plik, wykresy jako base64).

`generate_report({zmienna: VariableAnalysis})` iteruje `va.elements` i woła
`el.render_html()` — patrz spec/2026-07-07-raport-kolekcja.md §5. Osadzanie
figur i ich zamykanie należy do `FigureElement` (buckets/report_elements.py).
"""


def _ordered_items(collection: dict, order):
    """
    Kolejność sekcji raportu. `order`: "gini" (malejąco po GINI), "alpha"
    (alfabetycznie) albo jawna lista nazw zmiennych.
    """
    if isinstance(order, (list, tuple)):
        return [(name, collection[name]) for name in order if name in collection]
    if order == "alpha":
        return sorted(collection.items(), key=lambda kv: kv[0])
    return sorted(
        collection.items(), key=lambda kv: kv[1].gini_value, reverse=True
    )


def generate_variable_report(collection: dict, order="gini"):
    """
    Generuje sekcje HTML z kolekcji `{zmienna: VariableAnalysis}`, iterując
    `va.elements` i wołając `el.render_html()` (figury powstają i są zamykane
    w renderze elementu — kolekcja nie jest mutowana).

    Returns:
        tuple: (report_content, navigation_links_by_gini, navigation_links_alpha)
    """
    sections = []
    nav_links = []

    for name, va in _ordered_items(collection, order):
        section_html = (
            f'<div class="variable-section" id="{name}">\n'
            f"<h2>Zmienna: {name}</h2>\n"
        )
        for element in va.elements:
            section_html += element.render_html()
        section_html += "</div><hr>"
        sections.append(section_html)

        gini = round(va.gini_value * 100, 1)
        nav_links.append(
            f'<li title="{name}"><a href="#{name}">{name} ({gini})</a></li>'
        )

    return "\n".join(sections), "\n".join(nav_links), "\n".join(sorted(nav_links))


def fill_template(report_content: str, navigation_links: str, navigation_links2) -> str:
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Raport zmiennych</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 0; 
                display: flex; /* Ustawia układ flexbox */
            }}

            /* Stylizacja nawigacji */
            nav {{
                width: 250px;
                background-color: #f4f4f4;
                padding: 20px;
                border-right: 1px solid #ddd;
                position: fixed;
                height: 100vh;
                overflow-y: auto;
                box-sizing: border-box;
                font-size: 0.7em; /* Zmniejszenie czcionki */
                line-height: 1; /* Zmniejszenie odległości między liniami */
            }}
            nav a{{
                text-decoration: none; /* Usuwa podkreślenie w linkach */
            }}

            nav ul {{
                list-style-type: none;
                padding: 0;
            }}

            nav li {{
                margin-bottom: 10px;
                position: relative; /* Ustawienie nawigacji w kontekście dymków */
            }}

            .content {{ 
                margin-left: 250px; /* Zapewnia miejsce po lewej dla nawigacji */
                padding: 20px; 
                width: calc(100% - 250px); /* Reszta miejsca dla treści */
                box-sizing: border-box; 
            }}
            .table {{ 
                font-size: 0.8em; /* Zmniejszenie czcionki */
                border-collapse: collapse; 
                width: 100%; 
                margin-top: 20px; 
            }}
            .table th, .table td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }}
            .table th {{ 
                background-color: #f2f2f2; 
            }}
            .variable-section {{ 
                margin-bottom: 40px; 
            }}
            h2 {{ 
                color: #333; 
            }}
        </style>

    </head>
    <body>
        <nav>
            <h2>Według siły (GINI)</h2>
            <ul>
                {navigation_links}
            </ul>
            <h2>Alfabetycznie</h2>
                {navigation_links2}
        </nav>
        <div class="content">
            <h1>Raport zmiennych</h1>
            {report_content}
        </div>
    </body>
    </html>
    """

    return html_template


def generate_report(collection: dict, order="gini") -> str:
    """Buduje pełny HTML z kolekcji `{zmienna: VariableAnalysis}`."""
    report_content, navigation_links, navigation_links2 = generate_variable_report(
        collection, order=order
    )
    return fill_template(report_content, navigation_links, navigation_links2)


def save(report, filename):
    # Zapis strony do pliku HTML
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)



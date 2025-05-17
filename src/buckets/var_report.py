import base64
import os
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd

def generate_variable_report(data: dict):
    """
    Generuje raport HTML dla zmiennych na podstawie słownika, gdzie kluczem jest nazwa kolumny,
    a wartościami lista zawierająca statystyki jako DataFrame lub wygenerowany wykres.

    Args:
        data (dict): Słownik, gdzie kluczem jest nazwa kolumny, a wartościami lista z elementami:
                     - statystyki jako pandas.DataFrame
                     - wykres jako obiekt matplotlib.figure.Figure

    Returns:
        tuple: Zawiera dwie wartości:
            - report_content (str): Treść HTML z sekcjami dla zmiennych.
            - navigation_links (str): HTML z listą odnośników do zmiennych.
    """
    sections = []
    nav_links = []

    data = dict(sorted(data.items(), key=lambda x: x[1][0].iloc[0, 1], reverse=True))
    
    for column, elements in data.items():
        section_html = f"""
        <div class="variable-section" id="{column}">
            <h2>Zmienna: {column}</h2>
        """

        for element in elements:
            if isinstance(element, pd.DataFrame):
                # Jeśli element jest DataFrame, generujemy tabelę HTML
                stats_html = element.to_html(classes="table", border=0, index=False)
                section_html += f"""
                <h3>Statystyki</h3>
                {stats_html}
                """
            #elif isinstance(element, plt.Figure):
            # TODO: Dodaj obsługę wykresów jako obiektów Figure
            else :
                # Jeśli element jest wykresem, zapisujemy go do base64
                buffer = BytesIO()
                element.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                buffer.close()
                plt.close(element)

                section_html += f"""
                <h3>Wykres</h3>
                <img src="data:image/png;base64,{image_base64}" alt="Wykres dla {column}">
                """

        section_html += "</div>"
        sections.append(section_html)

        gini = round(elements[0].iloc[0, 1] * 100, 1)
        # Dodawanie linku nawigacji
        nav_links.append(f'<li title="{column}"><a href="#{column}">{column} ({gini})</a></li>')

    return "\n".join(sections), "\n".join(nav_links), "\n".join(sorted(nav_links))
# Struktura HTML
#def generate_variable_report(df: pd.DataFrame):
#    """
#    Generuje raport HTML dla zmiennych w DataFrame, zawierający statystyki i wykresy.
#
#    Args:
#        df (pandas.DataFrame): Ramka danych z danymi.
#
#    Returns:
#        tuple: Zawiera dwie wartości:
#            - report_content (str): Treść HTML z sekcjami dla zmiennych.
#            - navigation_links (str): HTML z listą odnośników do zmiennych.
#    """
#    sections = []
#    nav_links = []
#    for column in df.columns:
#        # Obliczanie statystyk
#        stats = df[column].describe().to_frame().T
#        stats_html = stats.to_html(classes="table", border=0, index=False)
#
#        # Tworzenie wykresu
#        plt.figure(figsize=(6, 4))
#        plt.plot(df[column], marker="o", color="skyblue", label=column)
#        plt.title(f"Wykres dla {column}")
#        plt.xlabel("Indeks")
#        plt.ylabel("Wartość")
#        plt.legend()
#
#        # Zapis wykresu do base64
#        buffer = BytesIO()
#        plt.savefig(buffer, format="png", bbox_inches="tight")
#        buffer.seek(0)
#        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
#        buffer.close()
#        plt.close()
#
#        # Dodawanie sekcji dla zmiennej
#        section_html = f"""
#        <div class="variable-section" id="{column}">
#            <h2>Zmienna: {column}</h2>
#            <h3>Statystyki</h3>
#            {stats_html}
#            <h3>Wykres</h3>
#            <img src="data:image/png;base64,{image_base64}" alt="Wykres dla {column}">
#        </div>
#        """
#        sections.append(section_html)
#
#        # Dodawanie linku nawigacji
#        nav_links.append(f'<li title="{column}"><a href="#{column}">{column}</a></li>')
#
#    return "\n".join(sections), "\n".join(nav_links)


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


def generate_report(report_objects: dict[str, list]) -> str:
    # Generowanie sekcji i nawigacji
    report_content, navigation_links, navigation_links2 = generate_variable_report(report_objects)
    return fill_template(report_content, navigation_links, navigation_links2)


def save(report, filename):
    # Zapis strony do pliku HTML
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    # Tworzenie wykresów
    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.gca()
    ax1.plot([10, 20, 15, 25, 30, 35])

    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.gca()
    ax2.plot([5, 10, 15, 20, 25, 30])

    # Dane przykładowe
    data = {
        "Zmienna1": [
            pd.DataFrame({"Statystyka": ["Średnia", "Mediana"], "Wartość": [20, 15]}),
            fig1  # Przekazujemy obiekt Figure
        ],
        "Zmienna2": [
            pd.DataFrame({"Statystyka": ["Min", "Max"], "Wartość": [5, 30]}),
            fig2  # Przekazujemy obiekt Figure
        ],
    }

    # Generowanie raportu
    report_content, navigation_links, navigation_links2 = generate_variable_report(data)
    html = fill_template(report_content, navigation_links2)

    # Zapis raportu
    os.makedirs("result", exist_ok=True)
    save(html, "./result/report.html")

    print("Raport został zapisany jako 'result/report.html'.")
    
#if __name__ == "__main__":
#    # Dane przykładowe
#    data = {
#        "Zmienna1": [10, 20, 15, 25, 30, 35],
#        "Zmienna2": [5, 10, 15, 20, 25, 30],
#        "Zmienna3": [100, 200, 300, 400, 500, 600],
#    }
#    df = pd.DataFrame(data)
#
#    html = generate_report(df)
#    os.makedirs("result", exist_ok=True)
#    save(html, "./result/report.html")
#
#    print("Raport został zapisany jako 'result/report.html'.")
#
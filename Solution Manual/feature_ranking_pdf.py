"""
Description: Create a pdf report based on the feature ranking analsysis
Author:      Razi Gaskari
Created:     Dec, 2022
limitation:  max 15 features can be shows in report
"""
import pandas as pd
from fpdf import FPDF  


def create_report(df: pd.DataFrame, path: str) -> None:
    """
    create a pdf report
    limitation: max 15 features can be shows in report
    """

    no_features = df.shape[1] + 2
    columns = df.columns.to_list()  # Get list of dataframe columns
    columns = [["Method"] + columns]

    rows_value = df.values.tolist()
    index_value = df.index.values.tolist()

    rows = [([sub] + rows_value[i]) for i, sub in enumerate(index_value)]
    data = columns + rows  # Combine columns and rows in one list

    # Start pdf creating
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.ln(20)
    pdf.cell(0, 0, "Feature Ranking :", ln=0, align="L")
    pdf.ln(20)
    pdf.set_font("Times", size=8)
    line_height = float(pdf.font_size * 3.1)
    col_width = pdf.epw / no_features  # distribute content evenly

    for row in data:
        for j, datum in enumerate(row):
            if j == 0:
                width = float(col_width * 2.1)
            else:
                width = float(col_width)

            pdf.multi_cell(
                w=width,
                h=line_height,
                txt=str(datum),
                border=1,
                new_y="TOP",
                max_line_height=pdf.font_size,
            )
        pdf.ln(line_height)

    pdf.ln(20)
    pdf.image(path + "plot.png", x=5, y=140, w=190, h=0, type="", link="")

    pdf.output(path + "report.pdf", "F")

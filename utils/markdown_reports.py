import html
from collections import defaultdict

from .constants import METRIC_HEADERS
from .types import Facet, View


def format_table(measurements, headers=METRIC_HEADERS, round=3, empty=""):
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows = []
    for measurement in measurements:
        title = f" title=\"{html.escape(measurement.description)}\""
        rows.append(f"<tr{title}>\n{measurement.to_table_cells(headers=headers, round=round, empty=empty)}</tr>")
    rows_html = "\n".join(rows)
    return f"""
<table>
<tr>
{header_html}
</tr>
{rows_html}
</table>
"""

def format_facet_grouped_table(measurements, headers=METRIC_HEADERS, round=3, empty=""):
    facet_groups = defaultdict(list)
    for measurement in measurements:
        facet_groups[measurement.facet].append(measurement)
    header_html = "".join(f"<th>{h}</th>" for h in ["Facet"] + list(headers))

    rows_html = ""
    for facet in Facet:
        facet_measurements = facet_groups[facet]
        facet_rowspan = max(len(facet_measurements), 1)
        facet_cell = f"<td rowspan={facet_rowspan}>{facet.value}</td>"
        if len(facet_measurements) > 0:
            for measurement in sorted(facet_measurements, key=lambda m: m.name):
                rows_html += f"<tr>{facet_cell}{measurement.to_table_cells(headers=headers, round=round, empty=empty)}</tr>"
                facet_cell = ""  # Clear it out so we only include it once at the beginning of the group
        else:
            rows_html += f"<tr>{facet_cell}</tr>"
    return f"""
<table>
<tr>
{header_html}
</tr>
{rows_html}
</table>
"""

def format_top_level_summary(measures_map, segment=None):
    top_measures = {m for group, measures in measures_map.items() for m in measures 
                    if show_in_top_measures(m, segment=segment)}
    significant_measures = {m for group, measures in measures_map.items() for m in measures 
                            if show_in_significant_measures(m, segment=segment)}
    
    return f"""
# Top-level metrics
{format_facet_grouped_table(top_measures)}

# Significant differences
{format_facet_grouped_table(significant_measures)}
""".strip()

def show_in_top_measures(measurement, segment=None):
    relevant_segment = segment is None or (measurement.subset_type == "segment" and measurement.subset == segment)
    return (measurement.view == View.top and relevant_segment) or \
           (measurement.view == View.subset and segment is not None and relevant_segment)

def show_in_significant_measures(measurement, segment=None):
    relevant_segment = segment is None or (measurement.subset_type == "segment" and measurement.subset == segment)
    is_significant = measurement.pval is not None and measurement.pval < 0.10
    return measurement.view in (View.significance, View.subset) and relevant_segment and is_significant
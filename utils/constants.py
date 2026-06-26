from .types import Direction

BASIC_HEADERS = {
    'N': "{n}",
    'Control': "{control}",
    'Experiment': "{experiment}",
    'Diff': "{diff}",
    'Prop diff': "{prop_diff}",
    'P': "{pval}"
}

METRIC_HEADERS = {
    'Metric': "{enriched_name}",
    **BASIC_HEADERS
}

SEGMENT_HEADERS = {
    'Segment': "{subset}",
    **BASIC_HEADERS
}

TYPE_HEADERS = {
    'Type': "{subset}",
    **BASIC_HEADERS
}

IMPORTANCE_HEADERS = {
    'Importance': "{subset}",
    **BASIC_HEADERS
}

BOLD_YELLOW = ' style="background: #ffc;"'
SLIGHT_YELLOW = ' style="background: #ffe;"'
BOLD_GREEN = ' style="background: #cfc;"'
SLIGHT_GREEN = ' style="background: #efe;"'
BOLD_RED = ' style="background: #fcc;"'
SLIGHT_RED = ' style="background: #fee;"'
ROW_STYLES = {
    None: {
        'significant': {'more': BOLD_YELLOW, 'less': BOLD_YELLOW},
        'marginal': {'more': SLIGHT_YELLOW, 'less': SLIGHT_YELLOW}
    },
    Direction.more_is_better: {
        'significant': {'more': BOLD_GREEN, 'less': BOLD_RED},
        'marginal': {'more': SLIGHT_GREEN, 'less': SLIGHT_RED}
    },
    Direction.less_is_better: {
        'significant': {'more': BOLD_RED, 'less': BOLD_GREEN},
        'marginal': {'more': SLIGHT_RED, 'less': SLIGHT_GREEN}
    }
}
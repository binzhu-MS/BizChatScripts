from enum import Enum


class View(str, Enum):
    top = "top"  # Show in top-level section
    subset = "subset"  # Show in top-level section for subset or otherwise when significant
    significance = "significance"  # Show only when significant


class Direction(str, Enum):
    more_is_better = "more is better"
    less_is_better = "less is better"


class Facet(str, Enum):
    overall = "overall"
    accuracy = "accuracy"
    completeness = "completeness"
    relevance = "relevance"
    usefulness = "usefulness"
    presentation = "presentation"
    experience = "experience"
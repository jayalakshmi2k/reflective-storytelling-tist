"""
Utilities for retrieving activity-related scales and terms
from the ACKTUS ontology used by the reflective storytelling system.
"""

import re

from rdflib import Graph, Literal, Namespace, URIRef


ACKTUS = Namespace("http://www.cs.umu.se/~helena/owl-files/ACKTUS.owl#")
RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")


def fetch_and_categorize_terms(file_path):
    scale_uri = (
        "http://www.cs.umu.se/~helena/owl-files/ACKTUS.owl#"
        "Scale095c6602-33c5-4ec4-bf25-7acfa7cb5692-seq"
    )

    skip_terms = {"Other:"}

    g = Graph()
    g.parse(file_path, format="turtle")

    scale_values = []
    for subj, pred, obj in g:
        if (
            str(subj) == scale_uri
            and str(pred).startswith(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#_"
            )
        ):
            scale_values.append(str(obj))

    english_terms = []
    seen_terms = set()

    for value_uri in scale_values:
        for subj, pred, obj in g:
            if (
                str(subj) == value_uri
                and "has-term" in str(pred)
                and isinstance(obj, Literal)
                and obj.language == "en"
            ):
                term = str(obj)

                if term not in seen_terms and term not in skip_terms:
                    english_terms.append(term)
                    seen_terms.add(term)

    categories = {
        "everyday": [],
        "physical": [],
        "recovery": [],
        "social": [],
    }

    for term in english_terms:
        if term in {
            "Walk the dog",
            "Exercise",
            "Shuffle snow",
            "Work in the garden",
        }:
            categories["physical"].append(term)

        elif term in {
            "Sleep/rest",
            "Relax watching TV, movies, etc.",
            "Relax, read book, news",
            "Have dinner",
        }:
            categories["recovery"].append(term)

        elif term in {
            "Socialise",
            "Attend public event",
            "Visit restaurant or pub",
            "Take care of children",
        }:
            categories["social"].append(term)

        elif term in {
            "Clean after dinner",
            "Make dinner",
            "Clean, laundry",
            "Shop groceries",
        }:
            categories["everyday"].append(term)

        else:
            # Default to everyday if uncategorized.
            categories["everyday"].append(term)

    return {
        "scale": scale_uri,
        "categories": categories,
    }


def fetch_done_with_values(file_path):
    scale_uri = (
        "http://www.cs.umu.se/~helena/owl-files/ACKTUS.owl#"
        "Scale7b7481a1-2aee-4193-a2cb-83a8e0ac0830-seq"
    )

    g = Graph()
    g.parse(file_path, format="turtle")

    done_with_values = []

    for subj, pred, obj in g:
        if str(subj) == scale_uri and str(pred).startswith(str(RDF_NS)):
            for term in g.objects(obj, ACKTUS["has-term"]):
                if isinstance(term, Literal) and term.language == "en":
                    done_with_values.append(term.toPython())

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(done_with_values))


def fetch_motivation_options(file_path):
    scale_uri = (
        "http://www.cs.umu.se/~helena/owl-files/ACKTUS.owl#"
        "Scale3824cca0-cb88-4c97-8985-212dfc1c2202-seq"
    )

    g = Graph()

    try:
        g.parse(file_path, format="turtle")
    except Exception as exc:
        print(f"[ERROR] Failed to parse RDF file: {exc}")
        return []

    scale_values = []

    for subj, pred, obj in g:
        if str(subj) == scale_uri and re.match(rf"{RDF_NS}_\d+", str(pred)):
            match = re.search(rf"{RDF_NS}_(\d+)", str(pred))

            if match:
                scale_values.append((int(match.group(1)), str(obj)))

    scale_values.sort(key=lambda item: item[0])

    motivation_options = []

    for _, value_uri in scale_values:
        english_term = None
        fallback_term = None

        for subj, pred, obj in g:
            if (
                str(subj) == value_uri
                and pred == ACKTUS["has-term"]
                and isinstance(obj, Literal)
            ):
                if obj.language == "en":
                    english_term = obj.toPython()
                elif obj.language == "sv":
                    fallback_term = obj.toPython()

        if english_term:
            motivation_options.append(english_term)
        elif fallback_term:
            motivation_options.append(f"{fallback_term} (SV)")

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(motivation_options))


def fetch_frequency_scale(file_path):
    scale_uri = (
        "http://www.cs.umu.se/~helena/owl-files/ACKTUS.owl#"
        "Scaled3209bea-2ce9-45fd-abc4-88f36446164c-seq"
    )

    g = Graph()

    try:
        g.parse(file_path, format="turtle")
    except Exception as exc:
        print(f"[ERROR] Failed to parse RDF file: {exc}")
        return []

    scale_values = []

    for subj, pred, obj in g.triples((URIRef(scale_uri), None, None)):
        if str(pred).startswith(str(RDF_NS["_"])):
            scale_values.append(
                (int(str(pred).split("_")[-1]), str(obj))
            )

    scale_values.sort(key=lambda item: item[0])

    frequency_options = []

    for _, value_uri in scale_values[:5]:
        for subj, pred, obj in g.triples(
            (URIRef(value_uri), ACKTUS["has-term"], None)
        ):
            if isinstance(obj, Literal) and obj.language == "en":
                frequency_options.append(obj.toPython())
                break

    return frequency_options
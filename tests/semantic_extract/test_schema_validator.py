"""Unit tests for ExtractionSchema and SchemaValidator (schema-guided validation)."""

from __future__ import annotations

from semantica.semantic_extract import (
    Entity,
    ExtractionSchema,
    ExtractionValidator,
    Relation,
    SchemaValidator,
    ValidationResult,
)

ONTOLOGY = {
    "classes": [{"name": "Person"}, {"name": "Organization"}, {"label": "City"}],
    "properties": [
        {"name": "worksAt", "domain": ["Person"], "range": ["Organization"]},
        {"name": "locatedIn", "domain": "Organization", "range": "City"},
        {"name": "knows"},  # unconstrained domain / range
    ],
}

TTL = """
@prefix : <https://example.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Person a owl:Class .
:Organization a owl:Class .
:worksAt a owl:ObjectProperty ;
    rdfs:domain :Person ;
    rdfs:range :Organization .
"""


def _schema() -> ExtractionSchema:
    return ExtractionSchema.from_ontology(ONTOLOGY)


def _person() -> Entity:
    return Entity(text="Alice", label="Person", start_char=0, end_char=5)


def _org() -> Entity:
    return Entity(text="Acme", label="Organization", start_char=0, end_char=4)


def _city() -> Entity:
    return Entity(text="Paris", label="City", start_char=0, end_char=5)


def _product() -> Entity:
    return Entity(text="Widget", label="Product", start_char=0, end_char=6)


# --------------------------------------------------------------------------- #
# ExtractionSchema
# --------------------------------------------------------------------------- #


def test_from_ontology_parses_concepts_and_predicates() -> None:
    schema = _schema()
    assert schema.concepts == frozenset({"Person", "Organization", "City"})
    assert set(schema.predicates) == {"worksAt", "locatedIn", "knows"}
    assert schema.predicates["worksAt"].domain == frozenset({"Person"})
    assert schema.predicates["worksAt"].range == frozenset({"Organization"})
    # Missing domain / range means unconstrained.
    assert schema.predicates["knows"].domain == frozenset()
    assert schema.predicates["knows"].range == frozenset()


def test_allows_relation_respects_domain_range() -> None:
    schema = _schema()
    assert schema.allows_relation("Person", "worksAt", "Organization")
    assert not schema.allows_relation("Person", "worksAt", "City")  # range violation
    assert not schema.allows_relation("Person", "unknownPred", "Organization")
    assert not schema.allows_relation("Product", "worksAt", "Organization")  # off-vocab
    # Unconstrained predicate accepts any known concepts.
    assert schema.allows_relation("Person", "knows", "City")


def test_from_owl_parses_turtle() -> None:
    schema = ExtractionSchema.from_owl(TTL, format="turtle")
    assert {"Person", "Organization"} <= schema.concepts
    assert schema.predicates["worksAt"].domain == frozenset({"Person"})
    assert schema.predicates["worksAt"].range == frozenset({"Organization"})


# --------------------------------------------------------------------------- #
# SchemaValidator — entities
# --------------------------------------------------------------------------- #


def test_validate_entities_all_conforming() -> None:
    result = SchemaValidator(_schema()).validate_entities([_person(), _org(), _city()])
    assert isinstance(result, ValidationResult)
    assert result.valid
    assert result.score == 1.0
    assert result.metrics["out_of_vocabulary"] == 0


def test_validate_entities_flags_out_of_vocabulary() -> None:
    result = SchemaValidator(_schema()).validate_entities([_person(), _product()])
    assert not result.valid
    assert result.metrics["out_of_vocabulary"] == 1
    assert result.metrics["unknown_labels"] == ["Product"]
    assert result.score == 0.5
    assert result.errors


def test_validate_entities_empty_is_vacuously_valid() -> None:
    result = SchemaValidator(_schema()).validate_entities([])
    assert result.valid
    assert result.score == 1.0


def test_validate_entities_batch_returns_list_with_index() -> None:
    results = SchemaValidator(_schema()).validate_entities([[_person()], [_product()]])
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].valid
    assert not results[1].valid
    assert results[0].metadata["batch_index"] == 0
    assert results[1].metadata["batch_index"] == 1


# --------------------------------------------------------------------------- #
# SchemaValidator — relations
# --------------------------------------------------------------------------- #


def test_validate_relations_conforming() -> None:
    rels = [
        Relation(subject=_person(), predicate="worksAt", object=_org()),
        Relation(subject=_person(), predicate="knows", object=_city()),
    ]
    result = SchemaValidator(_schema()).validate_relations(rels)
    assert result.valid
    assert result.score == 1.0


def test_validate_relations_flags_unknown_predicate_and_domain_range() -> None:
    rels = [
        Relation(subject=_person(), predicate="worksAt", object=_org()),  # ok
        Relation(subject=_person(), predicate="founded", object=_org()),  # unknown pred
        Relation(subject=_person(), predicate="worksAt", object=_city()),  # range viol
    ]
    result = SchemaValidator(_schema()).validate_relations(rels)
    assert not result.valid
    assert result.metrics["unknown_predicate"] == 1
    assert result.metrics["domain_range_violation"] == 1
    assert result.metrics["conforming"] == 1
    assert result.score == 1 / 3


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def test_filter_by_schema_drops_off_vocabulary() -> None:
    kept = SchemaValidator(_schema()).filter_by_schema([_person(), _org(), _product()])
    assert [e.label for e in kept] == ["Person", "Organization"]


def test_filter_relations_by_schema_keeps_only_conforming() -> None:
    rels = [
        Relation(subject=_person(), predicate="worksAt", object=_org()),  # keep
        Relation(subject=_person(), predicate="founded", object=_org()),  # drop
        Relation(subject=_person(), predicate="worksAt", object=_city()),  # drop
        Relation(subject=_person(), predicate="knows", object=_city()),  # keep
    ]
    kept = SchemaValidator(_schema()).filter_relations_by_schema(rels)
    assert [r.predicate for r in kept] == ["worksAt", "knows"]


# --------------------------------------------------------------------------- #
# Composition with the confidence-based ExtractionValidator (orthogonal axis)
# --------------------------------------------------------------------------- #


def test_composes_with_extraction_validator_same_shape() -> None:
    entities = [_person(), _org()]
    confidence = ExtractionValidator().validate_entities(entities)
    conformance = SchemaValidator(_schema()).validate_entities(entities)
    assert isinstance(confidence, ValidationResult)
    assert isinstance(conformance, ValidationResult)

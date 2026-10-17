from typing import Any, Literal

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.utils import FunctionCallException, format_enumerate, format_section

import requests

def _construct_request(
    sparql: str,
    endpoint: str,
    headers: dict,
    params: dict,
    timeout: float = 30.0,
) -> str:
    req_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        **headers,
    }
    response = requests.post(
        endpoint,
        data={**params, "query": sparql},
        headers=req_headers,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise FunctionCallException(
            f"CONSTRUCT query failed: HTTP {response.status_code}\n"
            f"{response.text[:500]}"
        )
    return response.text

_ONTOLOGY_KG_NOTE = """\
The following knowledge graphs are ontology/vocabulary references. \
When deciding how to model entities and their predicates in the output, \
search these knowledge graphs to find well-defined classes and properties whose \
semantics and expected value types match your intended mapping. Do not generate \
CONSTRUCT queries targeting these KGs.\
"""

_STEPS_GUIDED_DESIGN = """\
You should follow a step-by-step approach:
1. Explore the knowledge graph — use the available notes (and shape profile \
if available) as a starting point, and run targeted queries where needed to \
fill gaps. Get a clear picture of what classes and entities/predicates exist, \
what the values look like, and what the data is about, and where the existing \
modelling is already sound versus where it needs improvement.
2. Design the mapping:
- Decide what entities to model, what type to assign each, and what fields represent \
inter-entity relationships.
- Decide how to map them: mint IRIs from a stable identifier of the entity itself, in a \
namespace that reflects the domain — replacing the source's if it does not.
- Identify what null looks like in the data and how to handle it in the mapping.
- Decide whether one CONSTRUCT query is enough to cover the data completely, \
or whether the relationships in the knowledge graph require multiple queries. \
There is no fixed number, but together the queries must cover all the data without loss.
- For each entity type, find a matching class for its rdf:type, and for each source \
predicate, find a matching property, by searching the ontology knowledge graphs — \
search_entity for classes and search_property for predicates are the most direct \
approach, but notes and other functions are available too. A term is a good match \
when its semantics and expected value type/domain/range align with your intended \
mapping. Only fall back to the custom namespace if no ontology term fits after a \
genuine search."""

_STEPS_CUSTOM_DESIGN = """\
You should follow a step-by-step approach:
1. Explore the knowledge graph — use the available notes (and shape profile \
if available) as a starting point, and run targeted queries where needed to \
fill gaps. Get a clear picture of what classes and entities/predicates exist, \
what the values look like, and what the data is about, and where the existing \
modelling is already sound versus where it needs improvement.
2. Design the mapping:
- Decide what entities to model, what type to assign each, and what fields represent \
inter-entity relationships.
- Decide how to map them: mint IRIs from a stable identifier of the entity itself, in a \
namespace that reflects the domain — replacing the source's if it does not.
- Identify what null looks like in the data and how to handle it in the mapping.
- Identify whether there are unit informations like currency, scale or language metadata etc. \
associated with the data and how to handle them in the mapping.\
- Decide whether one CONSTRUCT query is enough to cover the data completely, or whether the \
relationships in the knowledge graph require multiple queries. There is no fixed number — let \
what you find in the knowledge graph decide, but together the queries must cover \
all the data without loss."""

_STEPS_WRITE_TEST = """\
3. Write each CONSTRUCT query and test it independently using propose_construct. \
Give it a name the first time you test it, and reuse that same name in submit_construct. \
Check that it runs and produces non-empty, well-formed output."""

# inserted before submission, only when competency questions are given
_STEPS_COMPETENCY_CHECK = """\
{n}. Before submitting, check the queries you wrote against each competency question by \
sketching the SPARQL query that would answer it against the graph produced \
by your queries. Use it as a usability test to see if the model is semantically well-formed \
and easy to query. If the question cannot be answered against the model at all, the design is \
missing something that has to be fixed. Do not narrow or make the model redundant \
to make the competency questions easier to answer."""

_STEPS_SUBMIT = """\
{n}. Once every query you planned is ready, submit all queries together using submit_construct.
"""


def _steps(
    ontology_mode: Literal["guided", "custom"],
    competency_questions: list[str],
) -> str:
    parts = [
        _STEPS_GUIDED_DESIGN
        if ontology_mode == "guided"
        else _STEPS_CUSTOM_DESIGN
    ]
    parts.append(_STEPS_WRITE_TEST)
    n = 4
    if competency_questions:
        parts.append(_STEPS_COMPETENCY_CHECK.format(n=n))
        n += 1
    parts.append(_STEPS_SUBMIT.format(n=n))
    return "\n".join(parts)

_COMPETENCY_QUESTIONS_NOTE = """\
Treat each competency question listed below as a query someone may later write against your \
output knowledge graph, and use it as a semantic usability test of the model that you propose: if \
the query answering it is overly complex - that verbosity of the SPARQL query may point \
to a gap in the semantic modelling itself that can be improved unless the underlying \
data requires such modeling.

The list of competency questions is not exhaustive — it states the minimum the \
output knowledge graph must be able to answer, not the full scope of it. \
Modelling all of the source data completely still applies.
"""


def _system_base(kg_name: str) -> str:
    return (
        f'You are a semantic RDF generation assistant. Your task is to analyse the '
        f'"{kg_name}" knowledge graph and produce one or more SPARQL CONSTRUCT '
        f'queries that model it as a well-formed, semantically correct knowledge graph. '
        f'The knowledge graph may already model some of its data well — where it does, '
        f'preserve or refine that modelling; where it doesn\'t, remodel it properly. The goal '
        f'is the best possible semantic model of the input data.'
    )

def system_information(
    ontology_mode: Literal["guided", "custom"],
    data_kg_name: str,
    ontology_kg_names: list[str],
    competency_questions: list[str] | None = None,
) -> str:
    competency_questions = competency_questions or []

    parts = [_system_base(data_kg_name)]
    if ontology_mode == "guided" and ontology_kg_names:
        parts.append(
            _ONTOLOGY_KG_NOTE
            + "\nOntology KGs: "
            + ", ".join(f'"{n}"' for n in ontology_kg_names)
        )
    parts.append(_steps(ontology_mode, competency_questions))
    if competency_questions:
        parts.append(_COMPETENCY_QUESTIONS_NOTE)
        parts.append(
            format_section(
                "Competency questions",
                format_enumerate(competency_questions),
            )
        )
    return "\n\n".join(parts)

_RULES_COMMON = [
"Always add rdfs:label to every entity in the CONSTRUCT but never make the label \
a precondition for the entity's other triples — a missing label shouldn't drop \
the entity from the query.",
"Where a property may be missing for a particular entity, generate it independently (e.g. with \
OPTIONAL) so a missing value doesn't suppress triples for the other \
properties of the same entity — but only where the data is actually \
missing, not as a blanket pattern for every property.",
"Never emit a triple with a null marker itself as the value.",
"If the same entity type is referenced by more than one query, mint its \
IRIs using the exact same pattern in every query that touches it, and reuse the same predicate \
names and namespace for the same concept across queries, so the queries \
combine into one coherent, non-conflicting graph.",
"A field name may carry information the value itself doesn't — a unit, a currency, a scale. \
Keep that information in the model rather than losing it with the name.",
"When resubmitting after reviewer feedback, send only the queries you changed \
or added; to change a query, send it again under the same name. Queries you do \
not send are kept as they are. To delete a query, list it in the `removed` \
argument with the reason. The reply lists the full current set; check that it \
is what you intend."
]

class ConstructGenTask(GraspTask):
    name = "construct-gen"

    @property
    def include_common_prefixes(self) -> bool:
        return False

    def __init__(
        self,
        managers: list[KgManager],
        config: GraspConfig,
        known: set[str] | None = None,
        example_indices: dict | None = None,
    ) -> None:
        super().__init__(managers, config, known, example_indices)

        self.data_manager = self.managers[0]
        self.ontology_managers = self.managers[1:]

        self.ontology_mode: Literal["guided", "custom"] = "custom"
        self.competency_questions: list[str] = []

        # output state: the current set of submitted queries, by name
        self.queries: dict[str, dict] = {}
        # whether the most recent submit_construct call was accepted
        self.submit_ok = False
        # queries the model explicitly withdrew, with its reasons
        self.removed_queries: list[dict] = []
        # verdict of the last review round
        self.review: dict | None = None


    def system_information(self) -> str:
        return system_information(
            self.ontology_mode,
            self.data_manager.kg,
            [m.kg for m in self.ontology_managers],
            self.competency_questions,
        )

    def rules(self) -> list[str]:
        return _RULES_COMMON

    def function_definitions(self) -> list[dict]:
        return [
            {
                "name": "propose_construct",
                "description": (
                    "Test a SPARQL CONSTRUCT query by running it with LIMIT 10. "
                    "Returns the first triples produced so you can verify the output "
                    "looks semantically correct. Use this to iteratively refine your "
                    "query before submitting. Give each query a short unique identifier."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sparql": {
                            "type": "string",
                            "description": "The SPARQL CONSTRUCT query to test",
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Short, descriptive identifier for this query. Use the "
                                "same name when you later include it in "
                                "submit_construct."
                            ),
                        },
                    },
                    "required": ["sparql", "name"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "submit_construct",
                "description": (
                    "Submit or update the SPARQL CONSTRUCT queries. The current "
                    "set will be saved and executed to produce the semantic "
                    "RDF output. Only call this once you are satisfied with the "
                    "queries after testing them with propose_construct."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": (
                                "On your first call, all queries. On later calls, "
                                "only queries that are new or changed. A query with "
                                "the same name as a submitted one replaces it; "
                                "submitted queries you do not send are kept unchanged."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": (
                                            "Short, unique, descriptive identifier "
                                            "for this query."
                                        ),
                                    },
                                    "sparql": {
                                        "type": "string",
                                        "description": "The SPARQL CONSTRUCT query.",
                                    },
                                },
                                "required": ["name", "sparql"],
                                "additionalProperties": False,
                            },
                        },
                        "removed": {
                            "type": ["array", "null"],
                            "description": (
                                "Submitted queries to delete, each with the reason "
                                "it is no longer needed. Use null or an empty list "
                                "when nothing is deleted."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": (
                                            "Name of the previously submitted query "
                                            "being dropped."
                                        ),
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": (
                                            "Why this query is no longer part of the "
                                            "set, and where its data is covered now "
                                            "if it still is."
                                        ),
                                    },
                                },
                                "required": ["name", "reason"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["queries", "removed"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        if fn_name == "propose_construct":
            sparql = fn_args["sparql"].strip()
            name = fn_args["name"].strip()

            if "LIMIT" not in sparql.upper():
                sparql += "\nLIMIT 10"

            manager = self.data_manager

            try:
                result = _construct_request(
                    sparql,
                    manager.endpoint,
                    manager.headers,
                    manager.params,
                    timeout=self.config.sparql_query_timeout,
                )
            except FunctionCallException as e:
                return str(e)
            except Exception as e:
                return f"Error executing CONSTRUCT query:\n{e}"

            lines = [ln for ln in result.strip().splitlines() if ln.strip()]
            if not lines:
                return (
                    f"Query '{name}' returned no triples. "
                    "Check your WHERE clause, table filter, and NULL handling."
                )

            preview = "\n".join(lines[:20]) # at most 20 triples
            return f"Preview for '{name}' ({len(lines)} triples with LIMIT 10):\n{preview}"

        elif fn_name == "submit_construct":
            self.submit_ok = False
            queries = fn_args.get("queries") or []
            removed = fn_args.get("removed") or []

            if not queries and not removed:
                return (
                    "Your call contained no queries and no removals, so nothing "
                    "was changed. Call submit_construct again."
                )

            names = [q["name"].strip() for q in queries]
            if any(not n for n in names):
                return (
                    "Every query needs a non-empty name. Please resubmit with "
                    "a descriptive name for each query."
                )
 
            seen: set[str] = set()
            duplicates: set[str] = set()
            for n in names:
                if n in seen:
                    duplicates.add(n)
                seen.add(n)
            if duplicates:
                return (
                    "Query names must be unique. Found duplicate name(s): "
                    f"{', '.join(sorted(duplicates))}. Please resubmit with a "
                    "unique name for each query."
                )

            unknown = [r["name"].strip() for r in removed if r["name"].strip() not in self.queries]
            if unknown:
                return (
                    f"Cannot remove {', '.join(unknown)}: no submitted query has "
                    "that name. Nothing was changed."
                )

            # update a copy, so a rejected call leaves the current set untouched
            new = dict(self.queries)
            for r in removed:
                new.pop(r["name"].strip())
            added = [n for n in names if n not in new]
            replaced = [n for n in names if n in new]
            for q in queries:
                new[q["name"].strip()] = {"name": q["name"].strip(), "sparql": q["sparql"].strip()}

            if not new:
                return "This would leave no queries at all. Nothing was changed."

            self.queries = new
            self.removed_queries.extend( # accumulated reasons across rounds
                {"name": r["name"].strip(), "reason": r["reason"].strip()}
                for r in removed
            )
            self.submit_ok = True
            return (
                f"CONSTRUCT queries saved. Added: {', '.join(added) or 'none'}. "
                f"Replaced: {', '.join(replaced) or 'none'}. "
                f"Removed: {', '.join(r['name'].strip() for r in removed) or 'none'}. "
                f"Current set ({len(new)}): {', '.join(new)}."
            )
 
        raise FunctionCallException(f"Unknown function: {fn_name}")
 
    def done(self, fn_name: str) -> bool:
        return fn_name == "submit_construct" and self.submit_ok

    def setup(self, input: Any) -> str:
        assert isinstance(input, dict), "Input for construct-gen must be a dict"
        self.ontology_mode = input.get("ontology_mode", "custom")
        self.competency_questions = input.get("competency_questions", []) or []
        return (
            f'Analyse the "{self.data_manager.kg}" knowledge graph and generate '
            f"one or more SPARQL CONSTRUCT queries that model it as a well-formed, "
            f"semantically correct knowledge graph."
        )

    def output(self, messages: list[Message]) -> dict | None:
        if not self.queries:
            return None

        queries = list(self.queries.values())
        blocks = "\n\n".join(
            f"### {q['name']}\n```sparql\n{q['sparql']}\n```"
            for q in queries
        )
        return {
            "type": "output",
            "queries": queries,
            "removed": self.removed_queries,
            "kg": self.data_manager.kg,
            "review": self.review,
            "formatted": (
                f"CONSTRUCT queries for {self.data_manager.kg}:\n\n{blocks}"
            ),
        }


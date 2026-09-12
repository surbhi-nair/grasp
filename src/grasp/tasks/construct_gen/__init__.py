from typing import Any, Literal

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.utils import FunctionCallException

import requests

def _construct_request(
    sparql: str,
    endpoint: str,
    headers: dict,
    params: dict,
    timeout: float = 30.0,
) -> str:
    req_params = {**params, "query": sparql}
    req_headers = {**headers, "Accept": "text/plain"}
    response = requests.get(
        endpoint,
        params=req_params,
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

_STEPS_GUIDED = """\
You should follow a step-by-step approach:
1. Explore the knowledge graph — use the available notes (and shape profile \
if available) as a starting point, and run targeted queries where needed to \
fill gaps. Get a clear picture of what classes and entities/predicates exist, \
what the values look like, and what the data is about, and where the existing \
modelling is already sound versus where it needs improvement.
2. Design the mapping:
- Decide what entities to model, how to construct their IRIs from stable identifiers \
in the data, what type to assign them, what fields represent inter-entity relationships \
and what consistent domain-specific namespace to use.
- Decide whether one CONSTRUCT query is enough to cover the data completely, \
or whether the relationships in the knowledge graph require multiple queries. \
There is no fixed number, but together the queries must cover all the data without loss.
- For each entity type, find a matching class for its rdf:type, and for each source \
predicate, find a matching property, by searching the ontology knowledge graphs — \
search_entity for classes and search_property for predicates are the most direct \
approach, but notes and other functions are available too. A term is a good match \
when its semantics and expected value type/domain/range align with your intended \
mapping. Only fall back to the custom namespace if no ontology term fits after a \
genuine search.
3. Write each CONSTRUCT query and test it independently using propose_construct. \
Give it a name the first time you test it, and reuse that same name in submit_construct. \
Check that it runs and produces non-empty, well-formed output.
4. Once every query you planned is ready, submit all queries together using submit_construct.
"""

_STEPS_CUSTOM = """\
You should follow a step-by-step approach:
1. Explore the knowledge graph — use the available notes (and shape profile \
if available) as a starting point, and run targeted queries where needed to \
fill gaps. Get a clear picture of what classes and entities/predicates exist, \
what the values look like, and what the data is about, and where the existing \
modelling is already sound versus where it needs improvement.
2. Design the mapping:
- Decide what entities to model, how to construct their IRIs from stable identifiers \
in the data, what type to assign them, and what consistent domain-specific namespace to use.
- Decide whether one CONSTRUCT query is enough to cover the data completely, or whether the \
relationships in the knowledge graph require multiple queries. There is no fixed number — let \
what you find in the knowledge graph decide, but together the queries must cover \
all the data without loss.
3. Write each CONSTRUCT query and test it independently using propose_construct. \
Give it a name the first time you test it, and reuse that same name in submit_construct. \
Check that it runs and produces non-empty, well-formed output.
4. Once every query you planned is ready, submit all queries together using submit_construct.
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
) -> str:
    parts = [_system_base(data_kg_name)]
    if ontology_mode == "guided" and ontology_kg_names:
        parts.append(
            _ONTOLOGY_KG_NOTE
            + "\nOntology KGs: "
            + ", ".join(f'"{n}"' for n in ontology_kg_names)
        )
    steps = _STEPS_GUIDED if ontology_mode == "guided" else _STEPS_CUSTOM
    parts.append(steps)
    return "\n\n".join(parts)

_RULES_COMMON = [
"Always add rdfs:label to every entity in the CONSTRUCT.",
"Construct entity IRIs from a reliable unique identifier so the same \
entity always maps to the same IRI.",
"Not all fields may have a value for every entity — some might be missing \
or carry a dataset-specific null marker. Inspect the data to identify what \
null looks like in this knowledge graph before deciding how to handle it.",
"Where a field can be missing, generate it independently (e.g. with \
OPTIONAL) so a missing value doesn't suppress triples for the other \
fields of the same entity — but only where the data is actually \
nullable, not as a blanket pattern for every predicate. Never emit a \
triple with the null marker itself as the value.",
"If the same entity type is referenced by more than one query, mint its \
IRIs using the exact same pattern (the same identifier source and the same \
IRI template) in every query that touches it, and reuse the same predicate \
names and namespace for the same concept across queries, so the queries \
combine into one coherent, non-conflicting graph.",
"When resubmitting after reviewer feedback, always call submit_construct \
with the complete set of queries, including any that did not change — not \
just the one(s) you fixed."
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
    ) -> None:
        super().__init__(managers, config, known)

        self.data_manager = self.managers[0]
        self.ontology_managers = self.managers[1:]

        self.ontology_mode: Literal["guided", "custom"] = "custom"

        # output state
        self.final_queries: list[dict] | None = None


    def system_information(self) -> str:
        return system_information(
            self.ontology_mode,
            self.data_manager.kg,
            [m.kg for m in self.ontology_managers],
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
                    "Submit the final set of SPARQL CONSTRUCT query (or queries). "
                    "The query will be saved and executed to produce the semantic "
                    "RDF output. Only call this once you are satisfied with the "
                    "query after testing it with propose_construct."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": "The final list of SPARQL CONSTRUCT queries.",
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
                    },
                    "required": ["queries"],
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

            preview = "\n".join(lines[:20])
            return f"Preview for '{name}' ({len(lines)} triples with LIMIT 10):\n{preview}"

        elif fn_name == "submit_construct":
            queries = fn_args["queries"]
            self.final_queries = None

            if not queries:
                return "At least one CONSTRUCT query must be submitted."

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
 
            self.final_queries = [
                {"name": q["name"].strip(), "sparql": q["sparql"].strip()}
                for q in queries
            ]
            submitted = ", ".join(f"`{q['name']}`" for q in self.final_queries)
            return f"CONSTRUCT queries submitted ({submitted}). Saving output."
 
        raise FunctionCallException(f"Unknown function: {fn_name}")
 
    def done(self, fn_name: str) -> bool:
        return fn_name == "submit_construct" and self.final_queries is not None

    def setup(self, input: Any) -> str:
        assert isinstance(input, dict), "Input for construct-gen must be a dict"
        self.ontology_mode = input.get("ontology_mode", "custom")
        return (
            f'Analyse the "{self.data_manager.kg}" knowledge graph and generate '
            f"one or more SPARQL CONSTRUCT queries that model it as a well-formed, "
            f"semantically correct knowledge graph."
        )

    def output(self, messages: list[Message]) -> dict | None:
        if self.final_queries is None:
            return None
 
        blocks = "\n\n".join(
            f"### {q['name']}\n```sparql\n{q['sparql']}\n```"
            for q in self.final_queries
        )
        return {
            "type": "output",
            "queries": self.final_queries,
            "kg": self.data_manager.kg,
            "formatted": (
                f"CONSTRUCT queries for {self.data_manager.kg}:\n\n{blocks}"
            ),
        }


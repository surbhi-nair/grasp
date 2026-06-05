import os
from typing import Any

import requests

from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.utils import FunctionCallException


def system_information() -> str:
    return """\
You are a semantic RDF generation assistant. \
Your task is to analyse a knowledge graph and produce a SPARQL CONSTRUCT \
query that transforms it into semantically meaningful RDF.

You should follow a step-by-step approach:
1. Study the structural notes and shape profile already provided to \
understand the knowledge graph — what entities it contains, what \
properties they have, and how values are structured. Use get_shape or \
search_shape to get the full predicate inventory and cardinalities for \
each entity type.
2. Only if specific uncertainties remain after step 1, run targeted \
queries or searches to resolve them — for example to verify a stable \
identifier, inspect value patterns, or check for missing values.
3. Design the mapping: decide on entity IRI patterns, type assignments, \
predicate mappings to standard ontology terms, type casting, and how to \
handle missing or multi-valued fields.
4. Write the CONSTRUCT query and test it using propose_construct. \
Inspect the resulting triples and refine as needed.
5. Once satisfied, call submit_construct to finalise the output."""


def rules() -> list[str]:
    return [
        "Use the structural notes and shape profile as your primary source. \
Only run additional queries to resolve specific gaps they do not cover.",
        "Always add rdfs:label to every entity in the CONSTRUCT output.",
        "For property mappings, use standard terms where they fit precisely \
without compromise. For everything else, prefer a consistent \
domain-specific namespace that reflects the actual data rather than \
forcing ill-fitting terms from general vocabularies.",
        "Mint stable IRIs for entities using a reliable unique identifier \
present in the data.",
        "Filter out missing or empty values so they do not appear as triples \
in the output.",
        "Use UNION blocks for multi-valued or optional fields.",
        "Always call propose_construct at least once before submit_construct.",
    ]

def _construct_request(
    sparql: str,
    endpoint: str,
    headers: dict,
    params: dict,
    timeout: float = 120.0,
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
            f"SPARQL endpoint returned HTTP {response.status_code}:\n"
            f"{response.text[:500]}"
        )
    return response.text


class ConstructGenTask(GraspTask):
    name = "construct-gen"

    def __init__(
        self,
        managers: list[KgManager],
        config: GraspConfig,
        known: set[str] | None = None,
    ) -> None:
        super().__init__(managers, config, known)
        self.final_query: str | None = None
        self.final_kg: str | None = None

    def system_information(self) -> str:
        return system_information()

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        return [
            {
                "name": "propose_construct",
                "description": (
                    "Test a SPARQL CONSTRUCT query by running it with LIMIT 10. "
                    "Returns the first triples produced so you can verify the output "
                    "looks semantically correct. Use this to iteratively refine your "
                    "query before submitting."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to run the query against",
                        },
                        "sparql": {
                            "type": "string",
                            "description": "The SPARQL CONSTRUCT query to test",
                        },
                    },
                    "required": ["kg", "sparql"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "submit_construct",
                "description": (
                    "Submit the final SPARQL CONSTRUCT query. "
                    "The query will be saved and executed to produce the semantic RDF output. "
                    "Only call this when satisfied with the propose_construct output."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to run the query against",
                        },
                        "sparql": {
                            "type": "string",
                            "description": "The final SPARQL CONSTRUCT query",
                        },
                    },
                    "required": ["kg", "sparql"],
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
        manager, _ = find_manager(self.managers, fn_args["kg"])

        if fn_name == "propose_construct":
            sparql = fn_args["sparql"].strip()
            if "LIMIT" not in sparql.upper():
                sparql += "\nLIMIT 10"
            try:
                result = _construct_request(
                    sparql,
                    manager.endpoint,
                    manager.headers,
                    manager.params,
                    timeout=30.0,
                )
            except FunctionCallException as e:
                return str(e)
            except Exception as e:
                return f"Error executing CONSTRUCT query:\n{e}"

            lines = [l for l in result.strip().splitlines() if l.strip()]
            if not lines:
                return (
                    "Query returned no triples. "
                    "Check your WHERE clause, table filter, and NULL handling."
                )
            preview = "\n".join(lines[:20])
            return f"Preview ({len(lines)} triples with LIMIT 10):\n{preview}"

        elif fn_name == "submit_construct":
            # just store the query, handler does the actual execution and saving
            self.final_query = fn_args["sparql"]
            self.final_kg = fn_args["kg"]
            return "CONSTRUCT query submitted. Saving output."

        raise FunctionCallException(f"Unknown function: {fn_name}")

    def done(self, fn_name: str) -> bool:
        return fn_name == "submit_construct"

    def setup(self, input: Any) -> str:
        if isinstance(input, str) and input.strip():
            return input
        return (
            "Analyse the generic RDF knowledge graph and generate a "
            "SPARQL CONSTRUCT query that transforms it into semantic RDF."
        )

    def output(self, messages: list[Message]) -> dict | None:
        if self.final_query is None:
            return None
        return {
            "type": "output",
            "sparql": self.final_query,
            "kg": self.final_kg,
            "formatted": (
                f"CONSTRUCT query for {self.final_kg}:\n"
                f"```sparql\n{self.final_query}\n```"
            ),
        }

    @property
    def default_input_field(self) -> str | None:
        return None
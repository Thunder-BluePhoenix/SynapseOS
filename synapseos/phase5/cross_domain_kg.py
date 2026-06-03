"""
Phase 5 — Cross-Domain Knowledge Graph Expansion.

The Phase 1 KG is purely ERP/Frappe domain. The creativity engine's
analogy finder scores cross-domain concepts, but with only one domain
the analogies are intra-ERP (e.g. "Sales Invoice ↔ Purchase Invoice").

This module adds concept nodes from four non-ERP domains:
  - software_engineering : algorithms, data structures, design patterns
  - biology              : cells, metabolism, adaptation, signaling
  - physics              : systems, equilibrium, forces, feedback loops
  - economics            : markets, supply/demand, optimization, constraints

Structural attributes (degree, submittable-equivalent, child-equivalent)
are mapped consistently so the existing CreativityEngine scoring works
without modification.

The expanded graph is saved to:
    data/graph/knowledge_graph_expanded.json

CreativityEngine can be pointed at this file for richer cross-domain
analogy discovery.

Run:
    python -m synapseos.phase5.cross_domain_kg
    python -m synapseos.phase5.cross_domain_kg --output data/graph/knowledge_graph_expanded.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Domain concept definitions
# ---------------------------------------------------------------------------

@dataclass
class ConceptSpec:
    id: str
    module: str            # maps to ERP "module" field — drives domain classification
    is_submittable: bool   # True → the concept has a "commit/finalise" state
    is_child_table: bool   # True → the concept is a sub-component of another
    description: str
    links: list[str]       # IDs of concepts this one connects to


# Each domain is a list of concepts plus edges between them.
# We keep a consistent schema so CreativityEngine's domain classifier
# (which reads `module`) handles them automatically.

SOFTWARE_ENGINEERING: list[ConceptSpec] = [
    ConceptSpec("Algorithm",         "Software Engineering", True,  False, "Deterministic procedure for solving a problem", ["DataStructure", "ComplexityClass", "OptimizationProblem"]),
    ConceptSpec("DataStructure",     "Software Engineering", False, True,  "Container organizing data for efficient access",   ["Algorithm", "MemoryModel"]),
    ConceptSpec("DesignPattern",     "Software Engineering", True,  False, "Reusable solution to a recurring design problem",  ["Interface", "AbstractClass", "Component"]),
    ConceptSpec("API",               "Software Engineering", True,  False, "Contract defining interaction between components",  ["Endpoint", "AuthToken", "RateLimiter"]),
    ConceptSpec("Cache",             "Software Engineering", False, True,  "Fast storage layer reducing repeated computation",  ["DataStructure", "EvictionPolicy", "Invalidation"]),
    ConceptSpec("Queue",             "Software Engineering", False, True,  "FIFO structure decoupling producer and consumer",   ["Worker", "DeadLetterQueue", "Backpressure"]),
    ConceptSpec("Index",             "Software Engineering", False, True,  "Secondary structure accelerating lookup",           ["DataStructure", "Query", "TradeoffWriteRead"]),
    ConceptSpec("StateMachine",      "Software Engineering", True,  False, "Formal model of states and transitions",           ["State", "Transition", "Guard", "Event"]),
    ConceptSpec("Transaction",       "Software Engineering", True,  False, "Atomic, isolated, durable unit of work",           ["LockManager", "RollbackLog", "Commit"]),
    ConceptSpec("Interface",         "Software Engineering", False, False, "Abstract contract without implementation",         ["AbstractClass", "DesignPattern"]),
    ConceptSpec("Scheduler",         "Software Engineering", True,  False, "Assigns resources to tasks over time",             ["Queue", "Worker", "Priority"]),
    ConceptSpec("EventStream",       "Software Engineering", True,  False, "Ordered sequence of immutable event records",      ["Queue", "Consumer", "Producer"]),
    ConceptSpec("Worker",            "Software Engineering", False, True,  "Execution unit consuming items from a queue",      ["Queue", "Scheduler"]),
    ConceptSpec("RateLimiter",       "Software Engineering", False, True,  "Throttles request rate to protect a resource",     ["API", "Queue"]),
    ConceptSpec("CircuitBreaker",    "Software Engineering", False, True,  "Stops calls to a failing service to allow recovery", ["API", "StateMachine"]),
    ConceptSpec("LoadBalancer",      "Software Engineering", True,  False, "Distributes requests across a pool of replicas",   ["API", "Worker", "HealthCheck"]),
    ConceptSpec("HealthCheck",       "Software Engineering", False, True,  "Periodic probe confirming a node is alive",        ["LoadBalancer", "Scheduler"]),
    ConceptSpec("EvictionPolicy",    "Software Engineering", False, True,  "Rule deciding what to remove from cache",          ["Cache"]),
    ConceptSpec("OptimizationProblem","Software Engineering",True,  False, "Problem seeking minimum cost subject to constraints", ["Constraint", "Algorithm"]),
    ConceptSpec("Constraint",        "Software Engineering", False, True,  "Restriction limiting valid solution space",        ["OptimizationProblem"]),
]

BIOLOGY: list[ConceptSpec] = [
    ConceptSpec("Cell",              "Biology", False, True,  "Fundamental unit of life; encapsulates state and metabolism", ["Membrane", "Nucleus", "Organelle", "Metabolism"]),
    ConceptSpec("Organism",          "Biology", True,  False, "Complete living system composed of cells",                   ["Cell", "Organ", "Tissue", "Homeostasis"]),
    ConceptSpec("Metabolism",        "Biology", True,  False, "Totality of chemical reactions sustaining life",             ["Enzyme", "Substrate", "EnergyBalance"]),
    ConceptSpec("Enzyme",            "Biology", False, True,  "Protein catalyst accelerating a specific reaction",          ["Substrate", "Inhibitor", "Cofactor"]),
    ConceptSpec("Receptor",          "Biology", False, True,  "Surface protein binding signal molecules and transducing",   ["Ligand", "SignalCascade"]),
    ConceptSpec("SignalCascade",      "Biology", True,  False, "Chain of molecular events amplifying an initial signal",    ["Receptor", "Transcription", "GeneExpression"]),
    ConceptSpec("Homeostasis",       "Biology", True,  False, "Active maintenance of internal stability",                   ["FeedbackLoop", "Sensor", "Effector"]),
    ConceptSpec("FeedbackLoop",      "Biology", False, True,  "Output fed back as input to regulate a system",              ["Homeostasis", "Homeostasis"]),
    ConceptSpec("GeneExpression",    "Biology", True,  False, "Process decoding DNA into functional protein",               ["Transcription", "Translation", "Regulation"]),
    ConceptSpec("Adaptation",        "Biology", True,  False, "Long-term change in a population improving fitness",         ["NaturalSelection", "GeneticVariation"]),
    ConceptSpec("NaturalSelection",  "Biology", True,  False, "Differential survival and reproduction based on fitness",    ["Adaptation", "GeneticVariation"]),
    ConceptSpec("Membrane",          "Biology", False, True,  "Selective barrier controlling substance passage",            ["Receptor", "TransportProtein"]),
    ConceptSpec("Organ",             "Biology", False, True,  "Group of tissues performing a specialized function",         ["Tissue", "Organism"]),
    ConceptSpec("Tissue",            "Biology", False, True,  "Collection of similar cells with common function",           ["Cell", "Organ"]),
    ConceptSpec("Inhibitor",         "Biology", False, True,  "Molecule reducing enzyme activity",                          ["Enzyme", "Regulation"]),
    ConceptSpec("Regulation",        "Biology", False, True,  "Control mechanism maintaining process within bounds",        ["GeneExpression", "Metabolism", "FeedbackLoop"]),
    ConceptSpec("EnergyBalance",     "Biology", False, True,  "Equilibrium between energy intake and expenditure",          ["Metabolism", "Homeostasis"]),
    ConceptSpec("GeneticVariation",  "Biology", False, True,  "Diversity in DNA sequences across individuals",              ["Adaptation", "NaturalSelection"]),
    ConceptSpec("Nucleus",           "Biology", False, True,  "Membrane-enclosed organelle housing DNA",                    ["GeneExpression", "Cell"]),
    ConceptSpec("Organelle",         "Biology", False, True,  "Specialized structure within a cell",                        ["Cell", "Metabolism"]),
]

PHYSICS: list[ConceptSpec] = [
    ConceptSpec("System",            "Physics", True,  False, "Defined region of space with boundary conditions",           ["Boundary", "ExternalForce", "State_P"]),
    ConceptSpec("Equilibrium",       "Physics", True,  False, "State where net forces and torques are zero",                ["System", "FeedbackMechanism"]),
    ConceptSpec("Force",             "Physics", False, True,  "Influence causing or preventing change in motion",           ["System", "Momentum", "WorkEnergy"]),
    ConceptSpec("Gradient",          "Physics", False, True,  "Rate of change of a quantity across space",                  ["DiffusionProcess", "FieldLine"]),
    ConceptSpec("DiffusionProcess",  "Physics", True,  False, "Net movement of particles down a concentration gradient",    ["Gradient", "Membrane_P", "Entropy"]),
    ConceptSpec("Entropy",           "Physics", False, True,  "Measure of disorder or unavailable energy in a system",      ["System", "Equilibrium"]),
    ConceptSpec("FeedbackMechanism", "Physics", False, True,  "Process where output modifies the input to maintain state",  ["Equilibrium", "System"]),
    ConceptSpec("FieldLine",         "Physics", False, True,  "Visual representation of a vector field direction",          ["Force", "Gradient"]),
    ConceptSpec("Resonance",         "Physics", True,  False, "Amplified oscillation at a system's natural frequency",      ["System", "OscillatingSystem"]),
    ConceptSpec("OscillatingSystem", "Physics", True,  False, "System with periodic motion around an equilibrium",          ["Resonance", "Equilibrium", "Force"]),
    ConceptSpec("PhaseTransition",   "Physics", True,  False, "Abrupt change in system state at a threshold",               ["System", "Entropy", "Equilibrium"]),
    ConceptSpec("ConservationLaw",   "Physics", True,  False, "Invariant quantity unchanged by system evolution",           ["System", "Symmetry"]),
    ConceptSpec("Symmetry",          "Physics", False, True,  "Invariance under a transformation",                          ["ConservationLaw"]),
    ConceptSpec("State_P",           "Physics", False, True,  "Complete description of a system at an instant",             ["System", "PhaseTransition"]),
    ConceptSpec("Boundary",          "Physics", False, True,  "Interface separating system from environment",               ["System", "Force"]),
    ConceptSpec("WorkEnergy",        "Physics", False, True,  "Scalar quantity measuring capacity to do work",              ["Force", "System"]),
    ConceptSpec("Momentum",          "Physics", False, True,  "Product of mass and velocity; conserved quantity",           ["Force", "ConservationLaw"]),
    ConceptSpec("Membrane_P",        "Physics", False, True,  "Permeable barrier separating regions of different potential", ["DiffusionProcess", "Gradient"]),
    ConceptSpec("ExternalForce",     "Physics", False, True,  "Force originating outside the defined system",               ["System", "Force"]),
    ConceptSpec("Attractor",         "Physics", False, True,  "State towards which a system tends to evolve",               ["Equilibrium", "System", "FeedbackMechanism"]),
]

ECONOMICS: list[ConceptSpec] = [
    ConceptSpec("Market",            "Economics", True,  False, "Mechanism coordinating buyers and sellers via price",       ["Supply", "Demand", "PriceSignal", "Arbitrage"]),
    ConceptSpec("Supply",            "Economics", False, True,  "Quantity producers offer at each price level",              ["Market", "ProductionFunction", "MarginalCost"]),
    ConceptSpec("Demand",            "Economics", False, True,  "Quantity consumers seek at each price level",               ["Market", "UtilityFunction", "ElasticityOfDemand"]),
    ConceptSpec("PriceSignal",       "Economics", False, True,  "Information conveyed by price about scarcity",              ["Market", "Arbitrage"]),
    ConceptSpec("Arbitrage",         "Economics", True,  False, "Risk-free profit from price discrepancies across markets",  ["Market", "PriceSignal"]),
    ConceptSpec("Equilibrium_E",     "Economics", True,  False, "State where supply equals demand at the clearing price",   ["Market", "Supply", "Demand"]),
    ConceptSpec("Incentive",         "Economics", False, True,  "Reward or penalty shaping agent behaviour",                 ["PrincipalAgent", "MoralHazard"]),
    ConceptSpec("PrincipalAgent",    "Economics", True,  False, "Relationship where one party acts on behalf of another",   ["Incentive", "InformationAsymmetry"]),
    ConceptSpec("InformationAsymmetry","Economics",False, True, "One party has more or better information than another",    ["PrincipalAgent", "AdverseSelection"]),
    ConceptSpec("AdverseSelection",  "Economics", True,  False, "Bad outcomes from information asymmetry before contract",  ["InformationAsymmetry"]),
    ConceptSpec("MoralHazard",       "Economics", True,  False, "Risk-taking when costs are borne by another party",        ["Incentive", "PrincipalAgent"]),
    ConceptSpec("ExternalityEffect", "Economics", False, True,  "Cost or benefit falling on uninvolved third parties",      ["Market", "PriceSignal"]),
    ConceptSpec("OptimizationAgenda","Economics", True,  False, "Formal maximization subject to resource constraints",      ["Constraint_E", "UtilityFunction"]),
    ConceptSpec("Constraint_E",      "Economics", False, True,  "Boundary limiting feasible choices",                       ["OptimizationAgenda"]),
    ConceptSpec("UtilityFunction",   "Economics", False, True,  "Numeric representation of agent preferences",              ["Demand", "OptimizationAgenda"]),
    ConceptSpec("ProductionFunction","Economics", False, True,  "Mapping from inputs to maximum possible output",           ["Supply", "MarginalCost"]),
    ConceptSpec("MarginalCost",      "Economics", False, True,  "Cost of producing one additional unit",                    ["ProductionFunction", "Supply"]),
    ConceptSpec("ElasticityOfDemand","Economics", False, True,  "Sensitivity of demand quantity to price change",           ["Demand", "PriceSignal"]),
    ConceptSpec("NetworkEffect",     "Economics", True,  False, "Value increases as more users join a platform",            ["Market", "Incentive"]),
    ConceptSpec("Liquidity",         "Economics", False, True,  "Ease of converting an asset to cash without loss",        ["Market", "Arbitrage"]),
]

ALL_DOMAINS: list[list[ConceptSpec]] = [
    SOFTWARE_ENGINEERING,
    BIOLOGY,
    PHYSICS,
    ECONOMICS,
]


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------

class CrossDomainKGExpander:
    """
    Loads the existing ERP knowledge graph and merges cross-domain concepts.
    Produces an expanded graph compatible with CreativityEngine.
    """

    def __init__(self, base_kg_path: str = "data/graph/knowledge_graph.json"):
        self.base_path = Path(base_kg_path)
        self._base_data: dict[str, Any] = {}
        self._load_base()

    def _load_base(self) -> None:
        if self.base_path.exists():
            self._base_data = json.loads(self.base_path.read_text())
        else:
            print(f"[CrossDomainKG] Base graph not found at {self.base_path} — starting empty")
            self._base_data = {"nodes": [], "links": []}

    def _concepts_to_nodes(self, concepts: list[ConceptSpec]) -> list[dict]:
        return [
            {
                "id":             c.id,
                "module":         c.module,
                "is_submittable": c.is_submittable,
                "is_child_table": c.is_child_table,
                "description":    c.description,
                "cross_domain":   True,
            }
            for c in concepts
        ]

    def _concepts_to_links(self, concepts: list[ConceptSpec]) -> list[dict]:
        links = []
        existing_ids = {c.id for domain in ALL_DOMAINS for c in domain}
        for c in concepts:
            for target in c.links:
                links.append({
                    "source":       c.id,
                    "target":       target,
                    "relationship": "conceptual_link",
                    "cross_domain": True,
                })
        return links

    def _cross_domain_bridges(self, concepts: list[ConceptSpec], base_nodes: list[dict]) -> list[dict]:
        """
        Generate analogy edges between cross-domain concepts and ERP concepts
        based on structural similarity (is_submittable, is_child_table, degree class).
        These bridges are what enable the analogy engine to find cross-domain patterns.
        """
        bridges = []
        cross_ids = {c.id for domain in ALL_DOMAINS for c in domain}

        submittable_cross    = {c.id for domain in ALL_DOMAINS for c in domain if c.is_submittable}
        child_cross          = {c.id for domain in ALL_DOMAINS for c in domain if c.is_child_table}
        submittable_erp      = {n["id"] for n in base_nodes if n.get("is_submittable")}
        child_erp            = {n["id"] for n in base_nodes if n.get("is_child_table")}

        # Pair submittable ERP concepts with structurally similar cross-domain concepts
        STRUCTURAL_BRIDGES = [
            ("Transaction",        "Journal Entry",         "both_submittable_atomic"),
            ("StateMachine",       "Workflow",              "both_model_state_transitions"),
            ("API",                "DocType",               "both_define_contracts"),
            ("Cache",              "Stock Ledger Entry",    "both_record_ephemeral_state"),
            ("Queue",              "Email Queue",           "both_async_work_queues"),
            ("Index",              "Item Price",            "both_accelerate_lookup"),
            ("LoadBalancer",       "Assignment Rule",       "both_distribute_work"),
            ("CircuitBreaker",     "Dunning",               "both_prevent_cascade_failure"),
            ("Metabolism",         "General Ledger Entry",  "both_transform_resources"),
            ("Homeostasis",        "Budget",                "both_maintain_balance_target"),
            ("FeedbackLoop",       "Reorder Level",         "both_respond_to_threshold"),
            ("Adaptation",         "Lead Score",            "both_update_from_outcomes"),
            ("Equilibrium",        "Reconciliation",        "both_achieve_balance_state"),
            ("PhaseTransition",    "Fiscal Year",           "both_mark_sharp_boundary"),
            ("Market",             "Pricelist",             "both_coordinate_exchange"),
            ("Incentive",          "Salary Component",      "both_shape_agent_behaviour"),
            ("PrincipalAgent",     "Expense Claim",         "both_delegate_on_behalf"),
            ("NetworkEffect",      "Referral",              "both_grow_from_participation"),
            ("OptimizationAgenda", "Delivery Route",        "both_minimize_cost_constraints"),
            ("NaturalSelection",   "Supplier Scorecard",    "both_select_best_fit"),
        ]

        base_node_ids = {n["id"] for n in base_nodes}
        for cross_id, erp_id, relationship in STRUCTURAL_BRIDGES:
            if cross_id in cross_ids and erp_id in base_node_ids:
                bridges.append({
                    "source":       cross_id,
                    "target":       erp_id,
                    "relationship": relationship,
                    "cross_domain": True,
                    "bridge":       True,
                })
                bridges.append({
                    "source":       erp_id,
                    "target":       cross_id,
                    "relationship": relationship,
                    "cross_domain": True,
                    "bridge":       True,
                })

        return bridges

    def expand(self, domains: list[list[ConceptSpec]] | None = None) -> dict[str, Any]:
        """
        Build the expanded graph dict.
        domains: which domain concept lists to add (default: all four).
        """
        domains = domains or ALL_DOMAINS
        all_concepts = [c for domain in domains for c in domain]

        existing_ids = {n["id"] for n in self._base_data.get("nodes", [])}

        new_nodes = [n for n in self._concepts_to_nodes(all_concepts) if n["id"] not in existing_ids]
        new_links = self._concepts_to_links(all_concepts)
        bridges   = self._cross_domain_bridges(all_concepts, self._base_data.get("nodes", []))

        expanded = {
            "nodes": self._base_data.get("nodes", []) + new_nodes,
            "links": self._base_data.get("links", self._base_data.get("edges", [])) + new_links + bridges,
        }

        return expanded

    def save(self, output_path: str = "data/graph/knowledge_graph_expanded.json") -> dict:
        expanded = self.expand()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(expanded, indent=2))

        base_nodes  = len(self._base_data.get("nodes", []))
        base_edges  = len(self._base_data.get("links", self._base_data.get("edges", [])))
        total_nodes = len(expanded["nodes"])
        total_edges = len(expanded["links"])

        print(f"[CrossDomainKG] Base: {base_nodes} nodes, {base_edges} edges")
        print(f"[CrossDomainKG] Added: {total_nodes - base_nodes} nodes, {total_edges - base_edges} edges")
        print(f"[CrossDomainKG] Total: {total_nodes} nodes, {total_edges} edges")
        print(f"[CrossDomainKG] Saved → {output_path}")

        return {
            "base_nodes":  base_nodes,
            "base_edges":  base_edges,
            "added_nodes": total_nodes - base_nodes,
            "added_edges": total_edges - base_edges,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "output_path": output_path,
            "domains":     [domain[0].module for domain in ALL_DOMAINS],
        }

    def stats(self) -> dict:
        total_concepts = sum(len(d) for d in ALL_DOMAINS)
        return {
            "domains":              [domain[0].module for domain in ALL_DOMAINS],
            "concepts_per_domain":  {domain[0].module: len(domain) for domain in ALL_DOMAINS},
            "total_new_concepts":   total_concepts,
            "structural_bridges":   20,
            "base_graph_exists":    self.base_path.exists(),
        }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-Domain KG Expansion")
    parser.add_argument("--base",   default="data/graph/knowledge_graph.json")
    parser.add_argument("--output", default="data/graph/knowledge_graph_expanded.json")
    parser.add_argument("--stats",  action="store_true")
    args = parser.parse_args()

    expander = CrossDomainKGExpander(args.base)

    if args.stats:
        print(json.dumps(expander.stats(), indent=2))
        return

    result = expander.save(args.output)
    print(json.dumps(result, indent=2))

    # Quick demo: show sample analogies that the engine can now find
    print("\nSample structural bridges added:")
    bridges = [
        ("Transaction",        "Journal Entry",      "both submittable, atomic"),
        ("StateMachine",       "Workflow",            "both model state transitions"),
        ("Homeostasis",        "Budget",              "both maintain a balance target"),
        ("Market",             "Pricelist",           "both coordinate exchange via price"),
        ("NaturalSelection",   "Supplier Scorecard",  "both select the best-fit entity"),
    ]
    for cross, erp, reason in bridges:
        print(f"  {cross:25} ↔ {erp:22} [{reason}]")


if __name__ == "__main__":
    main()

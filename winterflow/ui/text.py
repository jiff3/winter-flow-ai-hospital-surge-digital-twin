from __future__ import annotations


METHODOLOGY_TEXT = """
WINTER-Flow is a synthetic command-center demonstration. The hospital network, virus waves,
patient arrivals, resource constraints, and outcomes are generated from deterministic seeded
models so scenarios can be reproduced exactly.

The workflow is:

1. Generate fictional Irish-style hospitals with capacity, workforce, and discharge-delay profiles.
2. Generate winter respiratory virus multipliers using Gaussian seasonal waves.
3. Sample patient arrivals with day-of-week effects, virus-specific severity, acuity, age group,
   admission probability, ICU probability, service time, length of stay, and discharge delay.
4. Run a SimPy discrete-event flow model from ED arrival through triage, ED care, admission,
   boarding, inpatient stay, discharge delay, and exit.
5. Aggregate hourly resource state into daily occupancy, trolley, waiting-time, staff-stress,
   and operational-risk metrics.

This is not a clinical model and does not use real patient data by default.
"""

POLICY_SANDBOX_TEXT = """
Use the sandbox to compare a no-intervention baseline against a policy package. Demand-side
controls shift virus timing and intensity. Resource-side controls add surge capacity, alter
staff availability, reduce elective pressure, accelerate discharge flow, or increase transfer
capacity.
"""

REPORT_PLACEHOLDER_TEXT = """
The report module will package the selected scenario, synthetic demand summary, simulation KPIs,
policy comparison, and disclaimers into a downloadable executive brief in a later phase.
"""


def format_scenario_name(scenario: str) -> str:
    return scenario.replace("_", " ").title().replace("Rsv", "RSV").replace("Covid", "COVID")


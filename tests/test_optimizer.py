from winterflow.data.hospitals import get_default_hospitals
from winterflow.data.scenarios import ScenarioConfig, generate_winter_scenario
from winterflow.data.synthetic_history import generate_daily_hospital_demand
from winterflow.optimization.actions import generate_feasible_actions
from winterflow.optimization.optimizer import optimize_resource_allocation


def _optimizer_inputs():
    hospitals = get_default_hospitals(seed=21).head(3)
    scenario = generate_winter_scenario(ScenarioConfig(scenario="combined_winter_surge", n_days=21, seed=21))
    demand = generate_daily_hospital_demand(hospitals, scenario, seed=21)
    return hospitals, demand


def test_do_nothing_policy_exists() -> None:
    hospitals, _ = _optimizer_inputs()

    actions = generate_feasible_actions(hospitals, budget=50)

    assert any(action.action_type == "do_nothing" for action in actions)


def test_generated_actions_are_feasible() -> None:
    hospitals, _ = _optimizer_inputs()

    actions = generate_feasible_actions(hospitals, budget=30)

    assert all(action.cost_points <= 30 for action in actions)
    assert all(action.amount >= 0 for action in actions)


def test_optimizer_returns_at_least_one_recommendation() -> None:
    hospitals, demand = _optimizer_inputs()

    result = optimize_resource_allocation(demand, hospitals, budget=60)

    assert not result["recommendations"].empty


def test_recommendations_are_sorted_by_objective_score() -> None:
    hospitals, demand = _optimizer_inputs()

    result = optimize_resource_allocation(demand, hospitals, budget=80)
    scores = result["recommendations"]["objective_score"].tolist()

    assert scores == sorted(scores)


def test_selected_plan_respects_budget() -> None:
    hospitals, demand = _optimizer_inputs()

    result = optimize_resource_allocation(demand, hospitals, budget=40)

    assert result["estimated_cost_points"] <= 40


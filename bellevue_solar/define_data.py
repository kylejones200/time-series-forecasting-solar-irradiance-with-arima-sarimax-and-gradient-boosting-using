"""Auto-split from legacy monolithic script."""

import pyomo.environ as pyo


def define_data() -> None:
    crudes = ["A", "B", "C"]
    cost = {"A": 70, "B": 80, "C": 65}
    api = {"A": 34, "B": 40, "C": 30}
    sulfur = {"A": 1.2, "B": 0.5, "C": 2.0}
    avail = {"A": 5000, "B": 3000, "C": 4000}
    target_volume = 6000
    api_min = 35
    sulfur_max = 1.0
    model = pyo.ConcreteModel()
    model.crudes = pyo.Set(initialize=crudes)
    model.vol = pyo.Var(model.crudes, domain=pyo.NonNegativeReals)
    model.cost = pyo.Objective(
        expr=sum((model.vol[c] * cost[c] for c in model.crudes)), sense=pyo.minimize
    )
    model.total_volume = pyo.Constraint(
        expr=sum((model.vol[c] for c in model.crudes)) == target_volume
    )
    model.sulfur_limit = pyo.Constraint(
        expr=sum((model.vol[c] * sulfur[c] for c in model.crudes))
        <= sulfur_max * target_volume
    )
    model.api_limit = pyo.Constraint(
        expr=sum((model.vol[c] * api[c] for c in model.crudes))
        >= api_min * target_volume
    )
    model.avail_limits = pyo.ConstraintList()
    for c in model.crudes:
        model.avail_limits.add(model.vol[c] <= avail[c])

    solver = pyo.SolverFactory("glpk")
    result = solver.solve(model)
    if (
        result.solver.status == pyo.SolverStatus.ok
        and result.solver.termination_condition == pyo.TerminationCondition.optimal
    ):
        print("Optimal blend:")
        for c in crudes:
            print(f"  Crude {c}: {model.vol[c]():.1f} bbl")
        total_cost = sum((model.vol[c]() * cost[c] for c in crudes))
        blend_api = sum((model.vol[c]() * api[c] for c in crudes)) / target_volume
        blend_sulfur = sum((model.vol[c]() * sulfur[c] for c in crudes)) / target_volume
        print(f"Total cost: ${total_cost:,.2f}")
        print(f"Blended API: {blend_api:.2f}")
        print(f"Blended sulfur: {blend_sulfur:.2f}%")
    else:
        print("No optimal solution found.")

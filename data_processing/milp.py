from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger


lambda_penalty = 0.4
distance_penalty = 0.2


def milp_optimization(supply, demand, distance):
    model = LpProblem("MIP_Optimization", LpMaximize)
    x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, cat=LpInteger) for i in supply for j in demand}

    model += lpSum(x[i, j] for i in supply for j in demand) \
             - lambda_penalty * lpSum((supply[i] - lpSum(x[i, j] for j in demand)) for i in supply) \
             - distance_penalty * lpSum(distance[i, j] * x[i, j] for i in supply for j in demand)

    for i in supply:
        model += lpSum(x[i, j] for j in demand) <= supply[i]
    for j in demand:
        model += lpSum(x[i, j] for i in supply) <= demand[j]

    model.solve()
    output = {}
    for key in x:
        val = x[key].varValue
        #if val != 0.0:
        output[key] = val
    return output

if __name__ == "__main__":
    import random
    supply = {i: random.randint(500, 1000) for i in range(5)}
    demand = {i: random.randint(50, 350) for i in range(20)}
    distance = {(i, j): random.randint(1, 20) for i in range(5) for j in range(20)}

    opt = milp_optimization(supply, demand, distance)

    unmet_supply = [supply[i] - sum(opt[(i, j)] for j in demand) for i in supply]
    unmet_demand = [demand[i] - sum(opt[(j, i)] for j in supply) for i in demand]
    print(f"Total supply: {sum(supply.values())}\tUnmet supply: {sum(unmet_supply)}")
    print(f"Total demand: {sum(demand.values())}\tUnmet demand: {sum(unmet_demand)}")



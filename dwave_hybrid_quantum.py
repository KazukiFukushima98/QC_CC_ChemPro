import symengine as sp
import re
from amplify import VariableGenerator, Model, solve, LeapHybridSamplerClient
import time


def simplify_binary_expression(expr):
    """
    Simplify only powers of binary variables in SymEngine expressions
    Preserve exponents after parentheses
    """
    expr_str = str(expr)
    pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\*\*[1-9][0-9]*'
    simplified_str = re.sub(pattern, r'\1', expr_str)
    result = sp.sympify(simplified_str)
    return result

for nlen in [5, 10, 15, 20, 25, 30]:
    list_var = sp.symbols("y_0:2")
    for i in ["x1", "x2", "s1", "s2", "s3"]:
        list_var += sp.symbols(i + "_0:{}".format(nlen))

    y_list = [list_var[0], list_var[1]]
    x1_list = list_var[2: 2 + nlen]
    x2_list = list_var[2 + nlen: 2 + 2 * nlen]
    s1_list = list_var[2 + 2 * nlen: 2 + 3 * nlen]
    s2_list = list_var[2 + 3 * nlen: 2 + 4 * nlen]
    s3_list = list_var[2 + 4 * nlen: 2 + 5 * nlen]

    x_min = 0
    x_max = 5
    X1 = x_min
    count = 0
    for i in x1_list:
        X1 += (x_max - x_min) / (2 ** nlen - 1) * 2 ** count * i
        count += 1
    X2 = x_min
    count = 0
    for i in x2_list:
        X2 += (x_max - x_min) / (2 ** nlen - 1) * 2 ** count * i
        count += 1

    s_min = 0
    s_max = 5

    s_list = []
    for i in [s1_list, s2_list]:
        S = s_min
        count = 0
        for j in i:
            S += (s_max - s_min) / (2 ** nlen - 1) * 2 ** count * j
            count += 1
        s_list.append(S)

    s_min = 0
    s_max = 5
    S = s_min
    count = 0
    for j in s3_list:
        S += (s_max - s_min) / (2 ** nlen - 1) * 2 ** count * j
        count += 1
    s_list.append(S)

    c1 = 1.5
    c2 = 1
    c3 = 6.1
    c4 = 5.9
    alpha1 = 0.75
    alpha2 = 0.80

    E = c1 * X1 ** 2 + c2 * X2 ** 2 + c3 * y_list[0] + c4 * y_list[1]
    const = [(X1 - x_max + s_list[0]) ** 2, (X2 - x_max + s_list[1]) ** 2,
             (3 - (y_list[0] * alpha1 * X1 + y_list[1] * alpha2 * X2) + s_list[2]) ** 2]
    E_sum = E
    for i in const:
        E_sum += 10000 * i

    gen = VariableGenerator()
    q = gen.array("Binary", len(list_var))

    sig = {}
    for i, j in zip(list_var, q):
        sig[str(i)] = j

    Sor = simplify_binary_expression(sp.expand(E_sum))

    Sb = eval(str(Sor), sig)

    st1 = time.time()
    model = Model(Sb)
    ed1 = time.time()

    shots = 5

    client = LeapHybridSamplerClient()
    client.token = "***"

    st = time.time()

    result_list = []
    result_list_2 = []
    for _ in range(shots):
        result = solve(model, client)
        if len(result) > 0:
            best_solution = result.best
            energy = best_solution.objective
            values = q.evaluate(best_solution.values)
            result_list.append((values, energy, result.total_time.seconds + result.total_time.microseconds * 10**(-6), result.client_result.info["run_time"]* 10**(-6),
                                (result.total_time.seconds + result.total_time.microseconds * 10**(-6) - result.client_result.info["run_time"]* 10**(-6)), result.client_result.info["qpu_access_time"] * 10**(-6), ed1-st1))

    ed = time.time()

    result_list.sort(key=lambda x: x[1])

    save_list = []
    save_list.append(result_list[0][1])

    count = 0
    for res in result_list:
        if res[1] < 19.9625 * 1.3:
            count += 1

    c = {}
    best_values = result_list[0][0]

    for idx, var_name in enumerate([str(v) for v in list_var]):
        var_symbol = sp.Symbol(var_name)
        c[var_symbol] = int(best_values[idx])

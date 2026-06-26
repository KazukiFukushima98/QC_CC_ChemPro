import symengine as sp
import re
import numpy as np

def simplify_binary_expression(expr):
    """
    Simplify only powers of binary variables in SymEngine expressions
    Preserve exponents after parentheses
    """
    # Convert SymEngine expression to string
    expr_str = str(expr)

    # Pattern matching only variable exponents (exclude exponents after parentheses)
    pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\*\*[1-9][0-9]*'

    # Replace matched parts with variable name only (remove exponent)
    simplified_str = re.sub(pattern, r'\1', expr_str)

    # Convert string back to SymEngine expression
    result = sp.sympify(simplified_str)
    return result

modes = ["GPU", "CPU"]
estmate_list = list(range(5, 201, 5))

for nlen in range(4, 31):
    list_var = sp.symbols("y_0:2")
    for i in ["x1", "x2", "s1", "s2", "s3"]:
        list_var += sp.symbols(i + "_0:{}".format(nlen))
    y_list = [list_var[0], list_var[1]]
    x1_list = list_var[2: 2+nlen]
    x2_list = list_var[2+nlen: 2+2*nlen]
    s1_list = list_var[2+2*nlen: 2+3*nlen]
    s2_list = list_var[2+3*nlen: 2+4*nlen]
    s3_list = list_var[2+4*nlen: 2+5*nlen]

    x_min = 0
    x_max = 5
    X1 = x_min
    count = 0
    for i in x1_list:
        X1 += (x_max - x_min)/(2**nlen - 1) * 2**count * i
        count += 1
    X2 = x_min
    count = 0
    for i in x2_list:
        X2 += (x_max - x_min)/(2**nlen - 1) * 2**count * i
        count += 1

    s_min = 0
    s_max = 5

    s_list = []
    for i in [s1_list, s2_list]:
        S = s_min
        count = 0
        for j in i:
            S += (s_max - s_min)/(2**nlen - 1) * 2**count * j
            count += 1
        s_list.append(S)

    s_min = 0
    s_max = 5
    S = s_min
    count = 0
    for j in s3_list:
        S += (s_max - s_min)/(2**nlen - 1) * 2**count * j
        count += 1
    s_list.append(S)

    c1 = 1.5
    c2 = 1
    c3 = 6.1
    c4 = 5.9
    alpha1 = 0.75
    alpha2 = 0.80

    E = c1*X1**2 + c2*X2**2 + c3 * y_list[0] + c4 * y_list[1]
    const = [(X1 - x_max + s_list[0])**2, (X2 - x_max + s_list[1])**2, (3 - (y_list[0] * alpha1*X1 + y_list[1]*alpha2*X2) + s_list[2])**2]
    E_sum = E
    for i in const:
        E_sum += 10000 * i

    ig = sp.symbols('ig_0:{}'.format(len(list_var)))
    sig = {}

    from pyqubo import Array, Constraint, Placeholder, Binary
    from hobotan import *
    import time
    q = symbols_list(len(list_var), "q_{}")
    for i, j in zip(list_var, q):
        sig[str(i)] = j
    Sor = simplify_binary_expression(sp.expand(E_sum))
    Sb = eval(str(Sor), sig)
    del sig["__builtins__"]

    for mode in modes:
        for shots in estmate_list:
            st1 = time.time()
            if mode == "GPU":
                hobo, offset = Compile(Sb).get_hobo()
            else:
                device = "cpu"
                hobo, offset = Compile(Sb, mode=mode, device=device).get_hobo()
            ed1 = time.time()
            solver = sampler.MIKASAmpler(mode=mode)

            save_list = []

            T_num = 50000
            st = time.time()
            result = solver.run(hobo, shots=shots, T_num=T_num)
            ed = time.time()
            save_list.append(result[0][1] + offset)
            count = 0
            for res in result:
                res[1] += offset
                if res[1] < 19.9625 * 1.3:
                    count += res[2]

            c = {key: result[0][0][str(value)] for key, value in sig.items()}

            import gc
            gc.collect()
            time.sleep(5)

"""
Practical Experiment 6 & 7: Fuzzy Membership Functions and Fuzzy Set Properties
Syllabus Unit: Unit III - Fuzzy Logic & Its Applications (LO3)
Description: Implements standard Fuzzy Membership Functions (Triangular, Trapezoidal, Gaussian)
             and verifies fundamental Fuzzy Set operations and properties (Union, Intersection,
             Complement, De Morgan's Laws, Involution).
"""

import numpy as np


# --- Experiment 6: Membership Functions ---
def triangular_membership(x, a, b, c):
    """Triangular Membership Function: trimf(x; a, b, c)."""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))


def trapezoidal_membership(x, a, b, c, d):
    """Trapezoidal Membership Function: trapmf(x; a, b, c, d)."""
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-9), 1), (d - x) / (d - c + 1e-9)))


def gaussian_membership(x, mean, sigma):
    """Gaussian Membership Function: gaussmf(x; mean, sigma)."""
    return np.exp(-0.5 * ((x - mean) / (sigma + 1e-9)) ** 2)


# --- Experiment 7: Fuzzy Set Operations and Properties ---
def fuzzy_union(set_a, set_b):
    """Fuzzy Union (S-Norm): mu_AUB(x) = max(mu_A(x), mu_B(x))."""
    return np.maximum(set_a, set_b)


def fuzzy_intersection(set_a, set_b):
    """Fuzzy Intersection (T-Norm): mu_AnB(x) = min(mu_A(x), mu_B(x))."""
    return np.minimum(set_a, set_b)


def fuzzy_complement(set_a):
    """Fuzzy Complement: mu_A'(x) = 1 - mu_A(x)."""
    return 1.0 - set_a


def verify_properties():
    """Verify core fuzzy algebraic properties: Involution and De Morgan's Laws."""
    x = np.linspace(0, 10, 100)
    A = triangular_membership(x, 2, 5, 8)
    B = trapezoidal_membership(x, 4, 6, 8, 10)

    # 1. Involution Law: (A')' == A
    involution_check = np.allclose(fuzzy_complement(fuzzy_complement(A)), A)

    # 2. De Morgan's Law 1: (A U B)' == A' n B'
    demorgan1_lhs = fuzzy_complement(fuzzy_union(A, B))
    demorgan1_rhs = fuzzy_intersection(fuzzy_complement(A), fuzzy_complement(B))
    demorgan1_check = np.allclose(demorgan1_lhs, demorgan1_rhs)

    # 3. De Morgan's Law 2: (A n B)' == A' U B'
    demorgan2_lhs = fuzzy_complement(fuzzy_intersection(A, B))
    demorgan2_rhs = fuzzy_union(fuzzy_complement(A), fuzzy_complement(B))
    demorgan2_check = np.allclose(demorgan2_lhs, demorgan2_rhs)

    print("\n--- FUZZY SET PROPERTIES VERIFICATION (EXP 7) ---")
    print(f"1. Involution Law ((A')' = A): {'VERIFIED [PASS]' if involution_check else 'FAIL'}")
    print(f"2. De Morgan's Law 1 ((A U B)' = A' n B'): {'VERIFIED [PASS]' if demorgan1_check else 'FAIL'}")
    print(f"3. De Morgan's Law 2 ((A n B)' = A' U B'): {'VERIFIED [PASS]' if demorgan2_check else 'FAIL'}")


def main():
    print("=" * 70)
    print("EXPERIMENTS 6 & 7: FUZZY MEMBERSHIPS & SET PROPERTIES (UNIT III)")
    print("=" * 70)

    # Demonstration of Exp 6
    x_test = 5.0
    tri_val = triangular_membership(x_test, 2, 5, 8)
    trap_val = trapezoidal_membership(x_test, 3, 4, 6, 7)
    gauss_val = gaussian_membership(x_test, 5, 1.5)

    print("\n--- MEMBERSHIP FUNCTIONS DEMONSTRATION (EXP 6) ---")
    print(f"Evaluation at Crisp Input x = {x_test}:")
    print(f"  - Triangular Membership (a=2, b=5, c=8):     mu(5) = {tri_val:.4f}")
    print(f"  - Trapezoidal Membership (a=3, b=4, c=6, d=7): mu(5) = {trap_val:.4f}")
    print(f"  - Gaussian Membership (mean=5, sigma=1.5):    mu(5) = {gauss_val:.4f}")

    # Demonstration of Exp 7
    verify_properties()

    print("\n" + "=" * 70)
    print("Fuzzy Logic Membership & Properties Experiments Completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()


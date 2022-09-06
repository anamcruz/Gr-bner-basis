# pylint: disable=consider-using-f-string
# pylint: disable = unrecognized-option
"""Polynomials generator."""

import numpy as np
import sympy as sym


def random_monomial_generator(degree, variables):
    """Generates random monomials.

    This function creates random monomials with predetermined number of
    variables and monomial degree.

    Args:
        degree: monomial degree(sum of exponents is equal to 'deg').
        variables: list of variables that appear in each monomial.

    Returns:
        monomial: returns a monomial.

    """
    l_variables = len(variables)
    powers_for_monomials = np.random.multinomial(
        degree, [1.0 / l_variables] * l_variables)  # random sum of integers
    # that sum up to deg

    monomial = 1
    for variable, power in zip(variables, powers_for_monomials):
        monomial = monomial * variable**power  # transform the variables
        # list and the powers_for_monomials into a monomial format

    return monomial


def random_sympy_polynomial_generator(variables, max_monomial_degree,
                                      min_monomial_degree, n_monomials,
                                      max_coefficient, domain):
    """Generates random polynomial.

    This function creates random polynomials by randomly choosing a degree
    from an interval, defined by a maximum and a minimum degree. The polynomial
    is build using random monomials that random_monomial_generator function
    returns with the random degree uniformly chosen.

    Args:
        variables: variables that appear in each monomial.
        max_monomial_degree: maximum sum of degrees of all monomial elements.
        min_monomial_degree: minimum sum of degrees of all monomial elements.
        n_monomials: number of monomials in the polynomial.
        max_coefficient: maximum limit on the random value sampled as
        coefficient of each monomial (minimum value is set to 1).
        domain: the field we want the monomial to be defined.

    Returns:
        polynomial: a polynomial following the given parameters.
    """

    monomials_list = [
        np.random.randint(1, max_coefficient) * random_monomial_generator(
            np.random.randint(min_monomial_degree, max_monomial_degree),
            variables) for _ in range(n_monomials)
    ]  # creates n_monomials with a random degree, uniformly chosen from an
    # interval
    polynomial = sum(monomials_list)  # uses the monomials to create a
    # polynomial
    return sym.poly(polynomial, domain=domain)

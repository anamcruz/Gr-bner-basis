# pylint: disable=consider-using-f-string
# pylint: disable = unrecognized-option
"""Demo for polynomials creation"""

from absl import logging
from absl import app

from grobner import polynomial_generator
from grobner import polynomial_utils

# creates variables list from x0 to xn_99
gens = polynomial_utils.get_variable_list(7)


def main(_):
    logging.info(
        'The following function, random_monomial_generator, prints a monomial'
        'with the powers sum is less than 45 and more than 5, and the number '
        'of variables used is 7')
    logging.info(
        polynomial_generator.random_sympy_polynomial_generator(
            variables=gens,
            max_monomial_degree=45,
            min_monomial_degree=5,
            n_monomials=4,
            max_coefficient=1068,  # 1068 was chosen because is a magical number
            domain='ZZ'))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)

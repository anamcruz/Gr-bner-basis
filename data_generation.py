# pylint: disable = consider-using-f-string
# pylint: disable = unrecognized-option
"""Data Generation"""

from absl import logging, app, flags
import itertools
import os
import sympy as sym
import time

import csv

from grobner import polynomial_generator
from grobner import polynomial_utils

FLAGS = flags.FLAGS

flags.DEFINE_list("max_monomial_degree", [24, 22, 20, 18, 16, 14],
                  "Maximum sum of degrees of all monomial elements.")

flags.DEFINE_integer("min_monomial_degree", 1,
                     "Minimum sum of degrees of all monomial elements.")

flags.DEFINE_list(
    "n_variables", [6, 5, 4, 3],
    "Defines the number the variables to appear in each monomial.")

flags.DEFINE_integer("n_monomials", 2, "Number of monomials in the polynomial.")

flags.DEFINE_integer(
    "max_coefficient", 1068,
    "Maximum limit on the random value sampled as coefficient of each "
    "monomial (minimum value is set to 1).")

flags.DEFINE_string("domain", "ZZ",
                    "The field we want the monomial to be defined.")

flags.DEFINE_list("n_polynomials", [10, 4],
                  "Number of polynomials in each polynomial set")

flags.DEFINE_integer("n_set", 100, "Number of polynomial sets to "
                     "generate.")


def create_data(max_monomial_degree, min_monomial_degree, gens, n_monomials,
                max_coefficient, domain, n_polynomials, n_set):
    """Generates polynomials sets.
    This function generates polynomials sets that generates ideals with
    the desired arguments for the polynomial ideal.

    Args:
        max_monomial_degree: maximum sum of degrees of the monomial elements.
        min_monomial_degree: minimum sum of degrees of the monomial elements.
        gens: generators of the polynomials.
        n_monomials: number of monomials in the polynomial.
        max_coefficient: maximum limit on the random value sampled as
        coefficient of each monomial (minimum value is set to 1).
        domain:  the field we want the monomial to be defined.
        n_polynomials: number of polynomials in each set.
        n_set: numbers of polynomials set to be generated.

    Returns:
        data: returns list of polynomials set.
    """
    data = []
    for _ in range(n_set):
        f = [(polynomial_generator.random_sympy_polynomial_generator(
            gens, max_monomial_degree, min_monomial_degree, n_monomials,
            max_coefficient, domain).as_expr()) for _ in range(n_polynomials)]
        data.append(f)
    return data


def sympy_groebner_basis(ideals_list, gens):
    """Computes Groebner Basis.

     This function uses the ideals generated previously and computes their
     reduced Groebner Basis and the corresponding execution time.

     Args:
         ideals_list: list of ideals.
         gens: generators of the ideals.

    Returns:
        groebner_basis: returns the reduced Groebner basis for a set of
        polynomials.
        time_loop: returns execution time.

    E.g. The function receives the following arguments:
            ideals_list: [[x0**3*x1*x2**5 + 8*x1**2*x2**2, 5*x0**5*x2 +
         3*x0**4*x1**6*x2**6, 6*x0**3*x1**11*x2**5 + 3*x0**2*x2**2,
         9*x0**7*x1**10*x2**2 + 6*x0**5*x1**7*x2**6]].
            gens: [x0, x1, x2].
         The output of will be a list of the resulting Groebner Basis and the
         last row corresponds to the computing time of each Groebner Basis:
            groebner_basis: GroebnerBasis([x0**5*x2, x0**2*x2**2, x1**2*x2**2],
            x0, x1, x2, domain='ZZ', order='grevlex'), [0.029143810272216797].

     """

    groebner_basis = []
    time_loop = []
    for _ in range(FLAGS.n_set):
        start_time = time.time()
        g = sym.groebner(ideals_list[_], gens, order="grevlex")
        groebner_basis.append(g)
        end_time = time.time()
        t = end_time - start_time
        time_loop.append(t)
    groebner_basis.append(time_loop)
    return groebner_basis


def main(_):

    output_polynomials_file = os.path.join(
        os.getcwd(), "dataset-{FLAGS.n_variables}-{"
        "FLAGS.max_monomial_degree}-{FLAGS.n_polynomials}.csv")

    logging.info("Saving data in file %s", output_polynomials_file)
    with open(output_polynomials_file, "w+", encoding="utf-8") as test:
        writer = csv.writer(test)

        for max_monomial_degree, n_variables, n_polynomials in \
                itertools.product(FLAGS.max_monomial_degree, FLAGS.n_variables,
                     FLAGS.n_polynomials):
            # creates a variables list with the number of variables desired
            gens = polynomial_utils.get_variable_list(n_variables)
            f = create_data(max_monomial_degree=max_monomial_degree,
                            min_monomial_degree=FLAGS.min_monomial_degree,
                            gens=gens,
                            n_monomials=FLAGS.n_monomials,
                            max_coefficient=FLAGS.max_coefficient,
                            domain=FLAGS.domain,
                            n_polynomials=n_polynomials,
                            n_set=FLAGS.n_set)
            writer.writerows(
                [[n_variables, max_monomial_degree, n_polynomials] +
                 set_polynomial for set_polynomial in f])


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)

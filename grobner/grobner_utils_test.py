# pylint: disable=consider-using-f-string
# pylint: disable = unrecognized-option
"""Unit tests for Buchberger algorithm and all the utilities"""

import sympy as sym

from grobner import grobner_utils as gb
from grobner import polynomial_utils as pl


def test_reduce_polynomial():
    """Tests for reduce_polynomial.
    reduce_polynomial is a function which computes a reduced generating set
    for a given polynomial set. The test example is from
    http://www.scholarpedia.org/article/Groebner_basis in the chapter
    'Definition via the Church-Rosser Property', calculated by
    hand and checked using Sympy function
    sympy.polys.polytools.reduced(f, G, *gens, **args)
    """
    x, y, z, _ = pl.get_standard_generators()
    gen = x, y, z
    list_of_polynomials = [
        sym.poly(x * z**2 - 3 * y * z + 1, gens=gen, domain='ZZ'),
        sym.poly(x**2 - 2 * y, gens=gen, domain='ZZ'),
        sym.poly(x * y - 5 * z, gens=gen, domain='ZZ')
    ]
    polynomial = sym.poly(3 * x**3 * z + -x, gens=gen, domain='ZZ')
    result_expected = sym.poly(-1 / 30 * x + z**2, gens=gen, domain='QQ')

    assert gb.reduce_polynomial(polynomial,
                                list_of_polynomials,
                                order='grevlex',
                                gens=gen) == result_expected


def test_reduce_generating_set():
    """Tests for reduce_generating_set()
     It computes a reduced generating set for a given polynomial set. Test
     example is taken from the book 'Ideals, Varieties, and Algorithms',
     Cox, D.A., Little, J., O'Shea, D., chapter 2 'Gröbner Bases': &7
     'Buchbergerger's Algorithm', page 94.
     """
    x, y, z, w = pl.get_standard_generators()
    gen = x, y, z, w
    list_of_polynomials = [
        sym.poly(3 * x - 6 * y - 2 * z, gens=gen, domain='ZZ'),
        sym.poly(2 * x - 4 * y + 4 * w, gens=gen, domain='ZZ'),
        sym.poly(x - 2 * y - z - w, gens=gen, domain='ZZ')
    ]
    result_expected = [
        sym.poly(x - 2 * y + 2 * w, gens=gen, domain='QQ'),
        sym.poly(z + 3 * w, gens=gen, domain='ZZ')
    ]
    reduced_gen_set = gb.reduce_generating_set(list_of_polynomials,
                                               order='grevlex',
                                               gens=gen)

    assert {polynomial.as_expr() for polynomial in reduced_gen_set} \
           == {polynomial.as_expr() for polynomial in result_expected}


def test_s_polynomial():
    """Tests for s_polynomial()
    It computes the syzygy polynomial of two polynomials. The test example
    is from http://www.scholarpedia.org/article/Groebner_basis in the chapter
    'The Syzygy Property'
    """
    x, y, z, _ = pl.get_standard_generators()
    gen = x, y, z
    polynomial_1 = sym.poly(x * y + 2 * x - z, gens=gen, domain='ZZ')
    polynomial_2 = sym.poly(x**2 + 2 * y - z, gens=gen, domain='ZZ')
    result_expected = sym.poly(2 * x**2 - x * z - 2 * y**2 + y * z,
                               gens=gen,
                               domain='ZZ')

    assert gb.s_polynomial(
        polynomial_1, polynomial_2, order='grevlex', gens=gen)\
        == result_expected


def test_buchberger_alg():
    """ This function tests buchberger_alg(), which
    finds the Grobner basis of a polynomial set.
    The test example is taken from the book
    'Ideals, Varieties, and Algorithms', Cox, D.A., Little, J., O'Shea, D.,
    chapter 2 'Gröbner Bases': &7 'Buchbergerger's Algorithm', page 93.
    """
    x, y, _, _ = pl.get_standard_generators()
    gen = x, y
    list_of_polynomials = [
        sym.poly(x**3 - 2 * x * y, gens=gen, domain='ZZ'),
        sym.poly(x**2 * y - 2 * y**2 + x, gens=gen, domain='ZZ')
    ]
    result_expected = [
        sym.poly(-(1 / 2 * x) + y**2, gens=gen, domain='QQ'),
        sym.poly(x**2, gens=gen, domain='ZZ'),
        sym.poly(x * y, gens=gen, domain='ZZ')
    ]

    grobner_basis = gb.buchberger_alg(list_of_polynomials,
                                      order='grevlex',
                                      gens=gen)

    assert {polynomial.as_expr() for polynomial in grobner_basis} \
           == {polynomial.as_expr() for polynomial in result_expected}

# pylint: disable=consider-using-f-string
# pylint: disable = unrecognized-option
"""Buchberger algorithm and all the utilities needed for it"""

import itertools
import random
import sympy as sym

from grobner import polynomial_utils as pl
from grobner import observers

_SILENT_OBSERVER = observers.SilentObserver()


def reduce_polynomial(polynomial,
                      polynomial_set,
                      order='grevlex',
                      gens=pl.get_standard_generators()):
    """Computes the reduction of a polynomial by a polynomial set.
    The reduction is also called multivariate division or normal form
    computation. For a pair of polynomials (f, g) if there is a monomial term m
    in f that is divisible by the leading term of the polynomial g, then the
    first step reduction of f by g is given by
        red_1(f,g) = f - (m/lt(g))*g
    where lt(g) is the leading term of g. This operation eliminates the term m
    from the polynomial f.
    The reduction of f by a given polynomial set is obtain by computing all
    possible first step reductions of f by the elements of the polynomial set.
    Args:
        polynomial: Sympy polynomial of type sympy.poly.
        polynomial_set: List of sympy polynomials.
    Returns:
        reduced_polynomial: As the name says, it is the reduced polynomial.
    """

    reduced_polynomial = polynomial

    is_reducible = True
    while is_reducible:
        is_reducible = False
        for element in polynomial_set:
            # take the terms that are divisible by the leading term of an
            # element in the polynomial set
            div_terms = pl.divisible_terms(
                reduced_polynomial, sym.LT(element, order=order, gens=gens))
            # if there is one
            if not pl.is_poly_zero(div_terms):
                # the reduced_polynomial can be further reduced
                is_reducible = True
                # so one subtracts these divisible terms from it
                reduced_polynomial = \
                    pl.subtract_divisible_terms(
                        reduced_polynomial, element,
                        order=order, gens=gens)

    # we return the reduced polynomial having 1 as leading coefficient
    reduced_polynomial = (1 / sym.LC(reduced_polynomial, gens=gens,
                                     order=order)) * reduced_polynomial

    return reduced_polynomial


def reduce_generating_set(polynomial_set,
                          order='grevlex',
                          gens=pl.get_standard_generators()):
    """Computes a reduced generating set for a given polynomial set.
    A Grobner basis is reduced if the leading term of each of its generators
    has coefficient equal to 1 and no other element of the basis contains a
    non-zero multiple of this leading term.
    E.g. [x**2+2*y, y+3] fails to be a Grobner basis as the leading term of
    y+3 is y, and y divides the term 2*y in x**2+2*y
    """

    #  TODO(@EdMDias): Correct the for loop to not be indexed by the list.

    is_non_reduced = True

    while is_non_reduced:
        is_non_reduced = False

        # order the polynomial set in ascending in ascending order for the
        # monomial order
        polynomial_set = pl.order_polynomials(polynomial_set, order=order)
        # set the leading coefficients to be all equal to 1
        polynomial_set = [
            (1 / sym.LC(polynomial, gens=gens, order=order)) * polynomial
            for polynomial in polynomial_set
        ]

        # if there is a single polynomial then there is nothing to do
        if len(polynomial_set) == 1:
            break

        # for every polynomial in the polynomial_set, starting with the one
        # with biggest leading term, we check if it can be reduced by the
        # remaining polynomial set

        for polynomial in polynomial_set[::-1]:
            # the order used to reduce the polynomials is a choice
            polynomial_set.remove(polynomial)
            reduction = reduce_polynomial(polynomial,
                                          polynomial_set,
                                          order=order,
                                          gens=gens)

            # if the polynomial does not change when reduced against the
            # rest of the polynomial set, we put it back in the set
            if pl.is_poly_zero(reduction - polynomial):
                polynomial_set.append(reduction)

            # if its reduction is the zero polynomial, we discard it
            elif pl.is_poly_zero(reduction):
                continue

            # if the reduction is a non-zero different polynomial, we had it
            # back to the polynomial set and will check all the polynomials
            # again
            else:
                polynomial_set.append(reduction)
                is_non_reduced = True

    return polynomial_set


def s_polynomial(poly_f,
                 poly_g,
                 order='grevlex',
                 gens=pl.get_standard_generators()):
    """Computes the syzygy polynomial of two polynomials.
    Given f, g, two polynomials, the syzygy polynomial, also called the
    subtraction polynomial of f and g for a given monomial order, is the
    polynomial obtain by the subtraction of the 'smallest' multiples of f and g
    that have the same leading terms.
    E.g. for the lexicographic order, consider
        f = x**2 + y
    and
        g = x*y**2.
    Then
        s_polynomail(f,g) = y**2*f - x*g = y**3.
    Notice that neither the leading term of f nor the one of g divides the
    leading term of s_polynomial(f,g)."""

    if poly_g == 0:
        return poly_f
    else:
        leading_term_f = sym.poly(sym.LT(poly_f, order=order, gens=gens),
                                  gens=gens)
        leading_term_g = sym.poly(sym.LT(poly_g, order=order, gens=gens),
                                  gens=gens)

        expression = sym.simplify(
            sym.lcm(leading_term_f, leading_term_g).as_expr() *
            (poly_f * leading_term_g - poly_g * leading_term_f).as_expr() *
            (1 / (leading_term_f * leading_term_g).as_expr()))

        return sym.Poly(expression, gens)


def buchberger_alg(polynomial_set,
                   order='grevlex',
                   gens=pl.get_standard_generators(),
                   observer=_SILENT_OBSERVER):
    """Finds the Grobner basis of a polynomial set.
    Given a polynomial set F, let I be the ideal generated by F, I = <F> and
    lt(I) the ideal of the leading terms of the elements of I. Then a
    polynomial set G is called a Grobner basis for F if I = <G> and moreover,
    lt(I) = <lt(G)>, i.e. the leading terms of the polynomials in G generate
    the ideal of the leading terms of I.
    E.g. consider
        F = [x**2 + y, x*y**2].
    Then
        y**2*(x**2 + y) - x*(x*y**2) = y**3
    is an element of I. So lt(I) has the terms
        x**2, x*y**2, y**3
    in the set of generators, which implies that G contains the set
        [x**2 + y, x*y**2, y**3].
    If we now compute the s_polynomial for the each pair obtain from these 3
    polynomials we find that there is no new leading term, hence we conclude
    that the Grobner basis for F is [x**2 + y, x*y**2, y**3].
    """

    groebner_basis = []
    polynomial_set = reduce_generating_set(polynomial_set,
                                           order=order,
                                           gens=gens)
    pairs_of_polynomials = [(f, 0) for f in polynomial_set]

    while len(pairs_of_polynomials) > 0:
        observer.update(len(pairs_of_polynomials))
        poly_1, poly_2 = random.choice(pairs_of_polynomials)
        # which pair to pick is an arbitrary choice. Here we randomly pick one
        # of these pairs.
        pairs_of_polynomials.remove((poly_1, poly_2))
        s_pol = s_polynomial(poly_1, poly_2, order=order, gens=gens)
        if pl.is_poly_zero(s_pol):
            # if the s_polynomial of between two polynomials is zero then
            # there is no element to be added to the Grobner basis
            continue

        elif pl.is_poly_constant(s_pol, gens=gens):
            # if the s_polynomial is a non-zero constant, then there is a
            # constant in the ideal generated by the polynomial set which
            # implies that the only generator needed is the identity
            groebner_basis = [sym.poly(1, gens=gens)]
            break

        else:
            # in any other case, the s_polynomial is a candidate to be a
            # member of the Grobner basis, and it is so if its reduction by
            # the already computed Grobner basis elements is non-zero
            reduced = reduce_polynomial(s_pol,
                                        groebner_basis,
                                        order=order,
                                        gens=gens)
            if reduced != 0:
                pairs_of_polynomials += list(
                    itertools.product([reduced], groebner_basis))
                groebner_basis += [reduced]

    return reduce_generating_set(groebner_basis, order=order, gens=gens)

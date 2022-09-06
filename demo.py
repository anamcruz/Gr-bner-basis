"""Demo to see Buchberger algorithm for a small list of polynomials."""

# TODO(@EdMDias): re-write the demo as a Jupyter Notebook
# pylint: disable=consider-using-f-string
# pylint: disable = unrecognized-option
from grobner import grobner_utils as gb
import sympy as sym
from absl import logging, app

# Define the variables used in the examples
x, y, z = sym.symbols('x, y, z')

# The lists of polynomials we will test on
list_of_polynomials1 = [sym.poly(x**2 + y), sym.poly(x * y**2)]
list_of_polynomials2 = [sym.poly(x * y - z**2), sym.poly(x**3 - y**2)]
list_of_polynomials3 = [sym.poly(x * z - y**2), sym.poly(x**3 - z**2)]
list_of_polynomials4 = [sym.poly(x**2 + y + 1), sym.poly(y**3 + x + 1)]


def main(_):
    print('In this demo Buchberger\'s algorithm will compute the Grobner '
          'basis for some sets of polynomials.\n')
    print('Case 1:\nConsider \nF1 = [x**2 + y, x*y**2]. \nThen')
    print('Grobner_basis(F1) = ',
          gb.buchberger_alg(list_of_polynomials1, order='grevlex'))
    print('Case 2:\nConsider \nF2 = [x*y - z**2, x**3-y**2]. \nThen')
    print('Grobner_basis(F2) = ',
          gb.buchberger_alg(list_of_polynomials2, order='grevlex'))
    print('An interesting fact about the last example is, if you replace the '
          'order of the variables (or trade them in the polynomials), '
          'you will get a smaller Grobner basis!!!')
    print('Case 2.1:\nConsider \nF2.1 = [x*z - y**2, x**3-z**2]. \nThen')
    print('Grobner_basis(F2.1) = ',
          gb.buchberger_alg(list_of_polynomials3, order='grevlex'))
    print('To finish we present an example on how a Grobner basis can be '
          'used to solve a system of polynomial equations. Consider\n'
          'F3 = [x**2 + y + 1, y**3 + x + 1] \nand consider now that '
          'the order is the lexicographical order. Then')
    print('Grobner_basis(F3) = ',
          gb.buchberger_alg(list_of_polynomials4, order='lex'))
    print('You see that one of the polynomials only contains the variable y '
          'so it can be solved by a numerical method (or tartaglia!) and '
          'for each value of y we can find the corresponding value of x.')


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)

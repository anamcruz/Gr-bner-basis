[![Python package](https://github.com/inductiva/grobner/actions/workflows/python-package.yml/badge.svg)](https://github.com/inductiva/grobner/actions/workflows/python-package.yml)


# Gröbner Basis

Gröbner bases are a fundamental concept in computational algebra. Since the creation of the theory behind them in 1949, by Wolfgang Gröbner, they became an important tool in any area where polynomial computations play a part, both in theory and in practice. Although they have proved to be very useful, their calculation is very expensive in certain cases. The first algorithm ever developed to compute these bases is the so-called Buchberger’s Algorithm and is still one of the most commonly used algorithms for this purpose. As a preliminary step in improving the efficiency of the algorithm, one would like to be able to predict, given an ideal, how complicated it is to compute its Gröbner basis using Buchberger’s Algorithm. In this research, we address precisely this issue, following the work of Mojsilovic, Peifer and Petrovic. We create a dataset consisting of some binomial and toric distributions. Some of their properties were studied in order to seek the relationship between these characteristics and the number of polynomial additions. Then we introduce linear regression and a simple neural network model to try to predict the number of iterations using the ideals properties. Then, it is used a recurrent neural network to study the relationship between the exponents of ideals and that of polynomial additions is studied. The performance of the three models is compared and we show that there is a considerable improvement when using a recurrent neural network model and conclude that we are able to predict the number of polynomial additions, in some cases.

## Bibliography
  1. Mojsilović, J., Peifer, D. & Petrović, S. 2021. [_Learning a performance metric of
Buchberger’s algorithm_](https://arxiv.org/abs/2106.03676).


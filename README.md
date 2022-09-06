[![Python package](https://github.com/inductiva/grobner/actions/workflows/python-package.yml/badge.svg)](https://github.com/inductiva/grobner/actions/workflows/python-package.yml)


# Gröbner Basis

## Introduction

1. Describe what Gröbner Basis (GB) are and why they are important. For most
readers, especially in the ML community, we need to start with the most applied
use of GB: helping in the process of solving sytems of polynomial equations
(apparently finding GB is "equivalent" to performing Gauss-elimination in linear
algebra). Enumerate some the the most important cases where solving systens of
polynomials is crucial. But we can also give examples of specific recent (10-15
years) works where the computation of GBs is a fundamental part of the process
and/or has been decisive in achieving an important result (e.g. scaling) in an
important field or application area (e.g.: cryptography). The key point here is
to show that GB are omnipresent and that they are important in many active
fields of research.


2. Some more theory and concepts, but only if really needed. (E.g. I am not sure
if we need to define Ideal. Can we frame the whole thing as "just" solving systems
of polynomials). And then present the classic  algorithm: the Buchberger's algo.
Explain briefly how it works. We do not have to go too deep on this because the
goal of the work is not to make improvements on the algo - for now - but to
introduce to a new dataset. Present just the basics and then point people to
surveys on the matter. We just need to explain what are the parameters that have
influence in the result (e.g. order, etc). What we need to do really well is
explain what are the implications of playing with those parameters: does the set
of poly's grow too much? Does the algorithm take too much time to run? What
"observable" metrics can we have on the performance of the algorithm to track the
impact of playing with these parameters.

Now that the problems with the Buchberger's algo are clear, we can maybe mention
some other alternatives to the Buchberger algo, but again, we do not have to go
too deep. What we need to really emphasize is that these are algorithms have also
been tuned "manually", or required lots of work to get right. We can show that
may be very good for some cases by showing a few examples. But we would also show
that, however, these algos are unable to work well for all cases. To perform this
overview, besides the standard references, there is a survey that seems particularly
interesting: https://eprint.iacr.org/2021/870.pdf. Also, there is a comparison
between F4 and F5 here: https://eprint.iacr.org/2021/051.pdf.


3. Now that is clear that all this is a lot of manual and brain work, and we
still have only *sub-optimal* algos, we can say that we hope that ML can come to
the rescue. Describe in high-level recent work on trying to use ML-based decision
in substituting the decisions on the S-Pairs(?), the right permutation of vars (?).
Please expand mostly on what previous works was trying to achieve, i.e. the goals.
Briefly summarize their results and limitations (briefly, because we will have a
dedicated section later in the paper).


4. Describe what we are proposing: following on previous work by(Mojsilović et al)
we want to build a much larger dataset for experimentation than they built, because
they covered only a couple of cases (torus and something else), and we want to cover
a much more comprehensive set of families systems of polynomial equation, such as 
X and Y which are important in certain problem domains (e.g. in cryptograpy?
in optimization of  something? in solving PDE's?).Our contribution will be a massive
dataset, that hopefully will contribute to bringing a bunch of ML researchers to
the field. This is, however, not a simple task, not only because the computation
time for generating this dataset is long, but also because it is not trivial to
decide on the type of poly's we will be including. This is why our work is relevant.
This way we just make it easy for a legion of ML researchers, who otherwise not
even consider this question, to attack this problem and maybe solve it, with
great impact for many problems and math in general.


## Bibliography
  1. Mojsilović, J., Peifer, D. & Petrović, S. 2021. [_Learning a performance metric of
Buchberger’s algorithm_](https://arxiv.org/abs/2106.03676).


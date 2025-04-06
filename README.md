# BICause Trees

The repo containing the code and experiments for the model described in [_Hierarchical Bias-Driven Stratification for Interpretable Causal Effect Estimation_](https://openreview.net/forum?id=WdcA7v7uzc); accepted to AISTATS 2025.

The code in this reporsitory cotains the version used to obtain the results from the paper.

## `Causallib` version
However, [version 0.10.0 of causallib](https://pypi.org/project/causallib/0.10.0) now contains a cleaner, better-tested, and improved version of the proposed model (see https://github.com/BiomedSciAI/causallib/pull/76). 

We therefore encourage people using the version existing from causallib if they intend to use the model for their own research.

This can be done by the following:
1. Install causallib:
   ```bash
   pip install causallib
   ```
2. Import BICauseTree:
   ```python
   from causallib.contrib.bicause_tree import BICauseTree
   ```
3. At this point `BICauseTree` behaves like any other `IndividualOutcomeEstimator`   in   causallib.  
   This means that, given covaraites  `X`, treatment assignment `a`, and outcome `y`, one can
   ```python
   bic = BICauseTree()  # See documentation for parametrization
   bic.fit(X, a, y)  # Fit the causal model based on the tree
   avg_outcomes = bic.estimate_population_outcome(X, a, y)
   ind_outcomes = bic.estimate_individual_outcome(X, a, y)
   # Specifying `y` may be optional depending on what type of `outcome_model` was passed to BICauseTree.
   ```


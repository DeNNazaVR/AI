# Summer Internship Project: Local ML Explanations with Pattern Structures

## 1. Project in one paragraph

Modern machine-learning models can make good predictions but can be difficult to explain.  
The goal of this project is to build a **local explanation method** for a prediction of a black-box classifier.

The project is inspired by **LIME**. LIME explains one prediction by generating artificial examples near the object we want to explain, asking the black-box model to predict them, and then fitting a simple model in this local neighborhood.

Here we will try a different idea: use **Formal Concept Analysis (FCA)** and, in particular, **Pattern Structures** as the local surrogate. Instead of only producing a list of feature weights, we want to obtain structured descriptions of nearby objects and organize them from **more specific** to **more general**.

You are **not expected to know FCA, Pattern Structures, LIME, or advanced machine learning before starting**. We will introduce the necessary ideas step by step.

---

## 2. Main question

The main research question is:

> Can a local Pattern-Structure-based surrogate explain the behavior of a black-box classifier around one selected object?

A useful explanation should help answer questions such as:

- Which conditions are sufficient for the black-box model to keep the same prediction?
- Which conditions can be removed while keeping the prediction?
- What happens when we move to a more general description?
- What happens when we move to a more specific description?
- How well does the local explanation agree with the black-box model?

The project is experimental. A negative result is also useful if we understand **why** the method does not work well in some situations.

---

## 3. What you will learn

During the project you will work with:

- Python for data analysis and machine learning;
- `numpy`, `pandas`, and `scikit-learn`;
- a black-box classifier such as Random Forest;
- LIME as a baseline explanation method;
- local sampling / perturbation of data;
- basic Formal Concept Analysis;
- Pattern Structures;
- interval descriptions for numerical data;
- evaluation of explanation quality;
- visualization of a local order / lattice fragment;
- Git and reproducible experiments.

The main goal is not to learn every mathematical detail. The goal is to understand enough theory to implement the method correctly and analyze its behavior.

---

# Part I. Background

## 4. What is a black-box model?

Suppose we have a trained classifier

\[
f(x) \in \{0,1\}.
\]

For example, `f` may be a Random Forest.

The model receives an object

\[
x = (x_1, x_2, \ldots, x_m)
\]

and predicts class `0` or class `1`.

Even if the model is accurate, it may be difficult to understand why it made one particular prediction.

This project is about explaining **one prediction at a time**.

---

## 5. Local explanation

Suppose we want to explain the prediction for one test object `x`.

Instead of trying to understand the entire classifier, we study the behavior of the classifier **near `x`**.

We generate nearby artificial objects:

\[
N(x) = \{z_1, z_2, \ldots, z_n\}.
\]

Then we ask the black-box model to predict each one:

\[
y_i = f(z_i).
\]

These are called **pseudo-labels**.

Important:

> We are trying to reproduce and explain the behavior of the black-box model, so the local explanation is built using `f(z)`, not the true dataset labels.

---

## 6. How LIME works

Very roughly, LIME does the following:

1. Select an object `x`.
2. Generate perturbed samples near `x`.
3. Ask the black-box model for predictions on these samples.
4. Give larger weights to samples that are closer to `x`.
5. Fit a simple local model, usually a sparse linear model.
6. Use this simple model to explain the prediction.

For example, LIME may produce something similar to:

```text
feature_2 > 4.3              +0.31
feature_5 <= 1.7             +0.22
feature_1 between 2.1 and 3  -0.08
```

This is useful, but it does not naturally show a hierarchy of explanations.

Our idea is to replace the linear local surrogate with a surrogate based on Pattern Structures.

---

# Part II. Pattern Structures

## 7. The basic idea

Formal Concept Analysis studies objects, their descriptions, and the structure created by common descriptions.

For numerical data, a convenient Pattern Structure uses **intervals**.

Suppose an object has two numerical features:

```text
x = (3, 8)
```

We can represent it as:

```text
([3, 3], [8, 8])
```

Another object

```text
z = (5, 6)
```

is represented as:

```text
([5, 5], [6, 6])
```

The common description of `x` and `z` is the smallest interval in every feature that contains both objects:

```text
([3, 5], [6, 8])
```

This is the interval Pattern Structure operation.

In general:

\[
[a_1,b_1] \sqcap [a_2,b_2]
=
[\min(a_1,a_2), \max(b_1,b_2)].
\]

For several features, apply the operation independently to each feature.

---

## 8. General and specific descriptions

Consider:

```text
A = ([3, 5], [6, 8])
B = ([3.5, 4.0], [7, 7.5])
```

`A` covers a larger region of the feature space.

Therefore:

- `A` is **more general**;
- `B` is **more specific**.

For interval patterns:

- wider intervals -> more general description;
- narrower intervals -> more specific description.

This order is important because it lets us move between different possible explanations.

A very specific explanation may be almost identical to the original object and therefore not useful.

A very general explanation may cover many objects from both classes and therefore also not be useful.

We want a useful compromise.

---

## 9. Extent of an interval pattern

Suppose a pattern is

```text
feature_1 in [2, 5]
feature_2 in [6, 9]
```

Its **extent** is the set of local samples whose feature values satisfy all these intervals.

For a pattern `d`, we can therefore compute:

```python
extent(d) = all local objects covered by d
```

This is one of the most important operations in the project.

---

## 10. Closure

Given a pattern `d`:

1. Find all local objects covered by `d`.
2. Compute the smallest interval pattern that contains all these objects.

This produces the **closure** of the pattern.

If closing the pattern does not change it, it is a closed description and corresponds to a Pattern Concept.

For the first implementation we do **not** need to enumerate every possible concept in the complete lattice.  
The complete lattice may be too large.

Instead, we will construct a **small local fragment** around the object being explained.

---

# Part III. The actual project

## 11. Overall pipeline

For every object `x` that we want to explain:

```text
Dataset
   |
   v
Train black-box model f
   |
   v
Choose test object x
   |
   v
Generate local neighborhood N(x)
   |
   v
Get black-box predictions f(z)
   |
   +--------------------+
   |                    |
   v                    v
LIME baseline     Pattern Structure
                        |
                        v
                Candidate patterns
                        |
                        v
                Closed descriptions
                        |
                        v
              General/specific order
                        |
                        v
                 Local explanation
                        |
                        v
              Compare and evaluate
```

---

# Part IV. Step-by-step work plan

## 12. Step 0 — Set up the project

Before working on the research part:

- install Python;
- install Git;
- clone this repository;
- create a virtual environment;
- install the required packages;
- make sure a simple notebook runs.

Recommended core packages:

```text
numpy
pandas
scikit-learn
matplotlib
lime
networkx
jupyter
```

We may add other packages later if needed.

Use a fixed random seed in experiments whenever possible.

Example:

```python
RANDOM_STATE = 42
```

---

## 13. Step 1 — Start with a very simple dataset

Do **not** begin with a complicated real dataset.

First use a simple 2-dimensional synthetic dataset such as:

```python
sklearn.datasets.make_moons
```

Why?

Because in two dimensions we can directly visualize:

- the black-box decision boundary;
- the object `x`;
- the generated neighborhood;
- which objects have prediction `0` or `1`;
- interval explanations around `x`.

This makes debugging much easier.

After the method works on a simple synthetic dataset, we will move to a real binary classification dataset.

We will choose the real dataset together.

---

## 14. Step 2 — Train a black-box classifier

Start with a Random Forest.

Example:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

Use a train/test split.

Check:

- training accuracy;
- test accuracy;
- confusion matrix.

Do not spend too much time optimizing the classifier.  
The classifier only needs to be good enough to provide a meaningful black-box model to explain.

---

## 15. Step 3 — Choose objects to explain

Do not explain only one object.

Eventually select several examples, for example:

1. an object predicted confidently as class `0`;
2. an object predicted confidently as class `1`;
3. an object close to the black-box decision boundary;
4. optionally, an incorrectly classified object.

The first implementation can use only one object.

---

## 16. Step 4 — Implement the LIME baseline

Before implementing our method, run standard LIME on the same object.

Save:

- the predicted class;
- class probabilities;
- LIME feature contributions;
- a visualization or text representation of the explanation.

This gives us something to compare with.

You do not need to implement LIME yourself.

Use the existing library.

---

## 17. Step 5 — Generate the local neighborhood

For an object `x`, generate nearby samples.

For numerical data, a first simple strategy is Gaussian perturbation:

\[
z_j = x_j + \epsilon_j.
\]

The noise size should depend on the scale of feature `j`.

A reasonable first implementation is:

```python
noise_j ~ Normal(0, sigma * feature_std_j)
```

where `sigma` is a parameter.

Try, for example:

```text
sigma = 0.05
sigma = 0.10
sigma = 0.20
```

The generated samples should remain plausible.

For every generated point `z`, store:

```text
z
distance(x, z)
black_box_prediction = f(z)
black_box_probability
```

A useful starting neighborhood size is:

```text
500-2000 samples
```

Start smaller while debugging.

---

## 18. Step 6 — Add proximity weights

Samples closer to `x` should usually matter more.

A simple kernel is:

\[
w_x(z)
=
\exp\left(-\frac{d(x,z)^2}{\sigma^2}\right).
\]

Before computing distances, numerical features should normally be standardized.

Keep both:

- standardized features for distances;
- original feature values for human-readable explanations.

---

## 19. Step 7 — Implement interval patterns

Create a simple Python representation of a numerical interval pattern.

For one feature:

```python
(lower, upper)
```

For `m` features:

```python
[
    (lower_1, upper_1),
    (lower_2, upper_2),
    ...
    (lower_m, upper_m)
]
```

Implement at least:

```python
point_to_pattern(x)
meet(pattern_a, pattern_b)
covers(pattern, x)
extent(pattern, local_samples)
closure(pattern, local_samples)
is_more_general(pattern_a, pattern_b)
```

Write unit tests for these functions.

For example:

```python
x = [3, 8]
z = [5, 6]

meet(
    point_to_pattern(x),
    point_to_pattern(z)
)
```

must produce:

```text
([3, 5], [6, 8])
```

---

## 20. Step 8 — Generate candidate explanations

We do not need the full concept lattice at first.

Start from the object `x`.

Its initial description is:

```text
[x_1, x_1]
[x_2, x_2]
...
[x_m, x_m]
```

This description is too specific because it normally covers only `x`.

Now gradually generalize it.

### Simple baseline strategy

1. Find the nearest local samples to `x`.
2. Combine `x` with one nearby sample using the Pattern Structure operation.
3. Close the resulting pattern.
4. Measure its quality.
5. Repeat with other nearby samples.
6. Combine promising patterns with additional samples.
7. Remove duplicates.

This creates descriptions with different levels of generality.

Example:

```text
P0 = exact description of x

P1 = hull(x, z1)

P2 = hull(x, z1, z2)

P3 = hull(x, z1, z2, z3)
```

As we include more objects, the intervals usually become wider and the description becomes more general.

Later we can implement a better search strategy.

---

## 21. Step 9 — Score every candidate pattern

Suppose the black-box prediction for `x` is class `c`.

For a candidate pattern `P`, compute its extent:

```text
A = extent(P)
```

### Support

Number of local objects covered by the pattern:

\[
support(P) = |A|.
\]

### Coverage

Fraction of the neighborhood covered:

\[
coverage(P)
=
\frac{|A|}{|N(x)|}.
\]

### Purity

Fraction of covered objects for which the black box predicts the same class as for `x`:

\[
purity(P)
=
\frac{
|\{z \in A : f(z)=f(x)\}|
}{
|A|
}.
\]

A weighted version can use proximity weights.

### Complexity

A simple explanation should not use unnecessarily wide or complicated intervals.

Possible complexity measures:

- number of active features;
- normalized total interval width;
- number of conditions shown to the user.

We will decide on a final score after the basic implementation works.

Do **not** invent a complicated score at the beginning.

First inspect support, coverage, purity, and interval widths separately.

---

## 22. Step 10 — Select useful explanations

A good explanation should:

- contain the original object `x`;
- have high purity;
- cover more than only one or two samples;
- not be unnecessarily specific;
- be understandable.

For example, a candidate could be accepted if:

```text
purity >= 0.90
support >= 20
```

These are only initial experimental thresholds, not universal rules.

The important research problem is the trade-off:

```text
more specific
    -> usually higher purity
    -> usually lower coverage

more general
    -> usually higher coverage
    -> may reduce purity
```

This trade-off is exactly what the generalization order can help us study.

---

## 23. Step 11 — Build a local generalization graph

After generating candidate closed descriptions, organize them by the generalization relation.

Create a graph where:

- each node is one candidate Pattern Concept;
- an edge connects nearby more-general / more-specific concepts;
- node information includes:
  - support;
  - purity;
  - predicted class;
  - interval description.

Use `networkx` for the first implementation.

Do not try to draw hundreds of nodes.

Filter the graph using criteria such as:

```text
minimum support
minimum purity
top-k candidates
maximum number of displayed nodes
```

The goal is a small graph that a human can inspect.

---

## 24. Step 12 — Compare with LIME

For the **same black-box model** and the **same object `x`**, compare:

### LIME

Produces approximately:

```text
feature importance / local linear conditions
```

### Pattern-Structure explanation

Produces approximately:

```text
a local interval description
+
support
+
purity
+
coverage
+
more general descriptions
+
more specific descriptions
```

Questions to study:

- Does the Pattern Structure explanation agree with LIME about important features?
- Does it show interactions between features that are difficult to see in LIME?
- Can we remove a condition and keep high purity?
- How quickly does purity decrease when the explanation becomes more general?
- Is the result stable when the neighborhood is generated again?

---

# Part V. Evaluation

## 25. Minimum experiments

When the basic method works, run it for at least:

```text
3-5 different test objects
```

Include:

- at least one easy/high-confidence example;
- at least one example near a decision boundary.

For every explained object, save:

- black-box prediction;
- black-box probability;
- LIME explanation;
- selected Pattern Structure explanation;
- support;
- coverage;
- purity;
- generalization graph / local fragment.

---

## 26. Stability

Generate the local neighborhood several times using different random seeds.

For example:

```text
42
123
2026
```

Check whether:

- similar interval conditions appear;
- purity remains similar;
- coverage remains similar;
- the selected explanation changes dramatically.

If an explanation changes a lot, that is an important result.

Do not hide unstable or failed examples.

---

## 27. Important experimental rules

### Rule 1 — Explain the black box

The local pseudo-label is:

```python
model.predict(z)
```

not the true dataset label.

### Rule 2 — Use the same object for comparisons

Do not compare LIME on one object with Pattern Structures on another.

### Rule 3 — Keep train and test data separate

Fit preprocessing and the model using training data only.

### Rule 4 — Fix random seeds

Otherwise experiments are difficult to reproduce.

### Rule 5 — Keep generated objects plausible

Do not generate impossible values only because the code allows it.

### Rule 6 — Start simple

A correct implementation on a small dataset is more useful than an unfinished complicated implementation.

---

# Part VI. Suggested repository structure

A possible structure is:

```text
lime_fca/
│
├── notebooks/
│   ├── 01_blackbox_and_lime.ipynb
│   ├── 02_local_neighborhood.ipynb
│   ├── 03_interval_patterns.ipynb
│   └── 04_experiments.ipynb
│
├── src/
│   ├── data.py
│   ├── blackbox.py
│   ├── neighborhood.py
│   ├── interval_pattern.py
│   ├── candidate_search.py
│   ├── explanation.py
│   └── evaluation.py
│
├── tests/
│   └── test_interval_pattern.py
│
├── outputs/
│   ├── figures/
│   ├── explanations/
│   └── lattices/
│
├── notes/
│   └── progress.md
│
├── requirements.txt
├── README.md
└── INTERNSHIP_TASK.md
```

The exact structure can change during the project.

---

# Part VII. Work plan for approximately four weeks

## Week 1 — Understand the problem and reproduce the baseline

Goals:

- set up the repository;
- learn the basic idea of local explanations;
- train a Random Forest;
- generate predictions;
- run LIME;
- understand the basic ideas of FCA and Pattern Structures;
- implement and test interval operations.

Expected result:

> We can train a black-box model, explain one prediction with LIME, and correctly manipulate interval patterns.

---

## Week 2 — Build the local Pattern Structure explanation

Goals:

- generate local neighborhoods;
- compute pseudo-labels and distances;
- implement pattern extents and closure;
- generate candidate descriptions around `x`;
- compute support, coverage, and purity;
- inspect the first generalization paths.

Expected result:

> For one object, we can generate several local interval explanations and order them from more specific to more general.

---

## Week 3 — Evaluation and visualization

Goals:

- build a small local generalization graph;
- select useful explanations;
- compare with LIME;
- run the method on several objects;
- analyze easy and difficult examples.

Expected result:

> We have a working end-to-end prototype and several understandable examples.

---

## Week 4 — Improve, test, and document

Goals:

- test stability with different random seeds;
- improve neighborhood generation if necessary;
- clean the code;
- save reproducible experiments;
- prepare a short report;
- prepare a short final presentation/demo.

Expected result:

> A clean research prototype with experiments, limitations, and possible next steps.

---

# Part VIII. Minimum final result

By the end of the internship, the minimum useful result is:

- [ ] a reproducible Python environment;
- [ ] a trained black-box classifier;
- [ ] a working LIME baseline;
- [ ] local neighborhood generation;
- [ ] an implementation of interval Pattern Structure operations;
- [ ] candidate local Pattern Concepts or a restricted local concept fragment;
- [ ] support, coverage, and purity calculations;
- [ ] explanations for several test objects;
- [ ] a visualization of at least one local generalization structure;
- [ ] comparison with LIME;
- [ ] a short written report;
- [ ] clean code in the repository.

A complete new scientific method is **not** required.

The priority is:

> correct implementation -> understandable experiments -> careful analysis -> possible research extensions.

---

# Part IX. Optional extensions

Only work on these after the basic pipeline is complete.

## A. Better neighborhood generation

Compare:

- Gaussian perturbation;
- nearest-neighbor sampling;
- interpolation between `x` and nearby training objects;
- sampling from local empirical distributions.

---

## B. Weighted Pattern Concepts

Use the LIME-style proximity weights in:

- weighted purity;
- weighted support;
- candidate ranking.

---

## C. Counterfactual explanations

Find nearby descriptions associated with the opposite black-box prediction.

Question:

> What small change could move the object to a region where the black-box prediction changes?

---

## D. Alterfactual explanations

Try to generalize the explanation while keeping the same prediction.

Question:

> Which conditions can be relaxed or removed without changing the prediction?

This is especially natural in the Pattern Structure order.

---

## E. Compare several black-box models

For example:

- Random Forest;
- Gradient Boosting;
- nonlinear SVM;
- small neural network.

---

## F. Other data types

Pattern Structures are not limited to numerical intervals.

In principle, descriptions can be created for:

- sets;
- sequences;
- text;
- graphs;
- other structured objects.

This is an important motivation for the project, but it is **not part of the required first implementation**.

We will first make the numerical/tabular version work correctly.

---

# Part X. How to work on the project

## 28. Git

Commit regularly.

Good commit messages:

```text
Implement Gaussian local sampler
Add interval pattern meet operation
Add purity and coverage metrics
Fix closure computation
Add first LIME comparison
```

Avoid one huge final commit.

---

## 29. Keep short progress notes

Maintain:

```text
notes/progress.md
```

After each substantial work session, write three short sections:

```text
## Done
What I completed.

## Problems
What is unclear or not working.

## Next
What I plan to do next.
```

This is useful both for you and for discussions with the mentor.

---

## 30. When you are stuck

Research code often does not work immediately.

When something fails:

1. reduce the problem to a small example;
2. check intermediate values;
3. write a small test;
4. write down what you expected;
5. write down what actually happened;
6. try to identify the smallest failing component.

If you remain blocked after serious attempts, ask.

When asking, include:

```text
What I am trying to do
What I expected
What happened
What I already tried
Relevant code / error message
```

Do not spend an entire day silently stuck on one implementation detail.

---

## 31. Code quality

Prefer clear code over clever code.

Use:

- meaningful variable names;
- small functions;
- docstrings for important functions;
- type hints when useful;
- comments for non-obvious logic;
- tests for Pattern Structure operations.

Code, comments, function names, and the main technical documentation should preferably be in **English**.

---

# Part XI. What is not expected

You are **not** expected to:

- understand the full theory of FCA immediately;
- read an entire FCA textbook;
- prove new mathematical theorems;
- enumerate a huge complete lattice;
- build a production system;
- obtain a publishable scientific result in one month;
- test every possible type of data;
- optimize the black-box model for maximum benchmark accuracy.

The project should grow step by step.

---

# Part XII. First concrete tasks

Start with these tasks in this exact order.

### Task 1

Clone the repository and create the Python environment.

### Task 2

Create a notebook:

```text
notebooks/01_blackbox_and_lime.ipynb
```

Generate a `make_moons` dataset and visualize it.

### Task 3

Train a Random Forest and visualize its predictions on the 2D plane.

### Task 4

Choose one test object and run LIME for it.

### Task 5

Create:

```text
src/interval_pattern.py
```

Implement:

```python
point_to_pattern
meet
covers
extent
closure
is_more_general
```

### Task 6

Write tests for these operations using very small hand-made examples.

### Task 7

We will review the implementation together before moving to the candidate-search and explanation stages.

---

# References

You do not need to read all of these immediately.

The most relevant topics are:

1. **LIME**  
   Ribeiro, Singh, Guestrin — *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.*

2. **Formal Concept Analysis**  
   Ganter and Wille — *Formal Concept Analysis: Mathematical Foundations.*

3. **Pattern Structures**  
   Ganter and Kuznetsov — *Pattern Structures and Their Projections.*

4. Additional Pattern Structure papers and references are listed in the main repository README.

For the first week, understanding the high-level idea is enough. We will decide together which sections are worth reading in detail.

---

# Final perspective

There are two levels to this project.

### Engineering level

Can we build a working local explanation pipeline using interval Pattern Structures?

### Research level

Does the Pattern Structure representation provide something useful that a standard method such as LIME does not provide — for example, an interpretable generalization/specialization structure?

We start with the engineering question.

Only after the system works correctly do we try to answer the research question.

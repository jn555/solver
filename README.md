This project is for solving linear programming problems (LPs) using CUDA on GPUs. I'm still working on it, so it's incomplete.

The focus is on two methods:

1. Simplex Method: A classic algorithm for LPs. Basically, it moves from one "corner" of the solution space to another until it finds the best one.

2. Interior Point Method (IPM): A newer approach that skips the corners and goes through the "middle" of the solution space. More math-heavy and faster for some cases.

What Are LPs?

LPs are problems where you try to maximize or minimize a function (like profit or cost) under a bunch of linear constraints (like resource limits). Think of it like:

maximize c^T x
subject to A x <= b, x >= 0

Here, c is the profit/cost vector, A is the constraints, and x is what you’re solving for.

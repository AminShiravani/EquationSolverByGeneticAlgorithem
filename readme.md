# 🔬 Genetic Algorithm for Solving Systems of Equations

This project implements a **Genetic Algorithm (GA)** to solve systems of 2, 3, or 4 equations with 2, 3, or 4 unknowns, respectively. It is a practical application of heuristic optimization techniques in numerical equation solving.

---

## 📁 Project Structure

genetic-equation-solver/
├── src/
│ ├── solve_2x2.py ← Solves 2 equations with 2 unknowns
│ ├── solve_3x3.py ← Solves 3 equations with 3 unknowns
│ └── solve_4x4.py ← Solves 4 equations with 4 unknowns
├── docs/
│ ├── Doc.docx ← Documentation in Word format
│ └── Genetic_Algorithm.pdf ← Detailed PDF report
├── requirements.txt ← Required Python packages
├── README.md ← This file

---

## 🧠 Features

- Solves nonlinear systems using a population-based GA
- Handles different number of variables (2 to 4)
- Early stopping if solution is accurate or population stagnates
- Accepts user-defined equations at runtime
- Logs solution quality and best individual per generation

---

## 🔧 Requirements

Install required packages using:

```bash
pip install -r requirements.txt
Required libraries:
    numpy
    sympy
```

🚀 How to Run:
Example for 3 equations:
  python src/solve_3x3.py

Then enter your equations like:
    Equation 1: x + y + z = 6
    Equation 2: 2*x - y + z = 3
    Equation 3: x + 2*y + 3*z = 14

The algorithm will iterate and output the best solution per generation. At convergence or stagnation, it prints:
    Solution summary:
    x = 1.000000
    y = 2.000000
    z = 3.000000
    Final error (L1 norm): 0.0000000123


⚠️ Notes & Limitations
Accuracy depends on mutation rate, population size, and convergence threshold

Cannot guarantee finding a solution if the system has no solution or is highly ill-conditioned

Works best for well-defined numerical systems



```Challenges Faced
Designing a robust mutation strategy to prevent premature convergence

Parsing user input equations safely and converting to symbolic functions

Tuning hyperparameters (mutation probability, elite count, etc.)

Keeping execution time under 10 seconds even for large populations


📚 References
Genetic Algorithms - Wikipedia

SymPy Documentation

Class materials and custom experimental designs during development
---
## Foward-Forward Algorithm

This algorithm is designed by Geoffrey Hinton and presented at NEURIPS 2022 as a possible alternative to the all-popular backpropagation algorithm that powers most of present-day ML.

Note that the difference between this algorithm and regular backprop is not the prescence of gradients or lack thereof; it is the scale and purpose at which and for which the gradients are calculated.

For backprop, the gradients are calculated on a global scale; for the forward-forward algorithm, the gradients are local in scope. This means that each layer calcukated its own gradient independently of the others. It calculates it own gradients, updates its own paremeters, and then uses the new parameters to transform the input data sending the input data to the next layer in the network, where the process is repeated.

---
## Tech Stack

---
## Introduction

---
## Implementation

---
The following steps are to be followed to run this repo:
1. Run the `main.py` script.
---
## Results

---
## To-Dos

---
There are still a few additions to make to the project. They include:
1. Improve documentation.
2. Network does not train.
   1. Possible issue with Layer optimizer.


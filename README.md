# CISC121 – Bubble Sort Visualizer

This project is an interactive web application that **visualizes the Bubble Sort algorithm**.
Users can choose a case type (custom, best‑case, worst‑case, random, or stability demo), enter a
list of integers, and click **Run Algorithm**. The app animates every comparison and swap, and
users can navigate through steps using a **slider** or **Previous/Next** buttons.

A collapsible **Learn Bubble Sort** panel explains the algorithm, its stability, and its
complexity.

---

## Problem Breakdown & Computational Thinking

To build this visualizer, I used the four pillars of computational thinking:

### 🧩 1. Decomposition

Bubble Sort is broken down into:

* Outer loop → each pass through the list
* Inner loop → adjacent comparisons
* Swap operation → when values are out of order
* Early‑exit check → stop when no swaps occur

### 🔍 2. Pattern Recognition

* Each pass moves the largest remaining element to the end
* Duplicate values stay in order (**stable** sorting)
* Worst‑case occurs when the list is *reversed*
* Best‑case occurs when the list is *already sorted*

### ✨ 3. Abstraction

To keep the UI simple:

* Only values are shown as vertical bars
* Compared elements are highlighted
* Indices and code internals are hidden
* Slider abstracts the “timeline” of steps

### 🛠️ 4. Algorithm Design

* **Input:** Comma‑separated integers
* **Process:** Bubble Sort generates all intermediate array states
* **Output:** A visual bar plot + step explanation

The GUI is built using **Gradio**, a beginner‑friendly Python UI library.

---

## Case Options

| Case Type                 | Description                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| **Custom Input**          | Uses exactly the user‑entered list.                                     |
| **Best‑case (sorted)**    | Pre‑sorts the list → algorithm finishes in one pass.                    |
| **Worst‑case (reversed)** | Reverses the list → maximizes comparisons and swaps.                    |
| **Random**                | Generates a random list (same length as input, or length 5 if blank).   |
| **Stability Demo**        | Uses a predefined list with duplicates to show Bubble Sort’s stability. |

---

## Steps to Run Locally

1. Install **Python 3.8+**
2. Download or clone this repository
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

5. A Gradio interface will open at:

```
http://localhost:7860
```

6. Choose a case, enter a list, and click **Run Algorithm**.

---

## Learn Bubble Sort (In‑App Panel)

The learning section includes:

* Explanation of the algorithm
* Pass-by-pass diagrams
* Annotated pseudocode
* Why Bubble Sort is stable
* Complexity chart:

  * Best: **O(n)**
  * Average: **O(n²)**
  * Worst: **O(n²)**
  * Space: **O(1)**

---

## Testing & Verification

To satisfy Step 5 of the project guidelines (“Test and Verify”), the following cases were tested:

| Input           | Case           | Expected Result          | Result     |
| --------------- | -------------- | ------------------------ | ---------- |
| `1,2,3,4,5`     | Best‑case      | No swaps, early exit     | ✔️ Correct |
| `5,4,3,2,1`     | Worst‑case     | Maximum swaps            | ✔️ Correct |
| `3,1,3,2,3`     | Stability Demo | Duplicates stay in order | ✔️ Correct |
| `5,3,8,4,2,7,1` | Custom         | Normal sort              | ✔️ Correct |
| `-3,-1,0,2,1`   | Custom         | Handles negatives        | ✔️ Correct |
| *(empty)*       | Any            | Input error              | ✔️ Correct |
| `2,five,3`      | Any            | Invalid input error      | ✔️ Correct |
| `1`             | Any            | 1 step, no swaps         | ✔️ Correct |
| Random          | Random         | Random list sorts        | ✔️ Correct |

---
## Link to testing screenshots: 
https://docs.google.com/document/d/14nFZsjpls0wgAIwkGB0zxC0NOcyy7EKOpCHPoZZtkvk/edit?usp=sharing

## Hugging Face link:
https://huggingface.co/spaces/Cars0n67/Bubblesortvisualizer

---

## Author & Acknowledgment

This project was created for the **CISC 121 Final Project** to demonstrate computational thinking, algorithm design, and visual‑based learning.

Visual inspiration came from online sorting demos, but all code and documentation are my own.


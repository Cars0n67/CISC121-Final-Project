"""
Bubble Sort Visualizer
======================

This module defines a Gradio application that teaches bubble sort.  It records
every comparison and swap, then displays a bar chart with highlighted indices
and explains each action in plain language.  A slider lets users step through
the algorithm at their own pace.
"""

from __future__ import annotations
import io
from typing import List, Tuple
import matplotlib.pyplot as plt  # type: ignore
from PIL import Image  # type: ignore
import gradio as gr  # type: ignore

def bubble_sort_steps(arr: List[int]) -> List[dict]:
    """
    Run bubble sort while capturing a detailed trace of each operation.
    Each entry in the returned list records the array state, the compared
    indices, whether a swap occurred, the pass number and a descriptive
    message.  The first entry represents the initial unsorted array.
    """
    steps: List[dict] = []
    steps.append({
        "arr": arr.copy(),
        "highlight": None,
        "swapped": False,
        "pass_no": 0,
        "desc": "Initial array: the algorithm will repeatedly compare adjacent values and swap them if out of order."
    })
    n = len(arr)
    for i in range(n):
        swapped_in_pass = False
        for j in range(0, n - i - 1):
            a, b = arr[j], arr[j + 1]
            if a > b:
                arr[j], arr[j + 1] = b, a
                swapped = True
                swapped_in_pass = True
                desc = (
                    f"Pass {i+1}, comparing indices {j} and {j+1}: {a} > {b}, so swap them."
                    f"\n\nArray becomes {arr}."
                )
            else:
                swapped = False
                desc = (
                    f"Pass {i+1}, comparing indices {j} and {j+1}: {a} ≤ {b}, so keep them in place."
                )
            steps.append({
                "arr": arr.copy(),
                "highlight": (j, j + 1),
                "swapped": swapped,
                "pass_no": i + 1,
                "desc": desc
            })
        if not swapped_in_pass:
            steps.append({
                "arr": arr.copy(),
                "highlight": None,
                "swapped": False,
                "pass_no": i + 1,
                "desc": (
                    f"Pass {i+1} completed with no swaps, so the array is sorted and the algorithm stops early."
                )
            })
            break
    return steps

def plot_array(arr: List[int], highlight: Tuple[int, int] | None = None, swapped: bool = False) -> Image.Image:
    """
    Draw a bar chart of the current array state.
    The compared indices are coloured orange (or red if a swap occurred),
    while all other bars are blue.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors: List[str] = []
    for i, _ in enumerate(arr):
        if highlight and i in highlight:
            colors.append("#e74c3c" if swapped else "#f39c12")
        else:
            colors.append("#4fa3d1")
    ax.bar(range(len(arr)), arr, color=colors)
    ax.set_title("Bubble Sort Visualization", fontsize=14)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)

def parse_numbers(text: str) -> List[int]:
    """Convert a comma‑separated string into a list of integers."""
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("Please enter at least one integer.")
    return [int(part) for part in parts]

def run_sort(numbers: str):
    """
    Parse the user's input, run bubble sort, and set up the initial state.
    Returns a slider update, the first image, the first explanation,
    a status message, and the full list of steps.
    """
    try:
        values = parse_numbers(numbers)
    except Exception as exc:
        return (
            gr.update(visible=False),
            plot_array([], None, False),
            "",
            f"Error: {exc}",
            [],
        )
    steps = bubble_sort_steps(values.copy())
    slider_update = gr.update(visible=True, minimum=0, maximum=len(steps) - 1, value=0, step=1)
    first = steps[0]
    img = plot_array(first["arr"], highlight=first["highlight"], swapped=first["swapped"])
    description = first["desc"]
    status = f"Total steps: {len(steps)}" if len(steps) > 1 else "Array is already sorted."
    return slider_update, img, description, status, steps

def update_plot(step: int, steps: List[dict]):
    """
    Update the bar chart, explanation and status when the slider moves.
    Shows a completion message on the final step.
    """
    if not steps:
        return plot_array([], None, False), "", ""
    idx = max(0, min(int(step), len(steps) - 1))
    current = steps[idx]
    img = plot_array(current["arr"], highlight=current["highlight"], swapped=current["swapped"])
    description = current["desc"]
    if len(steps) > 1:
        status = f"Step {idx} of {len(steps) - 1}"
    else:
        status = ""
    if idx == len(steps) - 1:
        pass_no = current.get("pass_no", 0)
        status = f"Completed: array sorted after {pass_no} pass{'es' if pass_no != 1 else ''}."
    return img, description, status

def build_demo() -> gr.Blocks:
    """Assemble and return the Gradio interface."""
    with gr.Blocks(title="Bubble Sort Visualizer") as demo:
        gr.Markdown(
            """
            # Bubble Sort Visualizer

            This app demonstrates how the bubble sort algorithm works.  
            Enter a list of integers separated by commas and click **Run Bubble Sort**.  
            You can then use the slider to move through each comparison and swap.  
            The explanation box describes what happens on each step.
            """
        )
        with gr.Row():
            input_text = gr.Textbox(
                label="Input Numbers",
                placeholder="e.g., 5, 3, 8, 4, 2, 7, 1",
                lines=1,
            )
            run_button = gr.Button("Run Bubble Sort")
        step_slider = gr.Slider(
            minimum=0,
            maximum=0,
            value=0,
            step=1,
            label="Step",
            visible=False,
        )
        chart_output = gr.Image(type="pil", label="Visualization")
        desc_output = gr.Markdown(value="")
        status_label = gr.Textbox(label="Status", interactive=False)
        state = gr.State([])
        run_button.click(
            run_sort,
            inputs=[input_text],
            outputs=[step_slider, chart_output, desc_output, status_label, state],
        )
        step_slider.change(
            update_plot,
            inputs=[step_slider, state],
            outputs=[chart_output, desc_output, status_label],
        )
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()



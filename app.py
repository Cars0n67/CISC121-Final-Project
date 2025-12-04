"""
Bubble Sort Visualizer
======================

This module defines a small Gradio application that demonstrates the bubble
sort algorithm.  Users can enter a comma‑separated list of integers and step
through each iteration of the sorting process.  A bar chart illustrates how
the list changes over time, and the interface exposes a slider that lets
people move backwards and forwards through the recorded states.

The goal of this app is educational: it shows not only the final sorted
output but every intermediate arrangement that arises as adjacent values are
swapped.  The implementation is kept intentionally simple so that
instructors or students can read and modify it without wading through
unnecessary complexity.
"""

from __future__ import annotations

import io
from typing import List, Tuple, Any
import random

import matplotlib.pyplot as plt  # type: ignore
from PIL import Image  # type: ignore
import gradio as gr  # type: ignore

def bubble_sort_steps(arr: List[int]) -> List[dict]:
    """Return detailed step information for the bubble sort algorithm.

    This helper records every comparison and potential swap in a list of
    dictionaries.  Each entry captures the array state, the two indices
    compared, whether a swap occurred, a pass counter and an explanatory
    message.  The first entry of the returned list always represents the
    initial unsorted array.  Subsequent entries correspond to individual
    comparisons during the sorting process.

    Args:
        arr: A mutable list of integers to sort.  The original list is
            mutated during sorting; callers should pass a copy if they
            wish to preserve the original order.

    Returns:
        A list of dictionaries with the following keys:

        ``arr``
            A snapshot of the array after the comparison (and swap, if
            applicable).

        ``highlight``
            A 2‑tuple of indices that were compared.  For the initial
            entry, this is ``None``.

        ``swapped``
            A boolean indicating whether the comparison resulted in a
            swap.

        ``pass_no``
            The current pass number (starting at 1).  Bubble sort makes
            multiple passes over the list, pushing the next‑largest
            element into its final position on each pass.

        ``desc``
            A human‑readable description explaining what happened during
            the step.  It includes the values compared, whether a swap
            occurred, and points out when the algorithm terminates early.
    """
    steps: List[dict] = []
    # Record the initial state with no highlight
    steps.append({
        "arr": arr.copy(),
        "highlight": None,
        "swapped": False,
        "pass_no": 0,
        "desc": "Initial array: the algorithm will repeatedly compare adjacent values and swap them if out of order."
    })
    n = len(arr)
    # Outer loop controls passes; after each pass the largest remaining element
    # bubbles to its final position at the end of the list.
    for i in range(n):
        swapped_in_pass = False
        # On each pass, compare adjacent pairs up to the unsorted portion
        for j in range(0, n - i - 1):
            a, b = arr[j], arr[j + 1]
            if a > b:
                # Perform the swap
                arr[j], arr[j + 1] = b, a
                swapped = True
                swapped_in_pass = True
                desc = (
                    f"Pass {i+1}, comparing indices {j} and {j+1}: {a} > {b}, so swap them."
                    f"\n\nArray becomes {arr}."
                )
            else:
                swapped = False
                # If the values are equal, explain that no swap is needed and that
                # bubble sort preserves the relative order of equal elements (stability).
                if a == b:
                    desc = (
                        f"Pass {i+1}, comparing indices {j} and {j+1}: {a} == {b}, so keep them in place. "
                        "Because the values are equal, bubble sort leaves them in their original order, demonstrating its stability."
                    )
                else:
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
        # If we went through a full pass without swapping anything, the list
        # is already sorted and we can terminate early.  Record one more
        # step to explain this early exit.
        if not swapped_in_pass:
            # Record a termination message: no swaps in this pass
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


# Additional algorithm: Linear Search
def linear_search_steps(arr: List[int], target: int) -> List[dict]:
    """Generate step information for a linear search on an array.

    Each step records the current index being checked and whether the
    target has been found.  A final message is added if the search
    completes without finding the target.

    Args:
        arr: The list of integers to search through.
        target: The integer value to find.

    Returns:
        A list of dictionaries similar to those produced by
        ``bubble_sort_steps``.  The ``swapped`` field is reused to
        indicate when the target is found (True means found).
    """
    steps: List[dict] = []
    # Initial state description
    steps.append({
        "arr": arr.copy(),
        "highlight": None,
        "swapped": False,
        "pass_no": 0,
        "desc": f"Start linear search for {target} in the array."
    })
    found = False
    for idx, val in enumerate(arr):
        if val == target:
            found = True
            desc = f"Checking index {idx}: {val} == {target}, target found."
            steps.append({
                "arr": arr.copy(),
                "highlight": (idx, idx),
                "swapped": True,
                "pass_no": 1,
                "desc": desc
            })
            break
        else:
            desc = f"Checking index {idx}: {val} != {target}, continue searching."
            steps.append({
                "arr": arr.copy(),
                "highlight": (idx, idx),
                "swapped": False,
                "pass_no": 1,
                "desc": desc
            })
    if not found:
        steps.append({
            "arr": arr.copy(),
            "highlight": None,
            "swapped": False,
            "pass_no": 1,
            "desc": f"Target {target} not found in the array."
        })
    return steps


def plot_array(arr: List[int], highlight: Tuple[int, int] | None = None, swapped: bool = False) -> Image.Image:
    """Render a bar chart for the current state of the list with optional highlighting.

    Bars corresponding to the indices in ``highlight`` are coloured differently
    to draw attention to the comparison.  If ``swapped`` is True, the
    highlighted bars are coloured red to indicate a swap; otherwise they are
    orange.

    Args:
        arr: A list of integers representing the array to plot.
        highlight: A tuple of two indices that are being compared.  If None,
            all bars are coloured uniformly.
        swapped: Whether the highlighted comparison resulted in a swap.

    Returns:
        A PIL Image of the generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors: List[str] = []
    for i, _ in enumerate(arr):
        if highlight and i in highlight:
            # Use red for swaps and orange otherwise
            colors.append("#e74c3c" if swapped else "#f39c12")
        else:
            colors.append("#4fa3d1")
    ax.bar(range(len(arr)), arr, color=colors)
    # Use a generic title because the same plotting function is reused for search
    # and sorting algorithms.
    ax.set_title("Algorithm Visualization", fontsize=14)
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
    """Convert a comma‑separated string into a list of integers.

    Raises:
        ValueError: If any item cannot be parsed as an integer.

    Args:
        text: A string containing numbers separated by commas.

    Returns:
        A list of parsed integers.
    """
    # Split on commas and strip whitespace
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("Please enter at least one integer.")
    return [int(part) for part in parts]


def run_sort(algorithm: str, numbers: str, case_type: str, target: str) -> Tuple[Any, Image.Image, str, str, str, List[dict]]:
    """Determine which algorithm to run and initialise the first state.

    This callback parses the user input, selects the appropriate
    algorithm (bubble sort or linear search), prepares the input list
    based on the chosen case (e.g. reversed order for worst‑case),
    computes the sequence of steps, and returns the initial UI values.

    Args:
        algorithm: Name of the chosen algorithm ("Bubble Sort" or "Linear Search").
        numbers: A comma‑separated sequence of integers supplied by the user.
        case_type: Which scenario to illustrate (e.g. "Custom Input" or "Worst-case (reversed)").
        target: The target value for search algorithms; ignored for sorting.

    Returns:
        A tuple of (slider update, image, description, array string, status, steps).
    """
    # Attempt to parse the list of numbers
    try:
        values = parse_numbers(numbers)
    except Exception as exc:
        return (
            gr.update(interactive=False),
            plot_array([], None, False),
            "",
            "",
            f"Error: {exc}",
            [],
        )
    # Default to bubble sort if algorithm is empty
    algorithm = algorithm or "Bubble Sort"
    case_type = case_type or "Custom Input"
    arr: List[int] = values.copy()
    # Select the appropriate algorithm and prepare steps
    if algorithm == "Bubble Sort":
        # Adjust the array based on the chosen case.  These options allow the
        # learner to see best‑case (already sorted), worst‑case (reversed),
        # randomised input or a stability demonstration with duplicates.
        lower_case = case_type.lower()
        if "best" in lower_case:
            arr = sorted(arr)
        elif "worst" in lower_case:
            arr = sorted(arr, reverse=True)
        elif "random" in lower_case:
            # Generate a random list of the same length (or 5 elements if none provided)
            size = len(arr) if arr else 5
            # Use a fixed range for randomness; duplicates may occur
            arr = [random.randint(1, 99) for _ in range(size)]
        elif "stability" in lower_case:
            # Predefined list with duplicates to highlight stable sorting
            arr = [3, 1, 3, 2, 3]
        # Compute steps using bubble sort
        steps = bubble_sort_steps(arr.copy())
    elif algorithm == "Linear Search":
        # Parse the target
        target_val: int
        try:
            target_val = int(target.strip()) if target.strip() != "" else None  # type: ignore
        except Exception:
            return (
                gr.update(interactive=False),
                plot_array([], None, False),
                "",
                "",
                "Error: Invalid target value.",
                [],
            )
        if target_val is None:
            return (
                gr.update(interactive=False),
                plot_array([], None, False),
                "",
                "",
                "Error: Please enter a target value for search.",
                [],
            )
        steps = linear_search_steps(arr.copy(), target_val)
    else:
        return (
            gr.update(interactive=False),
            plot_array([], None, False),
            "",
            "",
            f"Error: Unknown algorithm {algorithm}",
            [],
        )
    # Configure slider update
    slider_update = gr.update(
        minimum=0,
        maximum=len(steps) - 1,
        value=0,
        step=1,
        interactive=(len(steps) > 1),
    )
    # Get the first step for initial display
    first = steps[0]
    img = plot_array(first["arr"], highlight=first["highlight"], swapped=first["swapped"])
    description = first["desc"]
    status = f"Total steps: {len(steps)}" if len(steps) > 1 else "Done."
    array_str = ", ".join(str(x) for x in first["arr"]) if first["arr"] else "(empty)"
    return slider_update, img, description, array_str, status, steps


def update_plot(step: int, steps: List[dict]) -> Tuple[Image.Image, str, str, str]:
    """Update the chart, description and status as the user moves the slider.

    This callback reads the current slider position and returns the
    corresponding visualisation along with explanatory text.  A summary
    status shows either the slider position or a completion message when
    the end is reached.

    Args:
        step: The index selected by the slider.
        steps: The detailed list of step dictionaries stored in state.

    Returns:
        A tuple of (Image, description, status) for the chosen step.
    """
    # If no state is available, return empty outputs
    if not steps:
        # If no state exists, return empty outputs (image, description, array string, status)
        return plot_array([], None, False), "", "", ""
    idx = max(0, min(int(step), len(steps) - 1))
    current = steps[idx]
    img = plot_array(current["arr"], highlight=current["highlight"], swapped=current["swapped"])
    description = current["desc"]
    # Default status shows the current slider position
    status: str
    if len(steps) > 1:
        status = f"Step {idx} of {len(steps) - 1}"
    else:
        status = ""
    # If we are on the final step and it represents completion, adjust the message.
    # For search algorithms the completion message indicates whether the target
    # was found or not.  For sorting algorithms we refer to the number of passes.
    if idx == len(steps) - 1:
        desc_lower = description.lower()
        if "target" in desc_lower and "found" in desc_lower:
            status = "Completed: target found."
        elif "not found" in desc_lower:
            status = "Completed: target not found."
        else:
            pass_no = current.get("pass_no", 0)
            status = f"Completed: array sorted after {pass_no} pass{'es' if pass_no != 1 else ''}."
    # Represent the array as a comma‑separated string for display
    array_str = ", ".join(str(x) for x in current["arr"]) if current["arr"] else "(empty)"
    return img, description, array_str, status


# Additional callbacks for stepping through the algorithm using buttons
def next_step(current_step: int, steps: List[dict]) -> Tuple[Any, Image.Image, str, str, str]:
    """Advance to the next step when the user clicks the 'Next' button.

    Args:
        current_step: The current slider index.
        steps: The full list of steps captured by the algorithm.

    Returns:
        A tuple updating the slider value and all other visual/text outputs.
    """
    if not steps:
        return gr.update(value=0, interactive=False), plot_array([], None, False), "", "", ""
    new_idx = min(int(current_step) + 1, len(steps) - 1)
    img, desc, arr_str, status = update_plot(new_idx, steps)
    slider_update = gr.update(value=new_idx, interactive=(len(steps) > 1))
    return slider_update, img, desc, arr_str, status


def prev_step(current_step: int, steps: List[dict]) -> Tuple[Any, Image.Image, str, str, str]:
    """Return to the previous step when the user clicks the 'Previous' button.

    Args:
        current_step: The current slider index.
        steps: The full list of steps captured by the algorithm.

    Returns:
        A tuple updating the slider value and all other visual/text outputs.
    """
    if not steps:
        return gr.update(value=0, interactive=False), plot_array([], None, False), "", "", ""
    new_idx = max(int(current_step) - 1, 0)
    img, desc, arr_str, status = update_plot(new_idx, steps)
    slider_update = gr.update(value=new_idx, interactive=(len(steps) > 1))
    return slider_update, img, desc, arr_str, status

# Callback to show or hide the target input based on algorithm choice
def toggle_target_visibility(algorithm: str) -> Any:
    """
    Return a Gradio update specifying whether the target input should be visible.

    The target field is only relevant when performing a search (e.g. linear search).
    When the user selects a sorting algorithm, the target field can be hidden to
    simplify the interface.

    Args:
        algorithm: The name of the algorithm selected by the user.

    Returns:
        A Gradio update object setting the ``visible`` property of the target input.
    """
    return gr.update(visible=(algorithm == "Linear Search"))


def build_demo() -> gr.Blocks:
    """Create and return the Gradio interface for the bubble sort visualizer."""
    with gr.Blocks(title="Sorting & Searching Visualizer") as demo:
        # Introductory Markdown explaining the OOP concepts and usage.  Emojis and
        # bullet points make the text engaging and help learners see at a glance what
        # the app demonstrates.  This header also clarifies that multiple algorithms
        # and scenarios are available.
        gr.Markdown(
            """
            🎓 **CISC 121 – OOP Sorting & Searching Visualizer**

            Learn object‑oriented programming concepts through algorithm visualisation!

            **This app demonstrates key OOP concepts:**
            * 📦 **Classes & Objects:** Although this demo uses functions, the underlying
              algorithms can be thought of as objects with state and behaviour.
            * 🎭 **Inheritance:** All sorting algorithms share common behaviours like
              stepping through a list and producing a trace of actions.
            * 🔄 **Polymorphism:** Swap between sorting and searching algorithms
              seamlessly using the drop‑down below.
            * 🏭 **Factory Pattern:** The code selects the appropriate algorithm
              implementation based on your choice of algorithm and case.

            **How to use:**
            1. Choose an algorithm (e.g. Bubble Sort or Linear Search) and a case type.
            2. Enter a comma‑separated list of numbers.  For search, also provide
               the target value.
            3. Click **Run Algorithm** to generate the step‑by‑step visualisation.
            4. Use the slider or **Previous**/**Next** buttons to navigate through
               each comparison, swap or check.  Explanations and the current array
               update automatically.
            """
        )
        # Top row for algorithm selection and case type.  The algorithm determines
        # whether a target field is needed.  The case type controls how the input
        # array is prepared (custom, best‑case, worst‑case, random or stability demo).
        with gr.Row():
            algorithm_dropdown = gr.Dropdown(
                choices=["Bubble Sort", "Linear Search"],
                value="Bubble Sort",
                label="Algorithm",
            )
            case_selector = gr.Dropdown(
                choices=[
                    "Custom Input",
                    "Best-case (sorted)",
                    "Worst-case (reversed)",
                    "Random",
                    "Stability Demo",
                ],
                value="Custom Input",
                label="Case",
            )
            target_input = gr.Textbox(
                label="Search Target",
                placeholder="Enter value to find (for search)",
                lines=1,
                visible=False,
            )
        # Row for entering the input list and running the algorithm.  The text box
        # accepts comma‑separated integers and the button triggers the run_sort
        # callback with the algorithm, numbers, case and target.
        with gr.Row():
            input_text = gr.Textbox(
                label="Input Numbers",
                placeholder="e.g., 5, 3, 8, 4, 2, 7, 1",
                lines=1,
            )
            run_button = gr.Button("Run Algorithm")
        # Slider to navigate through steps.  It is always visible but disabled until
        # steps are generated.  Once an algorithm runs, the slider becomes
        # interactive and shows the range of available steps.
        step_slider = gr.Slider(
            minimum=0,
            maximum=0,
            value=0,
            step=1,
            label="Step",
            interactive=False,
        )
        # Image display for the bar chart
        chart_output = gr.Image(type="pil", label="Visualization")
        # Explanation area using Markdown (no label parameter for compatibility)
        desc_output = gr.Markdown(value="")
        # Display the current array as a string
        array_output = gr.Textbox(label="Current Array", interactive=False)
        # Status line to report progress or errors
        status_label = gr.Textbox(label="Status", interactive=False)
        # Hidden state to hold the list of steps between callbacks
        state = gr.State([])

        # Row of navigation buttons for stepping through the algorithm
        with gr.Row():
            prev_button = gr.Button("◀ Previous Step")
            next_button = gr.Button("Next Step ▶")

        # Connect previous and next buttons to their callbacks
        prev_button.click(
            prev_step,
            inputs=[step_slider, state],
            outputs=[step_slider, chart_output, desc_output, array_output, status_label],
        )
        next_button.click(
            next_step,
            inputs=[step_slider, state],
            outputs=[step_slider, chart_output, desc_output, array_output, status_label],
        )

        # When the user runs the algorithm: pass the selected algorithm, numbers,
        # case and target.  The callback returns the configured slider, image,
        # description, array string, status and step list.
        run_button.click(
            run_sort,
            inputs=[algorithm_dropdown, input_text, case_selector, target_input],
            outputs=[step_slider, chart_output, desc_output, array_output, status_label, state],
        )
        # When the slider value changes: update the chart, explanation, array and status
        step_slider.change(
            update_plot,
            inputs=[step_slider, state],
            outputs=[chart_output, desc_output, array_output, status_label],
        )

        # Show or hide the target input when the algorithm changes.  When
        # ``Linear Search`` is selected the target field appears; otherwise it is hidden.
        algorithm_dropdown.change(
            toggle_target_visibility,
            inputs=[algorithm_dropdown],
            outputs=[target_input],
        )
    return demo


if __name__ == "__main__":
    # Build and launch the interface if this file is executed directly
    demo = build_demo()
    # Set server_name="0.0.0.0" to make the app accessible externally when
    # deploying on platforms like Hugging Face Spaces
    demo.launch()

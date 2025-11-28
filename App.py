import io
from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

def bubble_sort_steps(arr: List[int]) -> List[List[int]]:
    """Return a list of array states for each comparison/swapping pass."""
    steps: List[List[int]] = []
    steps.append(arr.copy())
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
            steps.append(arr.copy())
        if not swapped:
            break
    return steps

def plot_array(arr: List[int]) -> Image.Image:
    """Render a simple bar chart for the current state of the list."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(arr)), arr, color="#4fa3d1")
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

def run_sort(numbers: str) -> Tuple[gr.components.Slider, Image.Image, str, List[List[int]]]:
    """Callback to execute when the user clicks the button."""
    try:
        values = parse_numbers(numbers)
    except Exception as exc:
        return (
            gr.update(visible=False),
            plot_array([]),
            f"Error: {exc}",
            [],
        )
    steps = bubble_sort_steps(values.copy())
    slider_update = gr.update(visible=True, minimum=0, maximum=len(steps) - 1, value=0, step=1)
    img = plot_array(steps[0])
    status = f"Total steps: {len(steps)}"
    return slider_update, img, status, steps

def update_plot(step: int, steps: List[List[int]]) -> Image.Image:
    """Update the bar chart when the slider value changes."""
    if not steps:
        return plot_array([])
    idx = max(0, min(int(step), len(steps) - 1))
    return plot_array(steps[idx])

def build_demo() -> gr.Blocks:
    """Create and return the Gradio interface for the bubble sort visualizer."""
    with gr.Blocks(title="Bubble Sort Visualizer") as demo:
        gr.Markdown(
            """
            # Bubble Sort Visualizer

            Enter a list of integers separated by commas and click **Run Bubble Sort**
            to see how the bubble sort algorithm gradually orders the numbers.
            Use the step slider to move through each comparison and swap.
            """
        )
        with gr.Row():
            input_text = gr.Textbox(
                label="Input Numbers",
                placeholder="e.g., 5, 3, 8, 4, 2, 7, 1",
                lines=1,
            )
            run_button = gr.Button(label="Run Bubble Sort", variant="primary")
        step_slider = gr.Slider(
            minimum=0,
            maximum=0,
            value=0,
            step=1,
            label="Step",
            visible=False,
        )
        chart_output = gr.Image(type="pil", label="Visualization")
        status_label = gr.Textbox(label="Status", interactive=False)
        state = gr.State([])
        run_button.click(
            run_sort,
            inputs=[input_text],
            outputs=[step_slider, chart_output, status_label, state],
        )
        step_slider.change(
            update_plot,
            inputs=[step_slider, state],
            outputs=chart_output,
        )
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()

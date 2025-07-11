import numpy as np
import napari
from magicgui import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ExperimentStatsWidget(QWidget):
	def __init__(self):
		super().__init__()
		layout = QVBoxLayout()
		self.label = QLabel("Experiment-wide statistics will go here.")
		self.canvas = FigureCanvas(Figure(figsize=(4, 3)))
		layout.addWidget(self.label)
		layout.addWidget(self.canvas)
		self.setLayout(layout)

	def update_plot(self, all_nuclei_stats):
		# Example: histogram of nucleus areas
		areas = [stat['area'] for img_stats in all_nuclei_stats for stat in img_stats]
		ax = self.canvas.figure.subplots()
		ax.clear()
		ax.hist(areas, bins=20)
		ax.set_title("Nucleus Area Distribution")
		ax.set_xlabel("Area")
		ax.set_ylabel("Count")
		self.canvas.draw()


def run_gui(image_list, nuclei_masks, nuclei_stats_list):
	viewer = napari.Viewer()
	stats_widget = ExperimentStatsWidget()
	viewer.window.add_dock_widget(stats_widget, area='right')

	@magicgui(auto_call=True, image_index={"min": 0, "max": len(image_list) - 1})
	def slider_widget(image_index: int = 0):
		image_layer.data = image_list[image_index]
		mask_layer.data = nuclei_masks[image_index]
		current_stats.clear()
		for stat in nuclei_stats_list[image_index]:
			current_stats.append(stat)

	viewer.window.add_dock_widget(slider_widget, area='bottom')

	# Initial layers
	image_layer = viewer.add_image(image_list[0], name='Hydra Image')
	mask_layer = viewer.add_labels(nuclei_masks[0], name='Nuclei Mask')

	current_stats = nuclei_stats_list[0]  # Start with the first

	# Mouse hover event
	@viewer.mouse_move_callbacks.append
	def on_mouse_move(viewer, event):
		coords = np.round(event.position).astype(int)
		if mask_layer.data.ndim == 2 and all(0 <= c < s for c, s in zip(coords[::-1], mask_layer.data.shape)):
			label_val = mask_layer.data[tuple(coords[::-1])]
			if label_val > 0:
				stat = next((s for s in current_stats if s["label"] == label_val), None)
				if stat:
					viewer.status = f"Label: {label_val}, Area: {stat['area']}, Centroid: {stat['centroid']}"
			else:
				viewer.status = "No nucleus under cursor."

	# Button to show experiment-wide stats
	@magicgui(call_button="Show Experiment Stats")
	def stats_button():
		stats_widget.update_plot(nuclei_stats_list)

	viewer.window.add_dock_widget(stats_button, area='right')

	napari.run()


if __name__ == "__main__":
	# Dummy data for testing
	n_images = 5
	img_size = (256, 256)
	dummy_images = [np.random.rand(*img_size) for _ in range(n_images)]
	dummy_masks = []
	dummy_stats = []

	for _ in range(n_images):
		mask = np.zeros(img_size, dtype=np.int32)
		stats = []
		for i in range(1, 6):
			cx, cy = np.random.randint(50, 200, 2)
			rr, cc = np.ogrid[:img_size[0], :img_size[1]]
			circle = (rr - cy) ** 2 + (cc - cx) ** 2 < 20 ** 2
			mask[circle] = i
			stats.append({'label': i, 'area': int(np.sum(circle)), 'centroid': (cx, cy)})
		dummy_masks.append(mask)
		dummy_stats.append(stats)

	run_gui(dummy_images, dummy_masks, dummy_stats)

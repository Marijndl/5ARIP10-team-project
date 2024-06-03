import sys
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QSlider, QLabel, \
    QComboBox
from pyvistaqt import QtInteractor
from scipy.spatial.transform import Rotation as R


class PyVistaWindow(QWidget):
    def __init__(self, parent=None):
        super(PyVistaWindow, self).__init__(parent)
        self.plotter = QtInteractor(self)
        self.plotter.setFixedSize(800, 600)
        self.plotter.set_background('white')
        self.timer = QTimer()
        self.timer.timeout.connect(self.perform_rotation)
        self.rotation_direction = None

        # Add axis orientation widget
        self.plotter.add_axes(interactive=True)

        # Create buttons for movement and camera control
        self.rotate_acw_button = QPushButton('Rotate anti-clockwise')
        self.rotate_acw_button.setIcon(QIcon('rotate_left.jpg'))
        self.rotate_acw_button.pressed.connect(self.start_rotating_acw)
        self.rotate_acw_button.released.connect(self.stop_rotating)

        self.rotate_cw_button = QPushButton('Rotate clockwise')
        self.rotate_cw_button.setIcon(QIcon('rotate_right.jpg'))
        self.rotate_cw_button.pressed.connect(self.start_rotating_cw)
        self.rotate_cw_button.released.connect(self.stop_rotating)

        self.rotate_up_button = QPushButton('Rotate up')
        self.rotate_up_button.setIcon(QIcon('rotate_left.jpg'))
        self.rotate_up_button.pressed.connect(self.start_rotating_up)
        self.rotate_up_button.released.connect(self.stop_rotating)

        self.rotate_down_button = QPushButton('Rotate down')
        self.rotate_down_button.setIcon(QIcon('rotate_right.jpg'))
        self.rotate_down_button.pressed.connect(self.start_rotating_down)
        self.rotate_down_button.released.connect(self.stop_rotating)

        self.move_up_button = self.create_button('Move Up', None, self.move_up)
        self.move_down_button = self.create_button('Move Down', None, self.move_down)
        self.front_button = self.create_button('Move Front', None, self.move_front)
        self.back_button = self.create_button('Move Back', None, self.move_back)
        self.reset_camera_button = self.create_button('Reset Camera', None, self.reset_camera)

        # Radio buttons for color map
        self.toggle_points_button = self.create_toggle_button('Show Vessels', True, self.toggle_points_visibility)
        self.toggle_depth_color = self.create_toggle_button('Depth Coloring', False, self.depth_coloring)

        # Camera standard position dropdown
        self.camera_position_dropdown = QComboBox()
        self.camera_position_dropdown.addItems(["Front", "Top", "Bottom", "Left", "Right", "Back", "Isometric"])
        self.camera_position_dropdown.activated[str].connect(self.set_camera_position)

        # Label to display current position
        # self.position_label = QLabel('Position: (0, 0, 0)')

        self.camera_label = self.create_label('<b>Camera controls</b>', 150, 30)
        self.overlay_label = self.create_label('<b>Overlay controls</b>', 150, 30)
        self.line_opacity_label, self.line_opacity_slider = self.create_opacity_slider('Line Opacity',
                                                                                       self.set_line_opacity)
        self.image_opacity_label, self.image_opacity_slider = self.create_opacity_slider('Image Opacity',
                                                                                         self.set_image_opacity)

        # Create layout controls
        controls_layout = QVBoxLayout()

        # Create layout for Camera controls
        camera_layout = self.create_camera_layout()

        # Create layout for overlay controls
        overlay_layout = self.create_overlay_layout()

        controls_layout.addLayout(camera_layout)
        controls_layout.addLayout(overlay_layout)

        layout = QHBoxLayout()
        layout.addWidget(self.plotter.interactor)
        layout.addLayout(controls_layout)

        self.interactor = self.plotter.interactor
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.style)

        # Add observer for left button press event & release event
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        self.interactor.AddObserver("EndInteractionEvent", self.on_left_button_release)

        self.setLayout(layout)
        self.setWindowTitle("GUI interactive 3D overlay")
        self.setGeometry(100, 100, 800, 600)
        self.show()

        # Map depth values to colors (you can choose a colormap)
        self.add_line('red')
        self.add_line('summer')
        self.line_colored = False
        # Load an image
        image_path = "Assets/images.jpeg"  # Provide the path to your image
        try:
            texture = pv.read_texture(image_path)
        except Exception as e:
            print("Error loading texture:", e)

        # Create a plane mesh and apply the texture
        self.plane = pv.Plane(center=(30, -36, 0), direction=(0, 0, 1), i_size=100, j_size=100, i_resolution=1,
                              j_resolution=1)  # Increase size
        self.plane_actor = self.plotter.add_mesh(self.plane, texture=texture, ambient=1.0, show_edges=True,
                                                 color='white', opacity=0.5)

        self.plotter.view_xy()
        self.view = 'front'
        self.rotation_axis = None
        self.set_view('Front')
        self.image_opacity_slider.setValue(50)

    def create_button(self, text, icon_path, on_clicked):
        button = QPushButton(text)
        if icon_path:
            button.setIcon(QIcon(icon_path))
        button.clicked.connect(on_clicked)
        return button

    def create_toggle_button(self, text, checked, on_clicked):
        button = QPushButton(text)
        button.setCheckable(True)
        button.setChecked(checked)
        button.clicked.connect(on_clicked)
        return button

    def create_label(self, text, width, height):
        label = QLabel(text)
        label.setFixedSize(width, height)
        return label

    def create_opacity_slider(self, label_text, on_value_changed):
        # Create the label
        opacity_label = self.create_label(label_text, 150, 30)

        # Create the opacity slider
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(100)
        opacity_slider.valueChanged.connect(on_value_changed)

        return opacity_label, opacity_slider

    def create_camera_layout(self):
        camera_layout = QVBoxLayout()
        camera_layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_position_dropdown)
        camera_layout.addWidget(self.rotate_up_button)
        camera_left_right_layout = QHBoxLayout()
        camera_left_right_layout.addWidget(self.rotate_acw_button)
        camera_left_right_layout.addWidget(self.rotate_cw_button)
        camera_layout.addLayout(camera_left_right_layout)
        camera_layout.addWidget(self.rotate_down_button)
        camera_layout.addWidget(self.reset_camera_button)
        # camera_layout.addWidget(self.position_label)
        return camera_layout

    def create_overlay_layout(self):
        overlay_layout = QVBoxLayout()
        overlay_layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        overlay_layout.addWidget(self.overlay_label)
        overlay_layout.addWidget(self.toggle_points_button)
        overlay_layout.addWidget(self.toggle_depth_color)
        overlay_layout.addWidget(self.line_opacity_label)
        overlay_layout.addWidget(self.line_opacity_slider)
        overlay_layout.addWidget(self.image_opacity_label)
        overlay_layout.addWidget(self.image_opacity_slider)
        return overlay_layout

    def add_line(self, color) -> None:
        csv_file = "combined_segment_points.csv"
        try:
            df = pd.read_csv(csv_file)
            l = df['l']
            p = df['p']
            s = df['s']

            # Convert data to Cartesian coordinates
            z_data = l * np.sin(np.radians(p)) * np.cos(np.radians(s))
            x_data = l * np.sin(np.radians(p)) * np.sin(np.radians(s))
            y_data = l * np.cos(np.radians(p))

            # Create a PyVista line
            points = np.column_stack((x_data, y_data, z_data))
            self.line = pv.PolyData(points)
            z_coords = points[:, 2]
            if color != 'red':
                self.line_actor_color = self.plotter.add_mesh(self.line, scalars=z_coords, cmap=color,
                                                              show_scalar_bar=False)
                self.line_actor_color.visibility = False
            else:
                self.line_actor_red = self.plotter.add_mesh(self.line, color=color)
                self.line_actor_red.visibility = True
            self.line_visible = True


        except Exception as e:
            print("Error loading data:", e)

    def camera_around_axis(self, angle):
        if self.rotation_axis is None:
            return

        # Create a quaternion representing the rotation
        rotation = R.from_rotvec(self.rotation_axis * angle)

        # Get current camera position
        cam_pos, focal_point, view_up = self.plotter.camera_position

        # Compute the vector from the camera position to the focal point
        view_vector = [focal_point[i] - cam_pos[i] for i in range(3)]

        # Rotate the view vector
        rotated_view_vector = rotation.apply(view_vector)

        # Compute the new camera position
        new_cam_pos = focal_point - rotated_view_vector

        # Rotate the view-up vector
        rotated_view_up = rotation.apply(view_up)

        # Update camera position
        self.plotter.camera_position = [new_cam_pos, focal_point, rotated_view_up]
        # update the window
        self.plotter.update()

    def rotate_acw(self):
        self.camera_around_axis(-0.5)  # Rotation angle in degrees

    def rotate_cw(self):
        self.camera_around_axis(0.5)  # Rotation angle in degrees

    def set_rotation_axis(self, axis):
        self.rotation_axis = axis / np.linalg.norm(axis)

    def set_view(self, view):
        if view in ["Top", "Bottom"]:
            self.set_rotation_axis(np.array([1, 0, 0]))
        elif view in ["Left", "Right"]:
            self.set_rotation_axis(np.array([0, 1, 0]))
        else:
            self.set_rotation_axis(np.array([0, 1, 0]))  # Default axis for other views

    def move_up(self):
        transform = np.eye(4)
        transform[1, 3] += 1
        self.plane.points = np.dot(self.plane.points, transform[:3, :3].T) + transform[:3, 3]
        self.update_position_label()
        self.plotter.reset_camera()

    def move_down(self):
        transform = np.eye(4)
        transform[1, 3] -= 1
        self.plane.points = np.dot(self.plane.points, transform[:3, :3].T) + transform[:3, 3]
        self.update_position_label()
        self.plotter.reset_camera()

    def move_front(self):
        transform = np.eye(4)
        transform[2, 3] += 1
        self.plane.points = np.dot(self.plane.points, transform[:3, :3].T) + transform[:3, 3]
        self.update_position_label()
        self.plotter.reset_camera()

    def move_back(self):
        transform = np.eye(4)
        transform[2, 3] -= 1
        self.plane.points = np.dot(self.plane.points, transform[:3, :3].T) + transform[:3, 3]
        self.update_position_label()
        self.plotter.reset_camera()

    def reset_camera(self):
        self.set_camera_position('Front')

    def toggle_points_visibility(self):
        if self.toggle_points_button.isChecked():
            if self.line_colored:
                self.line_actor_color.visibility = True
                self.line_actor_red.visibility = False
            else:
                self.line_actor_red.visibility = True
                self.line_actor_color.visibility = False

            self.toggle_depth_color.setEnabled(True)
        else:
            self.line_actor_color.visibility = False
            self.line_actor_red.visibility = False
            self.toggle_depth_color.setEnabled(False)

    def update_position_label(self):
        position = self.plane.points.mean(axis=0)
        self.position_label.setText(f'Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})')

    def rotate_vector(self, vector, angle, axis):
        angle_rad = np.radians(angle)
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        cross_matrix = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = (
                cos_theta * np.eye(3) +
                sin_theta * cross_matrix +
                (1 - cos_theta) * np.outer(axis, axis)
        )
        return rotation_matrix.dot(vector)

    def set_camera_position(self, text):
        self.rotate_cw_button.setEnabled(True)
        self.rotate_acw_button.setEnabled(True)
        self.set_view(text)
        scale_factor = 300  # Increase this value to zoom out
        view_up = (0, 1, 0)  # Up direction along the y-axis
        focal_point = (30, -36, 0)  # Typically, the focal point is the origin
        if text == "Isometric":
            view_vector = (0.8, 0.8, 0.8)
        elif text == "Top":
            view_vector = (0, 1, 0)
            angle = 10  # Rotation angle in degrees
            view_vector = self.rotate_vector(view_vector, angle, (1, 0, 0))
            view_up = (0, 0, -1)
        elif text == "Bottom":
            view_vector = (0, -1, 0)
            view_up = (0, 0, 1)
        elif text == "Left":
            view_vector = (-1, 0, 0)
            angle = 10  # Rotation angle in degrees
            view_vector = self.rotate_vector(view_vector, angle, view_up)
        elif text == "Right":
            view_vector = (1, 0, 0)
            angle = -10  # Rotation angle in degrees
            view_vector = self.rotate_vector(view_vector, angle, view_up)
        elif text == "Front":
            view_vector = (0, 0, 1)
        elif text == "Back":
            view_vector = (0, 0, -1)

        camera_position = [focal_point[i] + scale_factor * view_vector[i] for i in range(3)]
        self.plotter.camera_position = [camera_position, focal_point, view_up]
        self.view = text

    def depth_coloring(self):
        if self.toggle_depth_color.isChecked():
            self.line_actor_color.visibility = True
            self.line_actor_red.visibility = False
            self.line_colored = True
        else:
            self.line_actor_color.visibility = False
            self.line_actor_red.visibility = True
            self.line_colored = False

    def on_left_button_press(self, obj, event):
        self.camera_position = self.plotter.camera_position
        self.rotate_cw_button.setEnabled(False)
        self.rotate_acw_button.setEnabled(False)
        global click_on
        click_on = 1
        print('pressed')

    def on_left_button_release(self, obj, event):
        global click_on
        if click_on == 1:
            print('released')
            click_on = 0
            self.plotter.camera_position = self.camera_position
            self.rotate_cw_button.setEnabled(True)
            self.rotate_acw_button.setEnabled(True)

    def start_rotating_up(self):
        self.rotation_direction = 'up'
        self.timer.start(10)

    def start_rotating_down(self):
        self.rotation_direction = 'down'
        self.timer.start(10)

    def start_rotating_acw(self):
        self.rotation_direction = 'acw'
        self.timer.start(10)  # Adjust the interval as needed for smoother rotation

    def start_rotating_cw(self):
        self.rotation_direction = 'cw'
        self.timer.start(10)  # Adjust the interval as needed for smoother rotation

    def stop_rotating(self):
        self.timer.stop()
        self.rotation_direction = None

    def perform_rotation(self):
        if self.rotation_direction == 'acw':
            self.set_rotation_axis(np.array([0, 1, 0]))
            self.camera_around_axis(0.02)
        elif self.rotation_direction == 'cw':
            self.set_rotation_axis(np.array([0, 1, 0]))
            self.camera_around_axis(-0.02)
        elif self.rotation_direction == 'up':
            self.set_rotation_axis(np.array([1, 0, 0]))
            self.camera_around_axis(-0.02)
        elif self.rotation_direction == 'down':
            self.set_rotation_axis(np.array([1, 0, 0]))
            self.camera_around_axis(0.02)

    def set_line_opacity(self, value):
        opacity = value / 100
        self.line_actor_color.GetProperty().SetOpacity(opacity)
        self.line_actor_red.GetProperty().SetOpacity(opacity)

    def set_image_opacity(self, value):
        opacity = value / 100
        self.plane_actor.GetProperty().SetOpacity(opacity)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PyVistaWindow()
    sys.exit(app.exec_())

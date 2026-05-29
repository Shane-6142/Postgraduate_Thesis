import numpy as np
import matplotlib.patches as patches

DIRECTIONS = {
    "right": 0,
    "right_up": 45,
    "up": 90,
    "left_up": 135,
    "left": 180,
    "left_down": 225,
    "down": 270,
    "right_down": 315
}

class Agent:
    def __init__(self, grid_x=0, grid_y=0):

        self.grid_x = grid_x
        self.grid_y = grid_y
        self.center_x = grid_x + 0.5
        self.center_y = grid_y + 0.5
        self.current_angle = 0
        self.color = (0, 255, 0)
        self.is_hold = False
        self.hold_obj_type = None
        self.hold_color_rgb = None
        self.collision_times = 0
        self.triangle = self._create_triangle()

    def _create_triangle(self):

        base_vertices = np.array([
            [0.3, 0],
            [-0.3, -0.3],
            [-0.3, 0.3]
        ])
        theta = np.radians(self.current_angle)
        rot_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        rotated_vertices = np.dot(base_vertices, rot_matrix.T) + [self.center_x, self.center_y]

        if self.is_hold and self.hold_color_rgb is not None:
            face_color = np.array(self.hold_color_rgb) / 255
            triangle = patches.Polygon(
                rotated_vertices,
                facecolor=face_color,
                label=f"agent_hold_{self.hold_obj_type}",
                zorder=30
            )
        else:
            face_color = np.array(self.color) / 255
            triangle = patches.Polygon(
                rotated_vertices,
                facecolor=face_color,
                label="agent",
                zorder=30
            )
            
        return triangle
    
    def get_xy(self):
        return self.grid_x, self.grid_y
    
    def get_state(self):
        return self.is_hold

    def rotate_to(self, direction, ax):
        if direction not in DIRECTIONS:
            return
        
        if self.current_angle == DIRECTIONS[direction]:
            return
        else:
            self.current_angle = DIRECTIONS[direction]
            new_triangle = self._create_triangle()
            self.triangle.remove()
            self.triangle = new_triangle
            ax.add_patch(self.triangle)
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
    
    def move(self, ax, map_data, cols=10, rows=10):
        theta = np.radians(self.current_angle)
        dx = np.cos(theta)
        dy = np.sin(theta)

        new_grid_x = self.grid_x + round(dx)
        new_grid_y = self.grid_y - round(dy)
        self.grid_x = round(self.center_x - 0.5)
        self.grid_y = round(self.center_y - 0.5)

        if new_grid_x < 0 or new_grid_x >= cols or new_grid_y < 0 or new_grid_y >= rows:
            self.collision_times += 1
            return

        current_pixel = tuple(map_data[new_grid_y, new_grid_x])
        # empty_color = [(255, 255, 255), (255, 0, 255)]

        if current_pixel == (0, 0, 0):
            self.collision_times += 1
            return

        self.grid_x = new_grid_x
        self.grid_y = new_grid_y
        self.center_x = self.grid_x + 0.5
        self.center_y = self.grid_y + 0.5

        new_triangle = self._create_triangle()
        self.triangle.remove()
        self.triangle = new_triangle
        ax.add_patch(self.triangle)
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()

    def pick(self, ax, map_data, ax_img, obj_type, hold_color_rgb):
        if self.is_hold:
            print("Already holding an item")
            return False
        if obj_type not in ["cube", "key"]:
            print(f"Cannot pick up object type: {obj_type}")
            return False
        target_grid_x = self.grid_x
        target_grid_y = self.grid_y
        self.is_hold = True
        self.hold_obj_type = obj_type
        self.hold_color_rgb = hold_color_rgb
        map_data[target_grid_y, target_grid_x] = (255, 255, 255)
        ax_img.set_data(map_data)
        white_rect = patches.Rectangle(
            (target_grid_x, target_grid_y),
            1,
            1,
            facecolor="white",
            edgecolor="none",
            zorder=15
        )
        ax.add_patch(white_rect)
        new_triangle = self._create_triangle()
        self.triangle.remove()
        self.triangle = new_triangle
        ax.add_patch(self.triangle)
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        print(f"Pick up {obj_type} successfully")
        return True
        

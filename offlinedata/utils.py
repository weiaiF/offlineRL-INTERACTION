import numpy as np
import torch

def point_in_rectangle(point, rect_min, rect_max):
    """
    Check if a point is inside a rectangle
    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]

def point_in_rotated_rectangle(point, center, length, width, angle):
    """
    Check if a point is inside a rotated rectangle
    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, [-length/2, -width/2], [length/2, width/2])


def has_corner_inside(rect1, rect2):
    """
    Check if rect1 has a corner inside rect2
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l1v = np.array([l1/2, 0])
    w1v = np.array([0, w1/2])
    r1_points = np.array([[0, 0],
                          - l1v, l1v, -w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1+np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def rotated_rectangles_intersect(rect1, rect2):
    """
    Do two rotated rectangles intersect?
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)

class ReplayBuffer(object):
    # 注意这里我将相应的buffer_size变小了
    def __init__(self, state_dim, action_dim, device, max_size=int(2e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)


    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")
        
        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]
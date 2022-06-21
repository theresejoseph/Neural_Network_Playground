__credits__ = ["Andrea PIERRÉ"]

import math
from multiprocessing.dummy import freeze_support
from typing import Optional, Union

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib
# matplotlib.use("Qt4Agg") # set the backend
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle

from multiprocessing import Process, Pool
import pygame
import matplotlib.widgets as wig
from matplotlib.patches import Rectangle 
import multiprocessing
import time 

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 550
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color = self.env.road_color / 255
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env_new_lap = True
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    """
    ### Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gym/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ### Action Space
    There are 3 actions: steering (-1 is full left, +1 is full right), gas,
    and breaking.

    ### Observation Space
    State consists of 96x96 pixels.

    ### Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited,
    where N is the total number of tiles visited in the track. For example,
    if you have finished in 732 frames, your reward is
    1000 - 0.1*732 = 926.8 points.

    ### Starting State
    The car starts at rest in the center of the road.

    ### Episode Termination
    The episode finishes when all of the tiles are visited. The car can also go
    outside of the playfield - that is, far off the track, in which case it will
    receive -100 reward and die.

    ### Arguments
    `lap_complete_percent` dictates the percentage of tiles that must be visited by
    the agent before a lap is considered complete.

    Passing `domain_randomize=True` enables the domain randomized variant of the environment.
    In this scenario, the background and track colours are different on every reset.

    Passing `continuous=False` converts the environment to use discrete action space.
    The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ### Version History
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ### References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ### Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        EzPickle.__init__(self)
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self._init_colors()

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])

        if not return_info:
            return self.step(None)[0]
        else:
            return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, {}

    def render(self, mode: str = "human"):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )

        pygame.font.init()

        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.screen is None and mode == "human":
            x = 0
            y = 0
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(self.surf, zoom, trans, angle, mode != "state_pixels")

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

        if mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, W, H):
        import pygame

        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        import pygame
        from pygame import gfxdraw

        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        import pygame

        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            self.isopen = False
            pygame.quit()


'''Parameters'''
N=[360,360] #number of neurons
neurons=[np.arange(0,N[0]), np.arange(0,N[1])]
curr_Neuron=[0,0]
num_links=[30,30]
excite=[20,47]
activity_mag=[1,1]
inhibit_scale=[0.005,0.005]
curr_parameter=[0,0]

class attractorNetworkSettling:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=excite_radius
        self.N=N  
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale
        
    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite.append((id + i) % self.N)
        return np.array(excite)

    def activation(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((int(id) + i) % self.N)
        return np.array(excite)

    def full_weights(self,radius):
        x=np.arange(-radius,radius+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def fractional_weights(self,non_zero_prev_weights,activeNeuron):
        frac=activeNeuron%1
        if frac == 0:
            return non_zero_prev_weights
        else: 
            inv_frac=1-frac
            frac_weights=np.zeros((len(non_zero_prev_weights)))
            frac_weights[0]=non_zero_prev_weights[0]*inv_frac
            for i in range(1,len(non_zero_prev_weights)):
                frac_weights[i]=non_zero_prev_weights[i-1]*frac + non_zero_prev_weights[i]*inv_frac
            return frac_weights

    def update_weights_dynamics(self,prev_weights,activeNeuron):

        delta=(int(activeNeuron)-np.argmax(prev_weights))%self.N

        indexes,non_zero_weights,non_zero_weights_shifted, inhbit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0
        # shifted_indexes=self.neuron_update(prev_weights)

        '''copied and shifted activity'''
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights

        if len(prev_weights[non_zero_idxs])==0:
            prev_weights[self.activation(activeNeuron)]=self.full_weights(self.num_links)
            non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        
        non_zero_weights[non_zero_idxs]=prev_weights[non_zero_idxs] 
        
        non_zero_weights_shifted[(non_zero_idxs+delta)%self.N]=self.fractional_weights(prev_weights[non_zero_idxs],activeNeuron) #non zero weights shifted by delta
        
        '''inhibition'''
        for i in range(len(non_zero_weights_shifted)):
            inhbit_val+=non_zero_weights_shifted[i]*self.inhibit_scale
        
        '''excitation'''
        excite=np.zeros(self.N)
        for i in range(len(non_zero_idxs)):
            excite[self.excitations(non_zero_idxs[i])]+=self.full_weights(self.excite_radius)*prev_weights[non_zero_idxs[i]]

        prev_weights+=(non_zero_weights_shifted+excite-inhbit_val)
        return prev_weights/np.linalg.norm(prev_weights)


def activityDecoding(prev_weights,radius,N,neurons):
    '''Isolating activity at a radius around the peak to decode position'''
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    weighted_sum=0
    for i in range(len(local_activity_idx)):
        weighted_sum+=local_activity_idx[i]*prev_weights[local_activity_idx[i]]
    return weighted_sum


def activityDecodingAngle(prev_weights,radius,N,neurons):
    '''Isolating activity at a radius around the peak to decode position'''
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(math.atan2(sum(y),sum(x))) % 360
    # changing range from [-179, 180] to [0,360]
    # if vect_sum<0:
    #     shifted_vec=vect_sum+360
    # else:
    #     shifted_vec=vect_sum
    # return shifted_vec*(N/360)
    return vect_sum


def driving_func(queue):
   a = np.array([0.0, 0.0, 0.0])
   import pygame

   def register_input():
      for event in pygame.event.get():
         if event.type == pygame.KEYDOWN:
               if event.key == pygame.K_LEFT:
                  a[0] = -1.0
               if event.key == pygame.K_RIGHT:
                  a[0] = +1.0                 
               if event.key == pygame.K_UP:
                  a[1] = +1.0     
               if event.key == pygame.K_DOWN:
                  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
               if event.key == pygame.K_RETURN:
                  global restart
                  restart = True
                  

         if event.type == pygame.KEYUP:
               if event.key == pygame.K_LEFT:
                  a[0] = 0
               if event.key == pygame.K_RIGHT:
                  a[0] = 0
               if event.key == pygame.K_UP:
                  a[1] = 0
               if event.key == pygame.K_DOWN:
                  a[2] = 0
               

   env = CarRacing()
   env.render()
   isopen = True
   
   while isopen:
      env.reset()
      total_reward = 0.0
      steps = 0
      global restart
      restart = False
      while True:

         register_input()

         posX= env.car.hull.position[0]
         posY= env.car.hull.position[1]

         linV=np.sqrt(np.square(env.car.hull.linearVelocity[0])+ np.square(env.car.hull.linearVelocity[1]))
         angV=env.car.hull.angularVelocity
         

         s, r, done, info = env.step(a)
         queue.put((posX,posY,done,restart))
         total_reward += r
         
         # if steps % 200 == 0 or done:
         #       print("\naction " + str([f"{x:+0.2f}" for x in a]))
         #       print(f"step {steps} total_reward {total_reward:+0.2f}")
         steps += 1
         isopen = env.render()

         time.sleep(0.1)
         
         if done or restart or isopen is False:
               break
   env.close()


def matplotlib_func(queue):
   # matplotlib stuff
   global curr_x, curr_y,decoded_x, decoded_y, pause, prev_weights
   curr_x, curr_y=[],[]
   decoded_x,decoded_y=[],[]
   pause=False 

   figw, figh = 9, 8
   fig = plt.figure(figsize=(figw, figh))
   # plt.get_current_fig_manager().window.setGeometry(500,0,800,800)
   # ax0 =  plt.subplot2grid(shape=(9, 16), loc=(2, 0), rowspan=7,colspan=7)
   # axx =  plt.subplot2grid(shape=(9, 16), loc=(0, 0), rowspan=2, colspan=7)
   # axy =  plt.subplot2grid(shape=(9, 16), loc=(2, 7), rowspan=7, colspan=2)

   # ax1 = plt.subplot2grid(shape=(9, 16), loc=(0, 10), colspan=6, rowspan = 9, facecolor="#15B01A")

   ax1 = plt.subplot(1,2,2, facecolor="#15B01A")
   ax0 = plt.subplot(1,2,1)
   plt.subplots_adjust(bottom=0.3)

   '''Slider for Parameters'''
   button_ax = plt.axes([.05, .03, .05, .04]) # x, y, width, height
   Nax = plt.axes([0.25, 0.17, 0.65, 0.03])
   exciteax = plt.axes([0.25, 0.1, 0.65, 0.03])
   inhax = plt.axes([0.25, 0.03, 0.65, 0.03])
   # Create a slider from 0.0 to 20.0 in axes axfreq with 3 as initial value
   start_stop=wig.Button(button_ax,label='$\u25B6$')
   # reset=wig.Button(button2_ax,'Reset')
   inhibit_scale=wig.Slider(inhax, 'Inhibition', 0, 0.05, 0.005)
   excite = wig.Slider(exciteax, 'Excitation', 1, 60, 10, valstep=2)
   N = wig.Slider(Nax, 'Neurons', 50, 300, 90, valstep=20)
   # delta2 = wig.Slider(delta2ax, 'Delta 2', -10, 10, 0, valstep=1)

   '''Initalise network'''            
   delta=[0,0]
   prev_weights=[np.zeros(int(N.val)), np.zeros(int(N.val))]
   # for i in range(len(delta)):
   #    net=attractorNetworkSettling(int(N.val),num_links[i],int(excite.val), activity_mag[i],inhibit_scale.val)
   #    prev_weights[i][net.activation(delta[i])]=net.full_weights(num_links[i])

   def animate(i):
      t = time.time()
      global curr_x, curr_y, prev_weights,decoded_x, decoded_y, pause 
      while not queue.empty() and not pause:
         posX,posY,done,restart = queue.get() 
         curr_x.append(posX)
         curr_y.append(posY)

         if done or restart:
            curr_x,curr_y=[],[]
         
         # if bool(decoded_x) == False or bool(decoded_y) == False:
         #    decoded_x.append(curr_x[-1])
         #    decoded_y.append(curr_y[-1])

         if i>1 and len(curr_x)>2:
            # '''encoding mangnitude and direction of movement'''
            # x1=curr_x[-2]
            # x2=curr_x[-1]

            # y1=curr_y[-2]
            # y2=curr_y[-1]
            
            # # delta[0]=np.sqrt(((x2-x1)**2)+((y2-y1)**2))                  #translation
            theta=np.rad2deg(math.atan2(curr_y[-1]-curr_y[-2],curr_x[-1]-curr_x[-2]) )          #angle
            delta[0]=curr_x[-1]
            delta[1]=curr_y[-1]

            '''updating network'''
            for j in range(len(delta)):
               net=attractorNetworkSettling(int(N.val),num_links[j],int(excite.val), activity_mag[j],inhibit_scale.val)
               prev_weights[j][:]= net.update_weights_dynamics(prev_weights[j][:],delta[j])
               prev_weights[j][prev_weights[j][:]<0]=0

               # if len(prev_weights[j][:]>0) == 0:
               #    prev_weights[j][net.activation(delta[j])]=net.full_weights(num_links[j])
            im=np.outer(prev_weights[1][:],prev_weights[0][:])
            
            '''plotting'''
            ax0.clear()   
            ax0.imshow(im, interpolation='nearest', aspect='auto')
            # ax0.axis('off')
            ax0.invert_yaxis()

            # axx.clear()
            # axx.set_title("Attractor Network")
            # axx.bar(np.arange(int(N.val)),prev_weights[0][:],width=0.8,color= '#C79FEF')
            # axx.axis('off')
            # axy.clear()
            # axy.barh(np.arange(int(N.val)),prev_weights[1][:],height=0.8,color= '#C79FEF')
            # axy.axis('off')

            ax1.clear()
            ax1.set_title('CarRacer Position')
            ax1.axis('equal')
            # ax0.add_patch(Rectangle((curr_x[-1]-2, curr_y[-1]-2), 8, 8,angle=theta,facecolor = '#929591'))
            ax1.scatter(curr_x, curr_y,s=10,c='r')
            


   def update(val):
      global prev_weights, num_links, activity_mag
      '''distributed weights with excitations and inhibitions'''
      # delta=[int(delta1.val),int(delta2.val)]
      prev_weights=[np.zeros(int(N.val)), np.zeros(int(N.val))]
      for j in range(len(delta)):
         net=attractorNetworkSettling(int(N.val),num_links[j],int(excite.val), activity_mag[j],inhibit_scale.val)
         prev_weights[j][:]= net.update_weights_dynamics(prev_weights[j][:],delta[j])
         prev_weights[j][prev_weights[j][:]<0]=0

   def onClick(event):
      global pause, prev_weights # resetDone
      (xm,ym),(xM,yM) = start_stop.label.clipbox.get_points()
      if xm < event.x < xM and ym < event.y < yM:
         pause ^= True


            # '''decoding mangnitude and direction of movement'''
            # trans=activityDecoding(prev_weights[0][:],num_links[0],N[0],neurons[0][:])#-prev_trans
            # angle=np.deg2rad(activityDecodingAngle(prev_weights[1][:],num_links[1],N[1],neurons[1][:]))

            # decoded_x.append(decoded_x[-1]+ (trans*np.cos(angle)))
            # decoded_y.append(decoded_y[-1]+ (trans*np.sin(angle)))

            # # decoded_x.append(np.argmax(prev_weights[0][:]))
            # # decoded_y.append(np.argmax(prev_weights[1][:]))


            # ax2.set_title("Decoded Attractor Network")
            # ax2.scatter(decoded_x,decoded_y, s=15)
            # ax1.axis('equal')
            
            
            
   '''animation for Place Cells'''
   excite.on_changed(update)
   # delta1.on_changed(update)
   N.on_changed(update)
   inhibit_scale.on_changed(update)
   fig.canvas.mpl_connect('button_press_event', onClick)
   ani = FuncAnimation(fig, animate, interval=1)
   plt.show() 
       
            
   
if __name__=="__main__":
    freeze_support()
    #conn1, conn2 = multiprocessing.Pipe()
    queue = multiprocessing.Queue()
    process_1 = multiprocessing.Process(target=driving_func, args=(queue,))
    process_2 = multiprocessing.Process(target=matplotlib_func, args=(queue,))
    process_1.start()
    process_2.start()
    process_1.join()
    process_2.join()


# def my_func(is_matplotlib):
#     if is_matplotlib:   
#         #matplotlib stuff    
#     else:
#         #carRacer stuff

# if __name__=="__main__":
#     freeze_support()

#     with Pool(processes=2) as pool:
#         values = [[True],[False]]
#         res = pool.starmap(my_func, values)
#         for r in res:
#             while True: pass

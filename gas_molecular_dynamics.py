import numpy as np
import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode

vmin = 0
vmax = 1  # used for defining the size of the display window


def norm(x):
    return np.sqrt((x**2).sum())


#### PARTICLES AND WALLS ####

R = 0.005  # radius of the particles


# The particles are represented through three vectors:
#   - ms : masses of the particles (N array)
#   - rs : position of the particles ((N x 2) array)
#   - vs : velocities of the particles ((N x 2) array)
def initialize(ms, rs=None, vs=None, v=0.0, initialization="only_one"):
    # INITIALIZES THE PARTICLES
    N = len(ms)
    if (rs is not None) and (vs is not None):
        # it is possible to pass the positions and velocities for the initialization
        # this is useful to simulate back in time from a given condition
        return rs, vs
    elif initialization == "only_one":
        # particles are organized in a grid
        # only the first particle has non-zero velocity
        interval = np.linspace(2 * R, 3 * R * np.sqrt(N), int(np.sqrt(N)))
        xs, ys = np.meshgrid(interval, interval)
        rs = np.column_stack((xs.flatten(), ys.flatten()))
        vs = np.zeros((N, 2))
        vs[0] = v * np.random.randn(2)
        return rs, vs
    elif initialization == "uniform":
        # particles are organized in a grid
        # potentially a fraction of the particles can have a different avg velocity
        interval = np.linspace(2 * R, 1 - 2 * R, int(np.sqrt(N)))
        xs, ys = np.meshgrid(interval, interval)
        rs = np.column_stack((xs.flatten(), ys.flatten()))
        vs = v * (np.random.randn(N, 2) - 0.5)
        vs[0 : N // 2] *= 2
        return rs, vs
    raise Exception("Initialization not recognized")


# The walls of the container (assumed to be square-shaped) are defined as objects
class Wall:
    # The wall is assumed to be long 1, and have a width specified by the user ('w')
    # Each wall has also:
    #   - a specific position 'r', the position of the centre of the rectangle that is the wall
    #   - orientation ("x" or "y")
    #   - mass 'M' (which can be infinity if the wall does not move)
    #   - and velocity 'v' (initialized to 0)
    def __init__(self, orientation, r, M, w):
        self.orientation = orientation
        self.r = r  # position
        self.v = 0.0  # velocity
        self.M = M  # mass
        self.w = w  # width

    def solve_collision(self, rs, vs, ms, i_next):
        # UPDATES THE VELOCITY OF A PARTICLE AFTER A COLLISION WITH THE WALL
        # and the velocity of the wall (if M < infinity). The collision is elastic
        M, orientation = self.M, self.orientation
        ind = 1 if (orientation == "x") else 0
        m = ms[i_next]
        if M < np.inf:
            # M < infinity -> elastic collision in the orthogonal direction to the wall
            dv_wall = 2 * m / (m + M) * vs[i_next, ind] - 2 * m / (M + m) * self.v
            vs[i_next, ind] = (
                2 * M / (m + M) * self.v + (m - M) / (M + m) * vs[i_next, ind]
            )
            self.v += dv_wall
        else:
            # M=infinity -> perfect reflection, the wall does not budge
            vs[i_next, ind] = -vs[i_next, ind]

    def evolve(self, delta_t):
        # COMPUTES THE NEW POSITION OF THE WALL (after delta_t)
        # assuming uniform motion with the velocity 'v' of the wall
        self.r += self.v * delta_t

    def plot(self):
        # DISPLAYS THE WALL (in black)
        orientation, r, w = self.orientation, self.r, self.w
        if orientation == "x":
            plt.plot(
                [0.0, 1.0, 1.0, 0.0],
                [r - w / 2, r - w / 2, r + w / 2, r + w / 2],
                c="k",
            )
        else:
            plt.plot(
                [r - w / 2, r - w / 2, r + w / 2, r + w / 2],
                [0.0, 1.0, 1.0, 0.0],
                c="k",
            )


#### FUNCTIONS for simulating the evolution of the system ####

dt = 0.01  # time step (for visualization)


def ballistic_evolution(rs, vs, delta_t):
    # UPDATES THE POSTION OF THE PARTICLES
    # assuming they move with uniform velocity for delta_t
    # be careful, this assumes no collision are taking place!
    rs += vs * delta_t


def solve_collision(rs, vs, ms, i, j):
    # UPDATES THE VELOCITY AFTER AN ELASTIC COLLISION (hard spheres approximation)

    # positions, velocities and masses of the particles involved in the collision
    r1, r2, v1, v2, m1, m2 = rs[i], rs[j], vs[i], vs[j], ms[i], ms[j]
    d_12 = norm(r1 - r2)  # distance btw particles
    l_o_c = (r2 - r1) / d_12  # line of centers
    v1_par = np.dot(v1, l_o_c)  # component of v1 parallel to the l_o_c
    v2_par = np.dot(v2, l_o_c)  # component of v2 parallel to the l_o_c

    # (general case)
    v1_par_new = (m1 - m2) / (m1 + m2) * v1_par + 2 * m2 / (m1 + m2) * v2_par
    v2_par_new = (m2 - m1) / (m1 + m2) * v2_par + 2 * m1 / (m1 + m2) * v1_par

    # (equal masses, (they exchange the parallel components of the velocity!))
    # v1_par_new = v2_par
    # v2_par_new = v1_par

    # remove old parallel components and add new ones
    v1 += (v1_par_new - v1_par) * l_o_c  # this changes v1 in place
    v2 += (v2_par_new - v2_par) * l_o_c  # this changes v2 in place


#### DETECTION OF THE NEXT COLLISION ####

# In order to know how long we can evolve the sistem with uniform motion for
# all the particles, we need to find when the next collision takes place


def next_collision_matrix(rs, vs, walls):
    # FULLY COMPUTES THE COLLISION MATRIX 'tnext'
    # When we start the simulation we have to find all the possible collision that
    # would take place between pairs of particles and
    # between particles and the walls of the container.
    # For each pair we RECORD THE TIME OF COLLISION in a matrix.
    # The time of collision for each pair is found by solving a simple second degree equation.
    # If no real solution is available,
    # or if the solution is negative (collision in the past)
    # we set the corresponding time of collision to infinity (meaning never happening).
    # The computational cost is O(N**2)

    N = rs.shape[0]  # number of particles
    n_walls = len(walls)  # number of walls

    # matrix containing the times of collision (to be computed, initialized to negative one)
    tnext = -np.ones((N, N + n_walls))

    # Coefficients of the second degree equation
    # a * t**2 + b*t + c = 0
    # we use vectors to solve all the collisions involving one particle "at the same time"
    a = np.zeros(N + n_walls)
    b = np.zeros(N + n_walls)
    c = np.zeros(N + n_walls)
    delta = np.zeros(N + n_walls)  # discriminant
    # place holder for the solutions of the collision for a single particle
    tm = -np.ones(N + n_walls)

    for i in range(N):
        # the function below updates row-by-row (particle by particle)
        # the matrix with the times of collision with all the other particles and walls
        update_tnext(rs, vs, walls, a, b, c, delta, tm, tnext, i)

    # FIND THE NEXT COLLISION THAT WILL TAKE PLACE (smallest time of collision)
    ind_next = np.argmin(tnext)  # linear index
    i_next, j_next = np.unravel_index(ind_next, tnext.shape)  # -> cartesian indices
    t_next_coll = tnext[i_next, j_next]  # time of NEXT collision

    # return everything, also the vectors for the coefficients, so we can reuse them
    return a, b, c, delta, tm, tnext, i_next, j_next, t_next_coll


def update_next_collision_matrix(
    rs, vs, walls, a, b, c, delta, tm, tnext, i_next, j_next, t_next_coll
):
    # UPDATES THE COLLISION MATRIX 'tnext'
    # assuming that a collision btw particle i_next and particle j_next just took place
    # Since only two particles (or a particle and a wall) changed velocity,
    # we don't have to recompute the entire matrix, but we can update only the
    # rows i_next and j_next of the matrix (the particles who changed their velocity
    # will now follow a different trajectory with different collisions),
    # and the columns i_next and j_next (all the other particles will now have to
    # recompute when they will collide with the particles that changed trajectory).
    # Limiting the update to these rows and columns reduces the computation to a
    # O(N) cost, instead of a O(N**2) cost

    N = rs.shape[0]

    # since we waited t_next_coll for the collision btw i_next and j_next,
    # we have to update all the times of collision, diminishing each by t_next_coll
    tnext -= t_next_coll

    # PARTICLE i_next UPDATES
    update_tnext(rs, vs, walls, a, b, c, delta, tm, tnext, i_next)  # row i_next
    tnext[:, i_next] = tnext[i_next, :N]  # column i_next

    # OBJECT j_next UPDATES
    # we distinguish btw the cases where j_next was a particle
    # or when j_next was wall
    if j_next < N:
        # particle-particle collision
        # update all the collision times involving the second particle j_next
        update_tnext(rs, vs, walls, a, b, c, delta, tm, tnext, j_next)  # row j_next
        tnext[:, j_next] = tnext[j_next, :N]  # column j_next
    elif walls[j_next - N].M < np.inf:
        # particle-wall collision, wall with finite mass (and moving)
        # update all the collision times with the moving wall
        update_tnext_wall(rs, vs, walls[j_next - N], tnext, j_next)  # column j_next

    # FIND THE NEXT COLLISION THAT WILL TAKE PLACE (smallest time of collision)
    ind_next = np.argmin(tnext)  # linear index
    i_next, j_next = np.unravel_index(ind_next, tnext.shape)  # -> cartesian indices
    t_next_coll = tnext[i_next, j_next]  # time of NEXT collision

    return i_next, j_next, t_next_coll


def update_tnext(rs, vs, walls, a, b, c, delta, tm, tnext, i):
    # UPDATES ALL THE TIME OF COLLISION FOR 1 PARTICLE (row)
    # solving the second degree eq:
    # || r1(t) - r2(t) ||^2 = (2R)**2 # particles collide at distance 2R!
    # Giving the 2nd degree eq:  a * t**2 + b * t + c = 0  , with:
    #   - a = (||v1 - v2||^2)
    #   - b =  2(v1 - v2)*(r1 - r2))
    #   - c = (||r1 - r2||^2 - (2R)**2)

    n = rs.shape[0]

    tm[:] = -1  # first reinitialize this array with -1, it will contain the roots

    # COMPUTE THE COEFFICIENTS of the second degree equation for the times of collision
    # first for the particle-particle collisions
    a[:n] = (vs[:, 0] - vs[i, 0]) ** 2 + (vs[:, 1] - vs[i, 1]) ** 2
    b[:n] = 2 * (
        (rs[:, 0] - rs[i, 0]) * (vs[:, 0] - vs[i, 0])
        + (rs[:, 1] - rs[i, 1]) * (vs[:, 1] - vs[i, 1])
    )
    c[:n] = (rs[:, 0] - rs[i, 0]) ** 2 + (rs[:, 1] - rs[i, 1]) ** 2 - (2 * R) ** 2
    for k in range(len(walls)):
        # then for the particle-wall collisions
        wall = walls[k]
        if wall.orientation == "x":
            a[n + k] = (vs[i, 1] - wall.v) ** 2
            b[n + k] = 2 * ((rs[i, 1] - wall.r) * (vs[i, 1] - wall.v))
            c[n + k] = (rs[i, 1] - wall.r) ** 2 - (R + wall.w) ** 2
        else:
            a[n + k] = (vs[i, 0] - wall.v) ** 2
            b[n + k] = 2 * ((rs[i, 0] - wall.r) * (vs[i, 0] - wall.v))
            c[n + k] = (rs[i, 0] - wall.r) ** 2 - (R + wall.w) ** 2

    # given the coefficients compute the discriminants
    delta[:] = b**2 - 4 * a * c
    msk = delta > 0  # find the indices where the discriminant is non-negative
    # in these indices compute the new roots (time of collisions)
    tm[msk] = (-b[msk] - np.sqrt(delta[msk])) / (2 * a[msk])

    # If the predicted time of collision is negative (in the past),
    # set it to infinity (no collision will happen in the future!)s
    tm[tm <= 0] = np.inf

    tnext[i, :] = tm  # finally, update the row of the 'tnext' matrix


def update_tnext_wall(rs, vs, wall, tnext, j_next):
    # SAME AS THE ABOVE FUNCTION, but for a wall update
    n = rs.shape[0]
    for i in range(n):
        a, b, c = 0.0, 0.0, 0.0
        if wall.orientation == "x":
            a = (vs[i, 1] - wall.v) ** 2
            b = 2 * ((rs[i, 1] - wall.r) * (vs[i, 1] - wall.v))
            c = (rs[i, 1] - wall.r) ** 2 - (R + wall.w) ** 2
        else:
            a = (vs[i, 0] - wall.v) ** 2
            b = 2 * ((rs[i, 0] - wall.r) * (vs[i, 0] - wall.v))
            c = (rs[i, 0] - wall.r) ** 2 - (R + wall.w) ** 2
        delta = b**2 - 4 * a * c
        if delta < 0:
            # no collision
            tnext[i, j_next] = np.inf
            continue
        tm = (-b - np.sqrt(delta)) / (2 * a)
        tnext[i, j_next] = tm if tm > 0 else np.inf


#### EVOLUTION OF THE SYSTEM ####


def trajectory(
    ms,
    seed=None,
    rs=None,
    vs=None,
    v=5.0,
    initialization="only_one",
    T=100000,
    plot_time=10,
):
    N = len(ms)

    # seed to fix the randomness in the simulation (initialization)
    if seed is not None:
        np.random.seed(seed)

    # initialize the particles...
    rs, vs = initialize(ms, rs=rs, vs=vs, v=v, initialization=initialization)
    # ...and the walls of the container
    walls = [
        Wall("x", 0.0, np.inf, 0.0),
        Wall("x", 1.0, np.inf, 0.0),
        Wall("y", 0.0, np.inf, 0.0),
        Wall("y", 1.0, np.inf, 0.0),
    ]

    # UNCOMMENT TO ADD A MOVING WALL in the box with params (orientation, position, mass, width)
    # walls.append(Wall("y", 0.5, 100, 0.01))

    a, b, c, delta, tm, tnext, i_next, j_next, t_next_coll = next_collision_matrix(
        rs, vs, walls
    )

    # UNCOMMENT TO STORE THE POSITION OF THE MOVING WALL (if any)
    # wall_positions = [walls[-1].r]

    for t in range(T):
        # We want to display the system at regular time intervals (multiple of 'dt'),
        # but many collisions might take place before the next display.
        # Therefore, we keep finding the next collision, and computing all the updates,
        # until we find that the next collision happens after 'dt' step.
        elapsed = 0.0  # time passed since the last "epoch", i.e. a time step 'dt'
        while (elapsed + t_next_coll) < dt:
            elapsed += t_next_coll
            # EVOLVE UNTIL THE NEXT COLLISION
            ballistic_evolution(rs, vs, t_next_coll)  # both the particles...
            for wall in walls:  # ...and the walls (if they are moving)
                wall.evolve(t_next_coll)
            # COMPUTE THE NEW VELOCITIES AFTER THE COLLISION
            if j_next < N:
                # particle-particle case
                solve_collision(rs, vs, ms, i_next, j_next)
            elif j_next >= N:
                # particle-wall case
                walls[j_next - N].solve_collision(rs, vs, ms, i_next)
            # FIND THE NEXT COLLISION
            i_next, j_next, t_next_coll = update_next_collision_matrix(
                rs, vs, walls, a, b, c, delta, tm, tnext, i_next, j_next, t_next_coll
            )

        # When no other collisions will take place within our 'dt' window, we
        # evolve for the remaining time, potentially display or print our metrics,
        # and move to the next epoch
        t_remaining = dt - elapsed
        # Evolve...
        ballistic_evolution(rs, vs, t_remaining)  # both the particles...
        for wall in walls:  # ...and the walls
            wall.evolve(t_remaining)
        tnext -= t_remaining  # remove the elapsed time from the matrix with the
        # times until the next collisions...
        t_next_coll -= t_remaining  # ...and from the very next collision time

        if t % plot_time == 0:
            ## Find which particles are left and which are right of the last wall
            ## in our wall list (useful when there is a moving wall)
            left = rs[:, 0] < walls[-1].r

            ## UNCOMMENT TO PLOT TRAJECTORIES
            ## (Also uncomment line 395 where figure f1 is created)
            plt.pause(0.001)  # force the plot immediately
            plt.figure(f1)
            plt.clf()
            for w in walls:
                w.plot()
            ## the code below is to find what point size will be close enough to the
            ## radius size we chose for the gas particles
            s = (
                (f1.get_window_extent().width / (vmax - vmin + 0.1) * 110.0 / f1.dpi)
                ** 2
            ) * R**2
            ## If there is a wall in the middle of the box, the particles to the left
            ## and to the right have different display colors
            plt.scatter(rs[left, 0], rs[left, 1], s=s, c="orange")
            plt.scatter(rs[~left, 0], rs[~left, 1], s=s, c="blue")
            plt.text(0.9, 1.1, f"t={np.round(t * dt, 3)}")
            plt.show()

            ## UNCOMMENT TO PLOT HISTOGRAM of (x)-velocities
            ## (Also uncomment line 396 where figure f2 is created)
            ## Useful for the "only-one" experiment
            # plt.pause(0.0001)
            # plt.figure(f2)
            # plt.clf()
            # # If there is a wall in the middle of the box, the particles to the left
            # # and to the right have different display colors
            # plt.hist(vs[left,0], bins=30, alpha=0.3)
            # plt.hist(vs[~left,0], bins=30, alpha=0.3)
            # plt.xlabel("v_x"); plt.ylabel("count")
            # plt.show()

            # UNCOMMENT TO PLOT POSITION OF MOVING WALL (if any)
            # # (Also uncomment lines 296, 301 and 394)
            # wall_positions.append(walls[-1].r)
            # plt.pause(0.0001)
            # plt.figure(f2)
            # plt.clf()
            # plt.plot(dt*np.arange(len(wall_positions)), wall_positions)
            # plt.xlabel("time"); plt.ylabel("wall position")
            # plt.show()

            ## UNCOMMENT TO PRINT THE "FORCE" ON THE TWO SIDES OF THE MOVING WALL (if any)
            ## (Also uncomment line 300)
            # K_l = 0.5*np.sum(ms[left]*np.sum(vs[left]**2, axis=1))
            # K_r = 0.5*np.sum(ms[~left]*np.sum(vs[~left]**2, axis=1))
            # print(f"t={t} K_l/A_l = {K_l/(walls[-1].r-walls[-1].w)} K_r/A_r = {K_r/(1-(walls[-1].r+walls[-1].w))}")

    plt.show(
        block=True
    )  # when the loop is finished, we show the plot one last time in a blocking way to keep the window open
    return rs, vs


plt.close("all")

f1 = plt.figure(figsize=(6.0, 8), dpi=120)
f2 = plt.figure(figsize=(8.0, 4))

## EXPERIMENT 1:
## 1) Initialize only one moving particle (intialization="only_one")
## 2) look at the velocity histogram
## 3) See how the velocity distribution approaches Maxwell's distribution

## EXPERIMENT 2:
## 1) Initialize only one moving particle (intialization="only_one")
## 2) look at the trajectories
## 3) evolve for a few time steps
## 4) then evolve back in time (change sing to the velocity)
## 5) Observe what happens if the simuation time is too long!

## EXPERIMENT 3:
## 1) Initialize uniformly the particles (intialization="uniform")
## 2) add the moving wall (choose the mass, but large w.r.t. the particles)
## 3a) look what happens when the velocities and masses of the particles to the right and the left are the same
## 3b) look what happens when the velocities of the particles to the right and the left are different
## 3c) Look what happens if the masses are different
## 4) You can track the position of the wall in time and plot only that (quicker)!
## 5) What happens after along simulation time? Why? What happens if the wall is more massive?

## EXPERIMENT 4:
## Use the setup above to "verify" the law of perfect gases!

## EXPERIMENT 5:
## Track the position of a single particle (plot it with a different color!)
## Is it spending similar time in all possible regions of the container?

## EXAMPLE CODE
N = 20 * 20  # too many particles and it will slow down a lot!
ms = np.ones(N)  # choose the mass of the particles (1 is fine for starters!)

rs, vs = trajectory(ms, T=2000, plot_time=2, v=5.0, seed=3234, initialization="uniform")
# backwards in time
# rs, vs = trajectory(ms, T=200+1, plot_time=2, rs=rs, vs=-vs)

"""Closed-loop active localization experiment.

Provenance: code sent by Alisa Slipko (previous RA), stored verbatim on
2026-09-23. NOT yet run, and NOT yet integrated with the rest of the repo.

Known gaps to resolve before running:
  - The snippet as received is truncated: `report(...)` has only its signature.
  - Module-level names are undefined here and must be supplied/wired up:
    `np`, `stimulus_gen`, `spatializer`, `brainstem`, `decoder`.
  - `decoder.predict_one` and `result.ild_neural_tst` need to be checked
    against the current `lso/decoder.py` and `lso/brainstem.py`.

See OVERVIEW.md for the closed-loop process description.
"""


def circular_diff(a, b):
    return ((a - b + 180) % 360) - 180


def circular_stats(alphas_deg):
    alphas_deg = np.asarray(alphas_deg, dtype=float)
    n = len(alphas_deg)

    rad = np.deg2rad(alphas_deg)
    x_bar = np.mean(np.cos(rad))
    y_bar = np.mean(np.sin(rad))
    r = np.sqrt(x_bar**2 + y_bar**2)


    alpha_bar = np.rad2deg(np.arctan2(y_bar, x_bar)) % 360

    oriented_mask = np.abs(alphas_deg) < 90
    n_oriented = int(np.sum(oriented_mask))

    if n_oriented > 0:
        beta = float(np.median(np.abs(alphas_deg[oriented_mask])))
    else:
        beta = np.nan

    return {
        "n": n,
        "r": r,
        "alpha_bar": alpha_bar,
        "beta": beta,
    }



def run_single_trial(source_angle, seed_base,
                      circle_step=1.25, n_circles=5,
                      distance_tolerance_factor=0.3):

    circle_radii = circle_step * np.arange(1, n_circles + 1)
    source_radius = circle_radii[-1]
    source_position = source_radius * np.array([
        np.cos(np.deg2rad(source_angle)),
        np.sin(np.deg2rad(source_angle))
    ])

    mouse_position = np.array([0.0, 0.0])
    head_angle = 0.0

    alphas = []
    path_length = 0.0
    reached = False

    for step in range(n_circles):
        dx = source_position[0] - mouse_position[0]
        dy = source_position[1] - mouse_position[1]
        true_world_angle = np.rad2deg(np.arctan2(dy, dx)) % 360
        true_relative_angle = (true_world_angle - head_angle) % 360

        noise = stimulus_gen.generate_noise(seed=seed_base + step)
        left, right = spatializer.spatialize(noise, true_relative_angle)
        result = brainstem.run(left, right)
        ild = np.asarray(result.ild_neural_tst, dtype=float)
        angle_pred = decoder.predict_one(ild)

        predicted_world_angle = (head_angle + angle_pred) % 360

        alpha = circular_diff(predicted_world_angle, true_world_angle)
        alphas.append(alpha)

        head_angle = predicted_world_angle
        target_radius = circle_radii[step]
        theta = np.deg2rad(predicted_world_angle)
        direction = np.array([np.cos(theta), np.sin(theta)])

        a = 1.0
        b = 2.0 * np.dot(mouse_position, direction)
        c = np.dot(mouse_position, mouse_position) - target_radius**2
        disc = b**2 - 4*a*c
        if disc < 0:
            t = abs(target_radius - np.linalg.norm(mouse_position))
        else:
            t = (-b + np.sqrt(disc)) / (2*a)
            if t < 0:
                t = (-b - np.sqrt(disc)) / (2*a)

        old_pos = mouse_position.copy()
        mouse_position = mouse_position + t * direction
        path_length += np.linalg.norm(mouse_position - old_pos)

        distance_to_source = np.linalg.norm(source_position - mouse_position)
        if distance_to_source < distance_tolerance_factor * circle_step:
            reached = True
            break

    relative_length = path_length / source_radius
    final_distance = np.linalg.norm(source_position - mouse_position)

    return alphas, relative_length, final_distance, reached


def run_experiment(n_trials=200, n_circles=5, circle_step=1.25,
                    source_angles=None, seed_offset=0):
    rng = np.random.default_rng(42)

    alphas_by_circle = [[] for _ in range(n_circles)]
    relative_lengths = []
    final_distances = []
    reached_flags = []

    for trial in range(n_trials):
        if source_angles is None:
            s_angle = float(rng.uniform(0, 360))
        else:
            s_angle = float(rng.choice(source_angles))

        seed_base = 1000 + seed_offset + trial * 100

        alphas, rel_len, final_dist, reached = run_single_trial(
            s_angle, seed_base, circle_step=circle_step, n_circles=n_circles
        )

        for i, a in enumerate(alphas):
            alphas_by_circle[i].append(a)

        relative_lengths.append(rel_len)
        final_distances.append(final_dist)
        reached_flags.append(reached)

    return alphas_by_circle, relative_lengths, final_distances, reached_flags


def report(n_trials=200, n_circles=5, circle_step=1.25, source_angles=None):
    alphas_by_circle, rel_lengths, final_dists, reached_flags = run_experiment(
        n_trials=n_trials, n_circles=n_circles,
        circle_step=circle_step, source_angles=source_angles
    )

    print(f"\n{'='*70}")
    print(f"ACTIVE LOCALIZATION. METRICS OVER {n_trials} TRIALS")
    print(f"{'='*70}\n")

    print(f"{'d.n':<6}{'n':<6}{'r':<8}{'alpha_bar':<12}{'beta':<8}")
    print("-" * 60)

    pooled_alphas = []
    for i, alphas in enumerate(alphas_by_circle):
        stats = circular_stats(alphas)
        pooled_alphas.extend(alphas)
        print(f"d.{i+1:<4}"
              f"{stats['n']:<6}"
              f"{stats['r']:.3f}   "
              f"{stats['alpha_bar']:6.2f}°     "
              f"{stats['beta']:5.2f}°  ")

    g_stats = circular_stats(pooled_alphas)
    print("-" * 60)
    print(f"d.G   "
          f"{g_stats['n']:<6}"
          f"{g_stats['r']:.3f}   "
          f"{g_stats['alpha_bar']:6.2f}°     "
          f"{g_stats['beta']:5.2f}°  ")

    return alphas_by_circle, g_stats

report(n_trials=50, n_circles=5, circle_step=1.25, source_angles=None)
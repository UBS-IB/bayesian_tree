from typing import Generator

import numpy as np
from scipy.special import betaln, gammaln


def multivariate_betaln(alphas):
    if len(alphas) == 2:
        return betaln(alphas[0], alphas[1])
    else:
        # see https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function
        return np.sum([gammaln(alpha) for alpha in alphas], axis=0) - gammaln(np.sum(alphas))


def r2_series_generator(n_dim: int) -> Generator[np.ndarray, None, None]:
    """
    Computes R2 pseudo-random sequence, see
    http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

    :param n_dim: The number of dimensions of the output
    :return: R2 series data points
    """

    if n_dim == 0:
        raise ValueError(f'n_dim must be > 0 but was {n_dim}')

    # compute phi
    phi = 2
    phi_old = phi
    while True:
        phi = pow(1+phi, 1/(n_dim+1))
        if phi == phi_old:
            break

        phi_old = phi

    # compute alpha array
    alpha = 1/phi**(1+np.arange(n_dim))

    # compute R2 sequence
    i = 0
    while True:
        yield (0.5 + alpha * (i+1)) % 1
        i += 1


def hypercube_to_hypersphere_surface(
        hypercube_points: np.ndarray,
        half_hypersphere: bool) -> np.ndarray:
    """
    Converts uniformly distributed points from a D-dimensional hypercube, [0, 1]^D,
    to uniformly distributed points on the the D-dimensional surface of a hyperplane
    (embedded in (D+1)-dimensional space), see algorithm 'YPHL' in
    https://core.ac.uk/download/pdf/82404670.pdf with 'n' = D and 'd' = 0 (specifying
    the hypersphere surface rather than the volume)

    :param hypercube_points: A 2-dimensional array of shape N * D
    :param half_hypersphere: If True then map the uniform points to the half-hypersphere;
        if False then map to the full hypersphere
    :return:
    """

    assert 1 <= hypercube_points.ndim <= 2
    assert np.all(hypercube_points >= 0)
    assert np.all(hypercube_points <= 1)

    n_dim_surface = hypercube_points.shape[-1]
    n_dim_embedding = 1+n_dim_surface
    if hypercube_points.ndim == 1:
        hypercube_points = hypercube_points.reshape(1, -1)
        n_points = 1
    else:
        n_points = hypercube_points.shape[0]

    surface_points = np.zeros((n_dim_embedding, n_points))

    hypercube_points = hypercube_points.T  # easier if 1st index is the dimension

    if n_dim_embedding % 2 == 0:
        # even
        phi = np.pi * (hypercube_points[0] - 0.5) if half_hypersphere else 2 * np.pi * hypercube_points[0]
        surface_points[0] = np.cos(phi)
        surface_points[1] = np.sin(phi)

        for i in range(1, n_dim_embedding//2):
            u = hypercube_points[2*i-1]
            h = u ** (1/(2*i))
            surface_points[:2*i] *= h

            sqrt_rho = np.sqrt(np.maximum(0, 1-np.sum(surface_points[:2*i]**2, axis=0)))
            phi = 2*np.pi * hypercube_points[2*i]
            surface_points[2*i] = sqrt_rho*np.cos(phi)
            surface_points[2*i+1] = sqrt_rho*np.sin(phi)
    else:
        # odd
        if half_hypersphere:
            surface_points[0] = 1
            next_dim = 1
        else:
            # see https://mathworld.wolfram.com/SpherePointPicking.html
            assert n_dim_embedding >= 3

            phi = np.arccos(2 * hypercube_points[0] - 1)
            theta = 2 * np.pi * hypercube_points[1]
            surface_points[0] = np.sin(phi) * np.cos(theta)
            surface_points[1] = np.sin(phi) * np.sin(theta)
            surface_points[2] = np.cos(phi)
            next_dim = 2

            # # **old algorithm, flawed**
            # # in theory x[0] should be the random sign (+/- 1) which would require another
            # # random number, but we don't have that available, so generate pseudo-random
            # # bits from two sources: the data itself (even/odd bit count) and a bit from
            # # a deterministic quasi-random sequence
            # pseudo_random_bits_data = 1 * np.array([np.sum(list(struct.pack('!d', value))) % 2 == 0 for value in hypercube_points.flatten()])
            # pseudo_random_bits_data = pseudo_random_bits_data.reshape(hypercube_points.shape)
            # pseudo_random_bits_data = np.sum(pseudo_random_bits_data, axis=0) % 2 == 0
            #
            # r2gen = r2_series_generator(n_dim=1)
            # pseudo_random_bits_gen = np.array([next(r2gen)[0] > 0.5 for i in range(hypercube_points.shape[1])])
            #
            # pseudo_random_bits = pseudo_random_bits_data ^ pseudo_random_bits_gen
            # surface_points[0] = 2*pseudo_random_bits-1
            # next_dim = 1

        for i in range(next_dim, (n_dim_embedding + 1) // 2):
            u = hypercube_points[2 * i - 2]
            h = u ** (1 / (2 * i - 1))
            surface_points[:2 * i - 1] *= h

            sqrt_rho = np.sqrt(np.maximum(0, 1 - np.sum(surface_points[:2 * i - 1] ** 2, axis=0)))
            phi = 2 * np.pi * hypercube_points[2 * i - 1]
            surface_points[2 * i - 1] = sqrt_rho * np.cos(phi)
            surface_points[2 * i] = sqrt_rho * np.sin(phi)

    surface_points = surface_points.squeeze().T
    surface_points = (surface_points.T / np.linalg.norm(surface_points, axis=-1)).T  # correct numerical round-off errors

    return surface_points

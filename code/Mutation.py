__author__ = 'Sander van Rijn <svr003@gmail.com>'


def normalDistribution(x, C):
    """
    Generate a vector sampled according to the normal distribution with the given parameters

    :param x:
    :param C:
    :return:
    """

    pass


def x1(x, sigma, z):
    """
    Mutation 1: x = x + sigma*z

    :param x:
    :param sigma:
    :param z:
    :return:
    """

    new_x = x + (sigma*z)
    return new_x


def adaptSigma(sigma, p_s, c=0.817):
    """
    Adapt parameter sigma based on the 1/5th success rule

    :param sigma:
    :param p_s:
    :param c:
    :return:
    """

    if p_s < 1/5:
        sigma *= c
    elif p_s > 1/5:
        sigma /= c

    return sigma


def calculateP_S(num_successes, num_failures):
    """
    Calculate the recent success rate

    :param num_successes:
    :param num_failures:
    :return:
    """

    return num_successes / (num_failures + num_successes)


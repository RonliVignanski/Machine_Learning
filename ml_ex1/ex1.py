import matplotlib.pyplot as plt
import numpy as np
import sys


def distance(pix, cent):
    """
    get distance between the pixel and the centroid.
    :param pix: each pixel in the picture
    :param cent: the centroid we got and then the new one which updated
    :return: the distance between them- norma
    """

    return np.linalg.norm(pix - cent, ord=2) ** 2


def loss(pi, cents):
    """
    get the avg loss
    :param pi: pixel
    :param cents: centroid
    :return: the avg loss per iteration
    """
    s = 0
    for p in pi:
        min_dist = distance(p, cents[0])
        for cent in cents:
            dis = distance(p, cent)
            if dis < min_dist:
                min_dist = dis
        s = s + min_dist
    return s / len(pi)
 
 
if __name__ == '__main__':

    image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
    z = np.loadtxt(centroids_fname)  # load centroids

    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255.
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    pixels = pixels.reshape(-1, 3)

    cent_for_loss = []
    list_of_loss = []

    # create a dictionary for the indexes of the centroids to the list of pixels which belong to them
    centroids = dict()
    for index in range(len(z)):
        centroids[index] = []

    for i in range(0, len(z)):
        list_of_cents = []

    f = open(out_fname, "w")
    # run 20 times or till convergence
    for epoch in range(20):
        old_z = z.copy()
        for pixel in pixels:
            minimum = float('inf')
            index = None
            # find the closest index of centroid
            for i, centroid in enumerate(z):
                dist = distance(pixel, centroid)
                if dist < minimum:
                    minimum = dist
                    index = i

            # put the pixel in the dictionary in the right place of the min_centroid
            centroids[index].append(pixel)

        # getting the avg
        for key in centroids.keys():
            if len(centroids[key]) != 0:
                # update the centroid
                z[key] = np.mean(centroids[key], axis=0).round(4)
            centroids[key] = []

        # for the plots
        # s = loss(pixels, z)
        # list_of_loss.append(s)

        # writing to the file
        f.write(f"[iter {epoch}]:{','.join([str(i) for i in z])}\n")

        # check when there is a convergence
        if (old_z == z).min():
            break
    f.close()
    
    # for showing the needed plots
    # x = list(range(len(list_of_loss)))
    # plt.plot(x, list_of_loss, c='red')
    # plt.xticks(x, x)
    # plt.xlabel('Iteration')
    # plt.ylabel('Avg. loss')
    # plt.title('K = 2')
    # plt.show()

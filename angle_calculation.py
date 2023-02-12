#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
a = np.array([10,10])
b = np.array([15,15])
c = np.array([20,10])


#%%
plt.figure()

ax = plt.gca()
ax.set_xlim([0, 25])
ax.set_ylim([0, 25])

for x1, y1 in [a,b,c]:
    plt.plot(x1, y1, 's')

# plt.plot(10, 10, 's')
# plt.plot(l2)

# plt.plot(X, y, 's')
# plt.plot(X, line_of_best_fit.predict(X), color='k')

plt.show()


#%%
def compute_vector(point_one, point_two):
    x1, y1 = point_one
    x2, y2 = point_two

    return np.array([x2 - x1, y2 - y1])

def compute_angle(v1, v2):
    dot = np.dot(v1, v2)

    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)

    cos_angle = dot / (v1_magnitude * v2_magnitude)

    angle_in_rad = np.arccos(cos_angle)

    angle_in_deg = np.degrees(angle_in_rad)

    return angle_in_deg
    

v1 = compute_vector(a,b)
v2 = compute_vector(b,c)
print(compute_angle(v1, v2))

#%%

def squared_angle_sum(coords):
    """Squared angle sum.

    For each triple of points (a,b,c),
    compute angle between (a,b) and (b,c).
    Then, square and sum all of these.

    Parameters
    ----------
    coords: array
        A numpy array containing the (t, x, y) coordinates of the track.

    Returns
    -------
    float
        The feature value for the entire array.

    """
    coords = coords[:, 1:]
    angle_values = []

    for i in range(len(coords) - 2):
        a = coords[i]
        b = coords[i + 1]
        c = coords[i + 2]

        v1 = compute_vector(a, b)
        v2 = compute_vector(b, c)

        angle = compute_angle(v1, v2)
        angle_values.append(angle)

    angle_values = np.array(angle_values)
    squared_angle_values = np.square(angle_values) # We do this so that everything is positive

    return np.sum(squared_angle_values)


data = np.array([[1,10,10], [2,15,15], [3,20,20]])
print(squared_angle_sum(data))

#%%
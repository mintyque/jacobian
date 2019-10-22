import numpy as np


def make_o(t):
    """
    Function to compute o-components of Jacobian column

    Oi corresponds to first three elements of the fourth column of Ti

    Parameter
    ---------
    t
        transformation matrix from base to a joint

    Returns
    -------
    List of three needed values
    """
    o = list([])
    o.append(t[0][3])
    o.append(t[1][3])
    o.append(t[2][3])
    return o


def make_z(t):
    """
    Function to compute z-components of Jacobian column

    Zi corresponds to first three elements of the third column of Ti. Those elements are chosen iff the joint
    changes end-effector's position towards Z-axis.

    Parameter
    ---------
    t
        transformation matrix from base to a joint

    Returns
    -------
    List of three needed values
    """
    z = list([])
    z.append(t[0][2])
    z.append(t[1][2])
    z.append(t[2][2])
    return z


def make_x(t):
    """
    Function to compute x-components of Jacobian column

    Zi corresponds to first three elements of the first column of Ti. Those elements are chosen iff the joint
    changes end-effector's position towards X-axis.

    Parameter
    ---------
    t
        transformation matrix from base to a joint

    Returns
    -------
    List of three needed values
    """
    x = list([])
    x.append(t[0][0])
    x.append(t[1][0])
    x.append(t[2][0])
    return x


def j_column_skew(o6, oi, r):
    """
    Function for computing a column of Jacobian matrix using skew theory.

    The matrix is computed as follows:
    J = [J1 J2 ... J6]

    And the i-th row is computed as follows:
    J[i] = [ r[i-1] * (O[n] - O[i-1]) ]
           [           r[i-1]         ]
    Where r[i] is the first three elements of the column which corresponds to the direction of the rotation.
    In our manipulator, J1 rotates towards X-axis, J2, J3 and J5 rotate towards Z-axis,
    J4 and J6 don't rotate towards any axis, so it can be any.
    O[i] is the first three elements of the fourth column in T[i] matrix.

    Parameters
    ----------
    o6
        transformation matrix from base to J6
    oi
        transformation matrix from base to Ji
    r
        elements of transformation matrix from base to previous joint

    Returns
    -------
    Column of a Jacobian matrix
    """
    j = np.empty((0, 3))
    j = np.append(j, np.cross(r, np.subtract(o6, oi)))
    j = np.append(j, r)
    return j


def j_skew(q):
    """
    Function for computing Jacobian matrix from a given array of angles for each joint using skew theory approach.

    Resulting Jacobian matrix is constructed row by row using j_column_skew defined above.

    To construct a Jacobian we need to know transformation matrices for all joints, together with transformation
    matrices from base point to every joint. They are all defined below, from the base point (T[0]) to the
    end-effector (T[6]). Translation portions for T[4] and T[6] are equal to 0 as there is no translation.
    Transformation matrices from base to i-th joint are shown as T[0-i]

    T[0] = [1, 0, 0, 0]
           [0, 1, 0, 0]
           [0, 0, 1, 0]
           [0, 0, 0, 1]
    T[1] = [cos(J1), -sin(J1), 0,  0 ]
           [sin(J1),  cos(J1), 0, 312]
           [  0,       0,      1, 670]
           [  0,       0,      0,  1 ]
    T[2] = [1,    0,        0,      0 ]
           [0, cos(J2), -sin(J2),   0 ]
           [0, sin(J2),  cos(J2), 1075]
           [0,    0,        0,      1 ]
    T[3] = [1,    0,        0,      0 ]
           [0, cos(J3), -sin(J3), 1280]
           [0, sin(J3),  cos(J3),  275]
           [0,    0,        0,      1 ]
    T[4] = [ cos(J4), 0, sin(J4), 0]
           [    0,    1,    0,    0]
           [-sin(J4), 0, cos(J4), 0]
           [    0,    0,    0,    1]
    T[5] = [1,    0,        0,      0 ]
           [0, cos(J5), -sin(J5),   0 ]
           [0, sin(J5),  cos(J5), 1075]
           [0,    0,        0,      1 ]
    T[6] = [ cos(J6), 0, sin(J6), 0]
           [    0,    1,    0,    0]
           [-sin(J6), 0, cos(J6), 0]
           [    0,    0,    0,    1]

    T[0-1] = T[0] * T[1]
    T[0-2] = T[0-1] * T[2]
    T[0-3] = T[0-2] * T[3]
    T[0-4] = T[0-3] * T[4]
    T[0-5] = T[0-4] * T[5]
    T[0-6] = T[0-5] * T[6]

    Parameters
    ----------
    q
        list of angles of all joints, where j[0] - angle of first joint, j[5] - angle of sixth joint

    Returns
    -------
    j
        Jacobian matrix for the manipulator
    """
    # construct all transformation matrices for each joint
    t0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    t1 = np.array([[np.cos(q[0]), -np.sin(q[0]), 0, 0],
                   [np.sin(q[0]), np.cos(q[0]), 0, 312],
                   [0, 0, 1, 670],
                   [0, 0, 0, 1]])
    t2 = np.array([[1, 0, 0, 0],
                   [0, np.cos(q[1]), -np.sin(q[1]), 0],
                   [0, np.sin(q[1]), np.cos(q[1]), 1075],
                   [0, 0, 0, 1]])
    t3 = np.array([[1, 0, 0, 0],
                   [0, np.cos(q[2]), -np.sin(q[2]), 1280],
                   [0, np.sin(q[2]), np.cos(q[2]), 275],
                   [0, 0, 0, 1]])
    t4 = np.array([[np.cos(q[3]), 0, np.sin(q[3]), 0],
                   [0, 1, 0, 0],
                   [-np.sin(q[3]), 0, np.cos(q[3]), 0],
                   [0, 0, 0, 1]])
    t5 = np.array([[1, 0, 0, 0],
                   [0, np.cos(q[4]), -np.sin(q[4]), 0],
                   [0, np.sin(q[4]), np.cos(q[4]), 1075],
                   [0, 0, 0, 1]])
    t6 = np.array([[np.cos(q[5]), 0, np.sin(q[5]), 0],
                   [0, 1, 0, 0],
                   [-np.sin(q[5]), 0, np.cos(q[5]), 0],
                   [0, 0, 0, 1]])
    # construct all transformation matrices from base to each joint
    t00 = np.dot(t0, t0)
    t01 = np.dot(t0, t1)
    t02 = np.dot(t01, t2)
    t03 = np.dot(t02, t3)
    t04 = np.dot(t03, t4)
    t05 = np.dot(t04, t5)
    t06 = np.dot(t05, t6)
    # construct all O elements to calculate Jacobian
    o0 = make_o(t00)
    o1 = make_o(t01)
    o2 = make_o(t02)
    o3 = make_o(t03)
    o4 = make_o(t04)
    o5 = make_o(t05)
    o6 = make_o(t06)
    # construct all Z elements to calculate Jacobian
    x0 = make_x(t00)
    x1 = make_x(t01)
    z2 = make_z(t02)
    z3 = make_z(t03)
    z4 = make_z(t04)
    z5 = make_z(t05)
    # calculate the Jacobian column by column
    j1 = j_column_skew(o6, o0, x0)
    j2 = j_column_skew(o6, o1, x1)
    j3 = j_column_skew(o6, o2, z2)
    j4 = j_column_skew(o6, o3, z3)
    j5 = j_column_skew(o6, o4, z4)
    j6 = j_column_skew(o6, o5, z5)
    # construct the final Jacobian matrix
    j = np.vstack((j1, j2, j3, j4, j5, j6))
    return j


q = [1, 1, 1, 1, 1, 1]
j = j_skew(q)
for r in range(6):
    for c in range(6):
        print(j[r][c], end=' ')
    print("")

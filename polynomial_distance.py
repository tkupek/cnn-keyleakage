
import scipy.spatial.distance as distance

if __name__ == "__main__":
    p0 = [2, 3, 0]
    p1 = [0.1, -1, 0]
    p2 = [2, 4, 0]
    p3 = [1, -2, -3]
    p4 = [0.5, 2, -2]
    p5 = [8, -20, -7]
    p6 = [7, -1, -1]

    polynomials = [p0, p1, p2, p3, p4, p5, p6]

    matrix = []
    for p in polynomials:
        line = []
        for p_temp in polynomials:
            line.append(distance.euclidean(p, p_temp))
        matrix.append(line)

    print('polynomial distance matrix: ')
    for line in matrix:
        for row in line:
            print(str(row), end = ',')
        print()
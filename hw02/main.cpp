#include <cmath>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <limits>
#include <ctime>

using namespace std;

struct Point {
    int id;
    double x;
    double y;
};

namespace ProjectInfo {
    const string COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI";
    const string COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS";
    const string ASSIGNMENT = "Homework Assignment 2";
    const string AUTHOR = "Jovan Mosurovic";
    const string INDEX = "2022/0589";
    const string SEPARATOR = "==============================================";
    const string SEPARATOR_SMALL = "----------------------------------------------";
    const string PROBLEM = "TSP PROBLEM - PCB DRILLING PATH OPTIMIZATION";
}

// Function for calculating L1 norm (Manhattan distance) between two points
double distanceBetweenTwoPoints(const Point &p1, const Point &p2) {
    return fabs(p1.x - p2.x) + fabs(p1.y - p2.y);
}

// Function for printing path with hole IDs
void printPath(const int* path, const int numPoints, const vector<Point>& points, const double length) {
    cout << "Hole visit order: ";
    for (int i = 0; i < numPoints; i++) {
        cout << points[path[i]].id;
        if (i < numPoints - 1) cout << " -> ";
    }
    cout << endl;
    cout << "Path length: " << fixed << setprecision(2) << length << " mm" << endl;
}

// Function for printing detailed path with coordinates and segment distances
void printDetailedPath(const int* path, const int numPoints, const vector<Point>& points, const vector<vector<double>>& distMatrix) {
    cout << "\nDetailed path breakdown:\n";
    cout << ProjectInfo::SEPARATOR_SMALL << "\n";
    for (int i = 0; i < numPoints; i++) {
        Point p = points[path[i]];
        cout << "Hole " << setw(2) << p.id << " (" << fixed << setprecision(1)
             << setw(5) << p.x << ", " << setw(5) << p.y << " )";
        if (i < numPoints - 1) {
            const double dist = distMatrix[path[i]][path[i + 1]];
            cout << " -> distance: " << fixed << setprecision(2) << setw(6) << dist << " mm";
        } else {
            cout << " (END)";
        }
        cout << endl;
    }
}

// Function to generate next permutation (from class)
int next_permutation(const int N, int *P) {
    int* first = &P[0];
    int* last = &P[N-1];
    int* k = last - 1;
    int* l = last;

    // Find largest k so that P[k] < P[k+1]
    while (k > first) {
        if (*k < *(k+1)) {
            break;
        }
        k--;
    }

    // If no P[k] < P[k+1], P is the last permutation in lexicographic order
    if (*k > *(k+1)) {
        return 0;
    }

    // Find largest l so that P[k] < P[l]
    while (l > k) {
        if (*l > *k) {
            break;
        }
        l--;
    }

    // Swap P[l] and P[k]
    int s = *k;
    *k = *l;
    *l = s;

    // Reverse the remaining P[k+1]...P[N-1]
    first = k + 1;
    while (first < last) {
        s = *first;
        *first = *last;
        *last = s;

        first++;
        last--;
    }

    return 1;
}

// Function for solving TSP problem for given number of holes
void solveTSP(vector<Point>& points, const int numPoints) {
    cout << "\n" << ProjectInfo::SEPARATOR << endl;
    cout << "Solving TSP for " << numPoints << " holes" << endl;
    cout << ProjectInfo::SEPARATOR << endl;

    const clock_t startTime = clock();

    const vector<Point> selectedPoints(points.begin(), points.begin() + numPoints);

    vector<vector<double>> distMatrix(numPoints, vector<double>(numPoints, 0.0));
    for (int i = 0; i < numPoints; i++) {
        for (int j = i + 1; j < numPoints; j++) {
            const double dist = distanceBetweenTwoPoints(selectedPoints[i], selectedPoints[j]);
            distMatrix[i][j] = dist;
            distMatrix[j][i] = dist;
        }
    }

    double globalMinLength = numeric_limits<double>::max();
    const auto globalBestPath = new int[numPoints];
    long long totalPermutations = 0;

    for (int startPoint = 0; startPoint < numPoints; startPoint++) {
        const auto P = new int[numPoints - 1];
        int idx = 0;
        for (int i = 0; i < numPoints; i++) {
            if (i != startPoint) {
                P[idx++] = i;
            }
        }

        do {
            totalPermutations++;

            const auto fullPath = new int[numPoints];
            fullPath[0] = startPoint;
            for (int i = 0; i < numPoints - 1; i++) {
                fullPath[i + 1] = P[i];
            }

            double currentLength = 0.0;
            for (int i = 0; i < numPoints - 1; i++) {
                currentLength += distMatrix[fullPath[i]][fullPath[i + 1]];
            }

            if (currentLength < globalMinLength) {
                globalMinLength = currentLength;
                for (int i = 0; i < numPoints; i++) {
                    globalBestPath[i] = fullPath[i];
                }
            }

            delete[] fullPath;

        } while (next_permutation(numPoints - 1, P));

        delete[] P;
    }

    const clock_t endTime = clock();
    const double timeSpent = (double)(endTime - startTime) / CLOCKS_PER_SEC;

    cout << "\nTotal permutations tested: " << totalPermutations << endl;
    cout << "Execution time: " << fixed << setprecision(3) << timeSpent << " seconds" << endl;
    cout << "\nShortest path found:" << endl;
    printPath(globalBestPath, numPoints, selectedPoints, globalMinLength);

    printDetailedPath(globalBestPath, numPoints, selectedPoints, distMatrix);

    delete[] globalBestPath;
}

int main() {
    vector<Point> holes = {
        {1, 34.7, 45.1},
        {2, 34.7, 26.4},
        {3, 33.4, 60.5},
        {4, 51.7, 56.0},
        {5, 45.7, 25.1},
        {6, 62.0, 58.4},
        {7, 57.7, 42.1},
        {8, 46.0, 45.1},
        {9, 54.2, 29.1},
        {10, 57.5, 56.0},
        {11, 67.9, 19.6},
        {12, 21.5, 45.8}
    };


    cout << ProjectInfo::SEPARATOR << endl;
    cout << ProjectInfo::COURSE << endl;
    cout << "(" << ProjectInfo::COURSE_EN << ")" << endl;
    cout << ProjectInfo::ASSIGNMENT << endl;
    cout << ProjectInfo::SEPARATOR_SMALL << endl;
    cout << "Author: " << ProjectInfo::AUTHOR << " " << ProjectInfo::INDEX << endl;
    cout << ProjectInfo::SEPARATOR << "\n" << endl;

    // cout << ProjectInfo::PROBLEM << endl;
    // cout << ProjectInfo::SEPARATOR << endl;
    cout << "\nTotal number of holes: " << holes.size() << endl;


    // (a) with 8 points
    solveTSP(holes, 8);

    // (b) with 12 points
    solveTSP(holes,12);


    return 0;
}
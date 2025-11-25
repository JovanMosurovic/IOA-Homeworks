#include <cfloat>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <climits>
#include <cstring>
#include <ctime>

using namespace std;

constexpr int NUM_CITIES = 10;

int cost_matrix[NUM_CITIES][NUM_CITIES] = {
    {0, 374, 101, 223, 108, 178, 252, 285, 240, 356},
    {374, 0, 27, 166, 433, 199, 135, 95, 136, 17},
    {101, 27, 0, 41, 52, 821, 180, 201, 131, 247},
    {223, 166, 41, 0, 430, 47, 52, 84, 40, 155},
    {108, 433, 52, 430, 0, 453, 478, 344, 389, 423},
    {178, 199, 821, 47, 453, 0, 91, 37, 64, 181},
    {252, 135, 180, 52, 478, 91, 0, 25, 83, 117},
    {285, 95, 201, 84, 344, 37, 25, 0, 51, 42},
    {240, 136, 131, 40, 389, 64, 83, 51, 0, 118},
    {356, 17, 247, 155, 423, 181, 117, 42, 118, 0}
};

char city_names[NUM_CITIES] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};

namespace ProjectInfo {
    const string COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI";
    const string COURSE_EN = "ENGINEERING OPTIMIZATION ALGORITHMS";
    const string ASSIGNMENT = "Homework Assignment 3";
    const string AUTHOR = "Jovan Mosurovic";
    const string INDEX = "2022/0589";
    const string SEPARATOR = "==============================================";
    const string SEPARATOR_SMALL = "----------------------------------------------";
    const string PROBLEM = "SPANNING TREE - POWER GRID OPTIMIZATION";
}

struct Edge {
    int u, v;
    int cost;
};

double min_cost = DBL_MAX;
Edge best_tree[9];
int best_prufer[8];
long long total_trees_checked = 0;
long long progress_counter = 0;
constexpr long long PROGRESS_INTERVAL = 10000000;

void decode_prufer(const int* prufer, const int n, Edge* edges) {
    int* degree = new int[n];

    for (int i = 0; i < n; i++) {
        degree[i] = 1;
    }

    for (int i = 0; i < n - 2; i++) {
        degree[prufer[i]]++;
    }

    int edge_count = 0;
    for (int i = 0; i < n - 2; i++) {
        for (int j = 0; j < n; j++) {
            if (degree[j] == 1) {
                edges[edge_count].u = j;
                edges[edge_count].v = prufer[i];
                edges[edge_count].cost = cost_matrix[j][prufer[i]];
                edge_count++;

                degree[j]--;
                degree[prufer[i]]--;
                break;
            }
        }
    }

    int remaining[2];
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (degree[i] == 1) {
            remaining[k++] = i;
        }
    }
    edges[edge_count].u = remaining[0];
    edges[edge_count].v = remaining[1];
    edges[edge_count].cost = cost_matrix[remaining[0]][remaining[1]];

    delete[] degree;
}

double calculate_cost(const Edge* edges, const int n) {
    double total_cost = 0.0;
    int* node_degree = new int[n];

    for (int i = 0; i < n; i++) {
        node_degree[i] = 0;
    }

    for (int i = 0; i < n - 1; i++) {
        int u = edges[i].u;
        int v = edges[i].v;
        total_cost += cost_matrix[u][v];
        node_degree[u]++;
        node_degree[v]++;
    }

    for (int i = 0; i < n; i++) {
        if (node_degree[i] >= 4) {
            total_cost += (node_degree[i] - 3) * 250;
        }
    }

    delete[] node_degree;
    return total_cost;
}

void printSummary(const Edge* tree, const int n, const double cost) {
    cout << "\nOptimal network connection:" << endl;
    cout << "Connection path: ";
    for (int i = 0; i < n - 1; i++) {
        cout << city_names[tree[i].u] << " " << city_names[tree[i].v];
        if (i < n - 2) cout << " - ";
    }
    cout << endl;
    cout << "Total network cost: " << fixed << setprecision(2) << cost << endl;
}

void printDetailedBreakdown(const Edge* tree, const int n) {
    cout << "\n" << ProjectInfo::SEPARATOR << endl;
    cout << "Detailed cost breakdown" << endl;
    cout << ProjectInfo::SEPARATOR << endl;

    int* node_degree = new int[n];
    for (int i = 0; i < n; i++) {
        node_degree[i] = 0;
    }

    double edge_cost_sum = 0;

    cout << "\nEdge costs:" << endl;
    for (int i = 0; i < n - 1; i++) {
        int u = tree[i].u;
        int v = tree[i].v;
        int cost = cost_matrix[u][v];

        node_degree[u]++;
        node_degree[v]++;
        edge_cost_sum += cost;

        cout << "Edge " << setw(2) << (i + 1) << ": "
             << city_names[u] << " - " << city_names[v]
             << " -> cost: " << setw(4) << cost << endl;
    }

    cout << ProjectInfo::SEPARATOR_SMALL << endl;
    cout << "Sum of edge costs: " << fixed << setprecision(2) << edge_cost_sum << endl;

    double penalty = 0;
    bool has_penalty = false;

    cout << "\nBranching analysis:" << endl;
    for (int i = 0; i < n; i++) {
        cout << "City " << city_names[i] << ": " << node_degree[i] << " branch(es)";
        if (node_degree[i] >= 4) {
            double node_penalty = (node_degree[i] - 3) * 250;
            penalty += node_penalty;
            cout << " -> penalty: " << fixed << setprecision(0) << node_penalty;
            has_penalty = true;
        }
        cout << endl;
    }

    if (!has_penalty) {
        cout << "No penalties (all nodes have < 4 branches)" << endl;
    }

    cout << ProjectInfo::SEPARATOR_SMALL << endl;
    cout << "Total penalty: " << fixed << setprecision(2) << penalty << endl;
    cout << "TOTAL COST: " << fixed << setprecision(2) << (edge_cost_sum + penalty) << endl;

    delete[] node_degree;
}

void exhaustive_search(int* prufer, const int pos, const int n) {
    if (pos == n - 2) {
        total_trees_checked++;
        progress_counter++;

        if (progress_counter >= PROGRESS_INTERVAL) {
            cout << "Progress: " << (total_trees_checked >= 1000000
                 ? to_string(total_trees_checked / 1000000) + "M"
                 : to_string(total_trees_checked)) << " trees checked, "
                 << "current best: " << fixed << setprecision(2) << min_cost << endl;
            progress_counter = 0;
        }

        Edge edges[9];
        decode_prufer(prufer, n, edges);
        double cost = calculate_cost(edges, n);

        if (cost < min_cost) {
            min_cost = cost;
            memcpy(best_tree, edges, sizeof(Edge) * (n - 1));
            memcpy(best_prufer, prufer, sizeof(int) * (n - 2));
            cout << "New best cost found: " << fixed << setprecision(2) << cost << endl;
        }

        return;
    }

    for (int i = 0; i < n; i++) {
        prufer[pos] = i;
        exhaustive_search(prufer, pos + 1, n);
    }
}

void saveResults(const Edge* tree, const int n, const double time_spent) {
    ofstream outfile("results.txt");
    if (!outfile.is_open()) {
        cerr << "Error: Could not open output file!" << endl;
        return;
    }

    outfile << ProjectInfo::SEPARATOR << endl;
    outfile << ProjectInfo::COURSE << endl;
    outfile << "(" << ProjectInfo::COURSE_EN << ")" << endl;
    outfile << ProjectInfo::ASSIGNMENT << endl;
    outfile << "Author: " << ProjectInfo::AUTHOR << " " << ProjectInfo::INDEX << endl;
    outfile << ProjectInfo::SEPARATOR << "\n" << endl;

    outfile << "OPTIMAL POWER GRID NETWORK" << endl;
    outfile << ProjectInfo::SEPARATOR << "\n" << endl;

    outfile << "Minimum cost: " << fixed << setprecision(2) << min_cost << endl;
    outfile << "\nConnection path: ";

    for (int i = 0; i < n - 1; i++) {
        outfile << city_names[tree[i].u] << " " << city_names[tree[i].v];
        if (i < n - 2) outfile << " - ";
    }
    outfile << endl;

    outfile << "\nPrufer sequence: ";
    for (int i = 0; i < n - 2; i++) {
        outfile << best_prufer[i];
        if (i < n - 3) outfile << " ";
    }
    outfile << endl;

    outfile << "\n" << ProjectInfo::SEPARATOR_SMALL << endl;
    outfile << "Edge breakdown:" << endl;
    outfile << ProjectInfo::SEPARATOR_SMALL << endl;

    for (int i = 0; i < n - 1; i++) {
        int u = tree[i].u;
        int v = tree[i].v;
        int cost = cost_matrix[u][v];
        outfile << "Edge " << setw(2) << (i + 1) << ": "
               << city_names[u] << " - " << city_names[v]
               << " -> cost: " << setw(4) << cost << endl;
    }

    outfile << "\n" << ProjectInfo::SEPARATOR_SMALL << endl;
    outfile << "Computation time: " << fixed << setprecision(3)
            << time_spent << " seconds" << endl;
    outfile << "Trees checked: " << total_trees_checked << endl;
    outfile << ProjectInfo::SEPARATOR_SMALL << endl;

    outfile.close();
}

int main() {
    constexpr int n = 10;
    int prufer[8];

    cout << ProjectInfo::SEPARATOR << endl;
    cout << ProjectInfo::COURSE << endl;
    cout << "(" << ProjectInfo::COURSE_EN << ")" << endl;
    cout << ProjectInfo::ASSIGNMENT << endl;
    cout << ProjectInfo::SEPARATOR_SMALL << endl;
    cout << "Author: " << ProjectInfo::AUTHOR << " " << ProjectInfo::INDEX << endl;
    cout << ProjectInfo::SEPARATOR << "\n" << endl;

    cout << ProjectInfo::PROBLEM << endl;
    cout << ProjectInfo::SEPARATOR << endl;

    cout << "\nProblem configuration:" << endl;
    cout << "Cities to connect: " << n << endl;
    cout << "Objective: Minimize total cost" << endl;
    cout << "Penalty: (g-3) x 250 for nodes with g >= 4 branches" << endl;
    cout << "Search space: 10^8 spanning trees" << endl;

    cout << "\n" << ProjectInfo::SEPARATOR << endl;
    cout << "Starting exhaustive search..." << endl;
    cout << ProjectInfo::SEPARATOR << "\n" << endl;

    clock_t start_time = clock();

    exhaustive_search(prufer, 0, n);

    const clock_t end_time = clock();
    const double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    cout << "\n" << ProjectInfo::SEPARATOR << endl;
    cout << "Search completed" << endl;
    cout << "Total trees checked: " << total_trees_checked << endl;
    cout << "Execution time: " << fixed << setprecision(3)
         << time_spent << " seconds" << endl;
    cout << ProjectInfo::SEPARATOR << endl;

    printSummary(best_tree, n, min_cost);
    printDetailedBreakdown(best_tree, n);

    saveResults(best_tree, n, time_spent);

    cout << "\n" << ProjectInfo::SEPARATOR << endl;
    cout << "Results saved to 'results.txt'" << endl;
    cout << ProjectInfo::SEPARATOR << endl;

    return 0;
}
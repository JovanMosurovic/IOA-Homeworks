#include <cmath>
#include <ctime>
#include "output_handler.h"

using namespace std;

constexpr double TARGET = 9.78;
constexpr double TOLERANCE = 0.0001;
constexpr int MAX_PRICE_CENTS = 978;

long long bruteForce4Prices(ofstream& output, vector<Solution>& solutions) {
    long long functionCalls = 0;

    OutputHandler::writeMethod4Header(output);
    OutputHandler::printMethod4Start();

    const clock_t start = clock();

    for (int p1 = 1; p1 <= MAX_PRICE_CENTS; p1++) {
        OutputHandler::printMethod4Progress(p1, MAX_PRICE_CENTS);
        const double price1 = p1 / 100.0;

        for (int p2 = 1; p2 <= MAX_PRICE_CENTS; p2++) {
            const double price2 = p2 / 100.0;

            for (int p3 = 1; p3 <= MAX_PRICE_CENTS; p3++) {
                const double price3 = p3 / 100.0;

                for (int p4 = 1; p4 <= MAX_PRICE_CENTS; p4++) {
                    const double price4 = p4 / 100.0;

                    // OPTIMIZACIJA: izbegavanje permutacija
                    // Broj provera se smanjuje približno 24 puta -> permutacije četiri broja imaju 4! = 24 kombinacije
                    // if (!(p1 <= p2 && p2 <= p3 && p3 <= p4))
                    //     continue;

                    functionCalls++;

                    const double sum = price1 + price2 + price3 + price4;
                    const double product = price1 * price2 * price3 * price4;

                    if (fabs(sum - TARGET) < TOLERANCE && fabs(product - TARGET) < TOLERANCE) {
                        Solution sol{price1, price2, price3, price4, sum, product};
                        solutions.push_back(sol);
                    }
                }
            }
        }
    }

    const clock_t end = clock();
    const double timeSpent = (double)(end - start) / CLOCKS_PER_SEC;

    OutputHandler::writeMethod4Results(output, functionCalls, timeSpent);
    OutputHandler::printMethod4Complete(functionCalls, timeSpent);

    return functionCalls;
}

long long bruteForce3Prices(ofstream& output, vector<Solution>& solutions) {
    long long functionCalls = 0;

    OutputHandler::writeMethod3Header(output);
    OutputHandler::printMethod3Start();

    const clock_t start = clock();

    for (int p1 = 1; p1 <= MAX_PRICE_CENTS; p1++) {
        OutputHandler::printMethod3Progress(p1, MAX_PRICE_CENTS);
        double price1 = p1 / 100.0;

        for (int p2 = 1; p2 <= MAX_PRICE_CENTS; p2++) {
            double price2 = p2 / 100.0;

            for (int p3 = 1; p3 <= MAX_PRICE_CENTS; p3++) {
                double price3 = p3 / 100.0;
                functionCalls++;

                double price4 = TARGET - price1 - price2 - price3;

                if (price4 < 0.01 || price4 > 9.78)
                    continue;

                double rounded = round(price4 * 100.0) / 100.0;
                if (fabs(price4 - rounded) > TOLERANCE)
                    continue;

                double product = price1 * price2 * price3 * price4;

                if (fabs(product - TARGET) < TOLERANCE) {
                    double sum = price1 + price2 + price3 + price4;
                    Solution sol{price1, price2, price3, price4, sum, product};
                    solutions.push_back(sol);
                }
            }
        }
    }

    const clock_t end = clock();
    const double timeSpent = (double)(end - start) / CLOCKS_PER_SEC;

    OutputHandler::writeMethod3Results(output, functionCalls, timeSpent);
    OutputHandler::printMethod3Complete(functionCalls, timeSpent);

    return functionCalls;
}

int main() {
    ofstream output("rezultati.txt");

    if (!output.is_open()) {
        cerr << "Greska pri otvaranju fajla rezultati.txt!\n";
        return 1;
    }

    OutputHandler::writeHeader(output);
    OutputHandler::printProgramHeader();

    // (a) Metod sa 4 cene
    vector<Solution> solutions_a;
    const clock_t start_a = clock();
    const long long calls_a = bruteForce4Prices(output, solutions_a);
    const clock_t end_a = clock();
    const double time_a = (double)(end_a - start_a) / CLOCKS_PER_SEC;

    OutputHandler::writeSolutions(output, solutions_a, "(a)");

    // (b) Metod sa 3 cene
    vector<Solution> solutions_b;
    const clock_t start_b = clock();
    const long long calls_b = bruteForce3Prices(output, solutions_b);
    const clock_t end_b = clock();
    const double time_b = (double)(end_b - start_b) / CLOCKS_PER_SEC;

    OutputHandler::writeSolutions(output, solutions_b, "(b)");

    // (v) Poredjenje
    OutputHandler::writeComparison(output, calls_a, calls_b, time_a, time_b);

    output.close();

    OutputHandler::printFinalSummary(calls_a, calls_b, time_a, time_b);

    return 0;
}
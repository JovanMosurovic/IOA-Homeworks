#ifndef OUTPUT_HANDLER_H
#define OUTPUT_HANDLER_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

using namespace std;

struct Solution {
    double p1, p2, p3, p4;
    double sum, product;
};

namespace ProjectInfo {
    const string COURSE = "INZENJERSKI OPTIMIZACIONI ALGORITMI";
    const string ASSIGNMENT = "Domaci zadatak 1";
    const string AUTHOR = "Jovan Mosurovic";
    const string INDEX = "2022/0589";
    const string SEPARATOR = "========================================";
}

class OutputHandler {
public:
    static void writeHeader(ofstream& output) {
        output << ProjectInfo::SEPARATOR << "\n";
        output << ProjectInfo::COURSE << "\n";
        output << ProjectInfo::ASSIGNMENT << "\n";
        output << "----------------------------------------\n";
        output << "Autor: " << ProjectInfo::AUTHOR << " " << ProjectInfo::INDEX << "\n";
        output << ProjectInfo::SEPARATOR << "\n\n";

        output << "ZADATAK:\n";
        output << "------------------\n";
        output << "Kupac je kupio cetiri proizvoda.\n";
        output << "Zbir svih cena: $9.78\n";
        output << "Proizvod svih cena: $9.78\n";
        output << "Cene se zaokruzuju na $0.01 (1 cent)\n\n";

        output << "USLOVI:\n";
        output << "p1 + p2 + p3 + p4 = 9.78\n";
        output << "p1 x p2 x p3 x p4 = 9.78\n";
        output << "0.01 <= pi <= 9.78, i = 1,2,3,4\n\n";
    }

    static void writeMethod4Header(ofstream& output) {
        output << ProjectInfo::SEPARATOR << "\n";
        output << "(a) METOD SA 4 CENE - POTPUNA PRETRAGA\n";
        output << ProjectInfo::SEPARATOR << "\n\n";
    }

    static void writeMethod4Results(ofstream& output, const long long functionCalls, const double timeSpent) {
        output << "Teorijski maksimalan broj poziva: 978^4 = 914,861,642,256\n";
        output << "Stvarni broj poziva: " << functionCalls << "\n";
        output << "Vreme izvrsavanja: " << fixed << setprecision(2) << timeSpent << " sekundi\n";
        output << "                   ≈ " << (timeSpent / 60.0) << " minuta\n";
        output << "                   ≈ " << (timeSpent / 3600.0) << " sati\n\n";
    }

    static void writeMethod3Header(ofstream& output) {
        output << ProjectInfo::SEPARATOR << "\n";
        output << "(b) METOD SA 3 CENE - REDUKOVAN PROSTOR\n";
        output << ProjectInfo::SEPARATOR << "\n\n";
    }

    static void writeMethod3Results(ofstream& output, const long long functionCalls, const double timeSpent) {
        output << "Teorijski maksimalan broj poziva: 978^3 = 935,441,352\n";
        output << "Stvarni broj poziva: " << functionCalls << "\n";
        output << "Vreme izvrsavanja: " << fixed << setprecision(2) << timeSpent << " sekundi\n";
        output << "                   ≈ " << (timeSpent / 60.0) << " minuta\n";
        output << "                   ≈ " << (timeSpent / 3600.0) << " sati\n\n";
    }

    static void writeSolutions(ofstream& output, const vector<Solution>& solutions, const string& method) {
        output << "PRONADJENA RESENJA - METOD " << method << "\n";
        output << ProjectInfo::SEPARATOR << "\n\n";

        if (solutions.empty()) {
            output << "Nije pronadjeno nijedno resenje!\n\n";
            return;
        }

        output << "Ukupan broj resenja: " << solutions.size() << "\n\n";

        // const int maxDisplay = min((int)solutions.size(), 20);
        const int maxDisplay = (int)solutions.size();

        for (int i = 0; i < maxDisplay; i++) {
            output << "Resenje " << (i + 1) << ":\n";
            output << fixed << setprecision(2);
            output << "  Cene: $" << solutions[i].p1
                   << ", $" << solutions[i].p2
                   << ", $" << solutions[i].p3
                   << ", $" << solutions[i].p4 << "\n";
            output << "  Zbir:     $" << solutions[i].sum << "\n";
            output << "  Proizvod: $" << solutions[i].product << "\n\n";
        }

        if (solutions.size() > maxDisplay) {
            output << "... (prikazano prvih " << maxDisplay << " od "
                   << solutions.size() << " resenja)\n\n";
        }
    }

    static void writeComparison(ofstream& output, const long long calls_a, const long long calls_b, const double time_a, const double time_b) {
        output << ProjectInfo::SEPARATOR << "\n";
        output << "(v) POREDJENJE BRZINE PROGRAMA\n";
        output << ProjectInfo::SEPARATOR << "\n\n";

        output << "TEORIJSKA ANALIZA:\n";
        output << "------------------\n";
        output << "Metod (a) - 4 cene:\n";
        output << "  Slozenost: O(n^4)\n";
        output << "  Teorijski maksimum: 978^4 = 914,861,642,256 poziva\n\n";

        output << "Metod (b) - 3 cene:\n";
        output << "  Slozenost: O(n^3)\n";
        output << "  Teorijski maksimum: 978^3 = 935,441,352 poziva\n\n";

        output << "Teorijsko ubrzanje: 978^4 / 978^3 = 978x\n\n";

        output << "\nSTVARNI REZULTATI:\n";
        output << "------------------\n";
        output << "Metod (a):\n";
        output << "  Broj poziva: " << calls_a << "\n";
        output << "  Vreme: " << fixed << setprecision(2) << time_a << " sekundi\n";
        output << "         ≈ " << (time_a / 60.0) << " minuta\n";
        output << "         ≈ " << (time_a / 3600.0) << " sati\n\n";

        output << "Metod (b):\n";
        output << "  Broj poziva: " << calls_b << "\n";
        output << "  Vreme: " << fixed << setprecision(2) << time_b << " sekundi\n";
        output << "         ≈ " << (time_b / 60.0) << " minuta\n";
        output << "         ≈ " << (time_b / 3600.0) << " sati\n\n";

        double callsRatio = (double)calls_a / calls_b;
        double timeRatio = time_a / time_b;

        output << "Odnos broja poziva: " << fixed << setprecision(1)
               << callsRatio << "x (metod a / metod b)\n";
        output << "Odnos vremena: " << fixed << setprecision(1)
               << timeRatio << "x (metod a / metod b)\n\n";

        output << "\nZAKLJUCAK:\n";
        output << "----------\n";
        output << "Metod (b) je brzi od metoda (a) za faktor " << fixed << setprecision(0)
               << callsRatio << "x.\n\n";
        output << "Razlog: Eliminisanjem jedne dimenzije pretrage (koristeci relaciju\n";
        output << "p4 = 9.78 - p1 - p2 - p3), redukovana je slozenost sa O(n^4) na O(n^3),\n";
        output << "sto direktno dovodi do " << (long long)(978) << "x manjeg broja iteracija.\n\n";
    }

    // Console output metode
    static void printProgramHeader() {
        cout << "========================================\n";
        cout << "INZENJERSKI OPTIMIZACIONI ALGORITMI\n";
        cout << "Domaci zadatak 1\n";
        cout << "----------------------------------------\n";
        cout << "Autor: Jovan Mosurovic 2022/0589\n";
        cout << "========================================\n";
    }

    static void printMethod4Start() {
        cout << "\n(a) Pokretanje potpune pretrage po 4 cene...\n";
    }

    static void printMethod4Progress(const int p1, const int maxPrice) {
        if (p1 % 100 == 0) {
            cout << "Progres: " << p1 << "/" << maxPrice
                 << " (" << (p1*100.0/maxPrice) << "%)\r" << flush;
        }
    }

    static void printMethod4Complete(const long long functionCalls, const double timeSpent) {
        cout << "\n\n(a) Zavrseno!\n";
        cout << "Broj poziva: " << functionCalls << "\n";
        cout << "Vreme: " << fixed << setprecision(2) << timeSpent << " sekundi "
             << "(≈" << (timeSpent / 60.0) << " minuta, ≈" << (timeSpent / 3600.0) << " sati)\n";
    }

    static void printMethod3Start() {
        cout << "\n(b) Pokretanje pretrage po 3 cene...\n";
    }

    static void printMethod3Progress(const int p1, const int maxPrice) {
        if (p1 % 50 == 0) {
            cout << "Progres: " << p1 << "/" << maxPrice
                 << " (" << (p1*100.0/maxPrice) << "%)\r" << flush;
        }
    }

    static void printMethod3Complete(const long long functionCalls, const double timeSpent) {
        cout << "\n\n(b) Zavrseno!\n";
        cout << "Broj poziva: " << functionCalls << "\n";
        cout << "Vreme: " << fixed << setprecision(2) << timeSpent << " sekundi "
             << "(≈" << (timeSpent / 60.0) << " minuta, ≈" << (timeSpent / 3600.0) << " sati)\n";
    }

    static void printFinalSummary(const long long calls_a, const long long calls_b, const double time_a, const double time_b) {
        cout << "\n========================================\n";
        cout << "Rezultati su uspesno upisani u fajl:\n";
        cout << "rezultati.txt\n";
        cout << "========================================\n\n";

        cout << "\nODGOVORI NA PITANJA:\n";
        cout << "--------------------\n";
        cout << "(a) Maksimalan broj poziva: " << calls_a << "\n";
        cout << "    Vreme: " << time_a << " sekundi\n\n";

        cout << "(b) Maksimalan broj poziva: " << calls_b << "\n";
        cout << "    Vreme: " << time_b << " sekundi\n\n";

        cout << "(v) Program (b) je brzi!\n";
        cout << "    Ubrzanje: " << (double)calls_a/calls_b << "x manje poziva\n";
        cout << "    Ubrzanje: " << time_a/time_b << "x brze vreme\n\n";
    }
};

#endif
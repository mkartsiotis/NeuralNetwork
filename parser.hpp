#ifndef PARSER_HPP
#define PARSER_HPP
#include "Matrix.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
// This is the parser part that was a bit vibe coded...
template <typename T>
bool load_mnist_csv(const std::string &filename, std::vector<Matrix<T>> &images, std::vector<Matrix<T>> &labels,
                    int max_rows)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    if (!getline(file, line))
        return false;

    int row_count = 0;
    while (getline(file, line) && row_count < max_rows)
    {
        std::stringstream ss(line);
        std::string token;

        if (getline(ss, token, ','))
        {
            int label_val = stoi(token);
            Matrix<T> target(10, 1);
            target.getData()[label_val][0] = 1.0; // 1.0 στη σωστή θέση, 0.0 παντού αλλού
            labels.push_back(target);
        }

        // Φτιάχνουμε τον Input std::vector (784x1) για τα pixels
        Matrix<T> image(784, 1);
        int pixel_idx = 0;

        // Διαβάζουμε τα υπόλοιπα 784 στοιχεία (τα pixels) χωρισμένα με κόμμα
        while (getline(ss, token, ',') && pixel_idx < 784)
        {
            T pixel_val = (T)stod(token);
            // Normalization: Διαιρούμε με το 255.0 για να πάει στο [0.0, 1.0]
            image.getData()[pixel_idx][0] = pixel_val / 255.0;
            pixel_idx++;
        }

        images.push_back(image);
        row_count++;
    }

    file.close();
    std::cout << "Successfully loaded " << row_count << " rows from " << filename << std::endl;
    return true;
}

#endif // !PARSER_HPP

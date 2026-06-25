#include "ML.hpp"
#include "parser.hpp"
#include <iostream>
using namespace std;
/*
*int main()
{
    cout << "INITIALIZING Matrix" << endl;
    Matrix MyMatrix1(3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            MyMatrix1.getData()[i][j] = i;
    Matrix MyMatrix2(3, 3);
    MyMatrix2 = MyMatrix1 * MyMatrix1;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << MyMatrix2.getData()[i][j] << " ";
        cout << endl;
    }
    MyMatrix2 = MyMatrix2.traspose();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            cout << MyMatrix2.getData()[i][j] << " ";
        cout << endl;
    }
}
*/
int main()
{
    NeuralNetwork<float> NN({784, 64, 10});

    vector<Matrix<float>> all_images;
    vector<Matrix<float>> all_labels;
    if (!load_mnist_csv<float>("mnist_test.csv", all_images, all_labels, 10000))
    {
        return -1;
    }

    vector<Matrix<float>> train_images(all_images.begin(), all_images.begin() + 7000);
    vector<Matrix<float>> train_labels(all_labels.begin(), all_labels.begin() + 7000);

    vector<Matrix<float>> test_images(all_images.begin() + 7000, all_images.end());
    vector<Matrix<float>> test_labels(all_labels.begin() + 7000, all_labels.end());

    cout << "Starting Training..." << endl;

    int epochs = 10;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_loss = 0;
        for (size_t i = 0; i < train_images.size(); i++)
        {
            // Forward Pass
            Matrix<float> output = NN.forward(train_images[i]);

            // Cost function reeoc calculation
            Matrix<float> error = output + (train_labels[i] * -1.0);

            // MSE loss
            for (int k = 0; k < 10; k++)
            {
                total_loss += error.getData()[k][0] * error.getData()[k][0];
            }

            // 3. Backward Pass
            NN.backward(error);

            // 4. Update Weights
            NN.update();
        }
        cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss / train_images.size() << endl;
    }

    cout << "Training finished!" << endl;

    cout << "\nStarting Testing..." << endl;
    int correct_predictions = 0;

    for (size_t i = 0; i < test_images.size(); i++)
    {
        Matrix<float> output = NN.forward(test_images[i]);
        int predicted_digit = 0;
        double max_val = output.getData()[0][0];
        for (int k = 1; k < 10; k++)
        {
            if (output.getData()[k][0] > max_val)
            {
                max_val = output.getData()[k][0];
                predicted_digit = k;
            }
        }
        int actual_digit = 0;
        for (int k = 0; k < 10; k++)
        {
            if (test_labels[i].getData()[k][0] == 1.0)
            {
                actual_digit = k;
                break;
            }
        }

        if (predicted_digit == actual_digit)
        {
            correct_predictions++;
        }
    }

    double accuracy = (double)correct_predictions / test_images.size() * 100.0;
    cout << "Testing Finished! Accuracy: " << accuracy << "%" << endl;

    return 0;
}

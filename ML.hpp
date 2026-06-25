#ifndef ML_HPP
#define ML_HPP
#include "Matrix.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>
// IMPORTANT!!!!THis is the learning rate variable. Tweak this for faster learning
const double learning_rate = 0.2;

// Here is where the machine learning actually starts...
// Abstract class that just creates the basic infrastructure
template <typename T> T sigmoid(T x)
{
    return 1.0 / (1.0 + exp(-x));
}
template <typename T> T sigmoid_derivative(T x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}
template <typename T> T relu(T x)
{
    return x > 0 ? x : 0;
}

template <typename T> class Layer
{
  private:
  public:
    virtual Matrix<T> forward(const Matrix<T> &input) = 0;
    virtual Matrix<T> backward(const Matrix<T> &previous_layer_error) = 0;
    virtual void update_weights() = 0;
    virtual ~Layer()
    {
    }
};

template <typename T> class DenseLayer : public Layer<T>
{
  private:
    Matrix<T> *W; // weights
    Matrix<T> *B; // biases
    Matrix<T> *Input;
    Matrix<T> *Z; // OUTPUT Z = W Input + B, Z is before the non linear normalizatrion function like RELU or σ
    Matrix<T> *dJ_dW;
    Matrix<T> *dJ_dB;

  public:
    DenseLayer(int input_size, int output_size)
    {
        W = new Matrix<T>(output_size, input_size);
        for (int i = 0; i < W->getRows(); i++)
            for (int j = 0; j < W->getCols(); j++)
                W->getData()[i][j] = 0.5 - ((double)(rand() % 10000) / 10000.0);
        B = new Matrix<T>(output_size, 1);
        Input = new Matrix<T>(input_size, 1);
        Z = new Matrix<T>(output_size, 1);
        dJ_dW = new Matrix<T>(output_size, input_size);
        dJ_dB = new Matrix<T>(output_size, 1);
    }
    ~DenseLayer()
    {
        delete W;
        delete B;
        delete Input;
        delete Z;
        delete dJ_dB;
        delete dJ_dW;
    }
    // New OPP Knowledge!Pointer to function!!
    Matrix<T> normalization_funct(double (*func)(double)) const
    {
        Matrix<T> ReturnMatrix(Z->getRows(), Z->getCols());
        for (int i = 0; i < ReturnMatrix.getRows(); i++)
            for (int j = 0; j < ReturnMatrix.getCols(); j++)
                ReturnMatrix.getData()[i][j] = func(Z->getData()[i][j]);
        return ReturnMatrix;
    }
    Matrix<T> forward(const Matrix<T> &input) override
    {
        *Input = input;
        *Z = ((*W) * (*Input)) + (*B);
        return normalization_funct(sigmoid);
    }
    Matrix<T> backward(const Matrix<T> &previous_layer_error) override
    {
        // It is true that delta of the level is the dJ_dB
        *dJ_dB = previous_layer_error.Hadamard(normalization_funct(sigmoid_derivative));
        *dJ_dW = (*dJ_dB) * Input->traspose();
        return (W->traspose() * (*dJ_dB));
    }
    void update_weights() override
    {
        *W = ((*W) + ((*dJ_dW) * (-learning_rate)));
        *B = ((*B) + ((*dJ_dB) * (-learning_rate)));
        *dJ_dW = (*dJ_dW) * 0.0;
        *dJ_dB = (*dJ_dB) * 0.0;
    }
};

// Now for the final integration

template <typename T> class NeuralNetwork
{
  private:
    std::vector<int> topology;
    std::vector<Layer<T> *> Layer_Vector;

  public:
    NeuralNetwork(const std::vector<int> top) : topology(top)
    {
        for (size_t i = 0; i < topology.size() - 1; i++)
            Layer_Vector.push_back(new DenseLayer<T>(topology[i], topology[i + 1]));
    }
    ~NeuralNetwork()
    {
        for (size_t i = 0; i < topology.size() - 1; i++)
            delete Layer_Vector.at(i);
    }
    Matrix<T> forward(Matrix<T> input)
    {
        Matrix<T> current_act = input;
        for (size_t i = 0; i < Layer_Vector.size(); i++)
            current_act = Layer_Vector.at(i)->forward(current_act);
        return current_act;
    }
    void backward(const Matrix<T> &output_error)
    {
        Matrix<T> current_error = output_error;
        for (int i = Layer_Vector.size() - 1; i >= 0; i--)
            current_error = Layer_Vector[i]->backward(current_error);
    }
    void update()
    {
        for (size_t i = 0; i < Layer_Vector.size(); i++)
            Layer_Vector.at(i)->update_weights();
    }
};

#endif // !ML_HPP

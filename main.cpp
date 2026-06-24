#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
using namespace std;

class Matrix
{
  private:
    int number_of_rows, number_of_columns;
    double **data;

  public:
    Matrix(int rows, int colums);
    Matrix(const Matrix &other);
    int getRows() const
    {
        return number_of_rows;
    }
    int getCols() const
    {
        return number_of_columns;
    }
    void setCols(int c)
    {
        number_of_columns = c;
    }
    void setRows(int r)
    {
        number_of_rows = r;
    }
    double **getData()
    {
        return data;
    }
    double **getDataConst() const
    {
        return data;
    }
    Matrix Hadamard(const Matrix &B) const
    {
        Matrix ReturnMatrix(this->getRows(), this->getCols());
        if (this->getRows() != B.getRows() || this->getCols() != B.getCols())
            return ReturnMatrix;
        for (int i = 0; i < ReturnMatrix.number_of_rows; i++)
            for (int j = 0; j < ReturnMatrix.number_of_columns; j++)
                ReturnMatrix.getData()[i][j] = this->getDataConst()[i][j] * B.getDataConst()[i][j];
        return ReturnMatrix;
    }
    Matrix operator*(const Matrix &B)
    {
        Matrix ReturnMatrix(this->number_of_rows, B.number_of_columns);
        if (B.number_of_rows != this->number_of_columns)
            return ReturnMatrix;
        for (int i = 0; i < ReturnMatrix.number_of_rows; i++)
            for (int j = 0; j < ReturnMatrix.number_of_columns; j++)
            {
                double element = 0;
                for (int k = 0; k < this->number_of_columns; k++)
                    element += (this->getData())[i][k] * (B.getDataConst())[k][j];
                (ReturnMatrix.getData())[i][j] = element;
            }
        return ReturnMatrix;
    }
    Matrix &operator=(const Matrix &other);
    Matrix operator+(const Matrix &B)
    {
        Matrix ReturnMatrix(this->number_of_rows, this->number_of_columns);
        if (B.number_of_rows != this->number_of_rows || B.number_of_columns != this->number_of_columns)
            return ReturnMatrix;
        for (int i = 0; i < ReturnMatrix.number_of_rows; i++)
            for (int j = 0; j < ReturnMatrix.number_of_columns; j++)
                (ReturnMatrix.getData())[i][j] = this->getData()[i][j] + B.getDataConst()[i][j];
        return ReturnMatrix;
    }
    Matrix traspose();
    ~Matrix();
};

Matrix::Matrix(int rows, int columns) : number_of_rows(rows), number_of_columns(columns)
{
    data = new double *[number_of_rows];
    for (int i = 0; i < number_of_rows; i++)
        data[i] = new double[number_of_columns]();
}
Matrix::Matrix(const Matrix &other) : number_of_rows(other.number_of_rows), number_of_columns(other.number_of_columns)
{
    data = new double *[number_of_rows];
    for (int i = 0; i < number_of_rows; i++)
        data[i] = new double[number_of_columns];
    for (int i = 0; i < number_of_rows; i++)
        for (int j = 0; j < number_of_columns; j++)
            data[i][j] = other.getDataConst()[i][j];
}
Matrix::~Matrix()
{
    for (int i = 0; i < number_of_rows; i++)
        delete[] data[i];
    delete[] data;
}
Matrix Matrix::traspose()
{
    Matrix ReturnMatrix(this->getCols(), this->getRows());
    for (int i = 0; i < ReturnMatrix.number_of_rows; i++)
        for (int j = 0; j < ReturnMatrix.number_of_columns; j++)
            (ReturnMatrix.getData())[i][j] = this->getData()[j][i];
    return ReturnMatrix;
}

Matrix &Matrix::operator=(const Matrix &other)
{
    if (this == &other)
        return *this;
    for (int i = 0; i < number_of_rows; i++)
    {
        delete[] data[i];
    }
    delete[] data;
    this->number_of_rows = other.number_of_rows;
    this->number_of_columns = other.number_of_columns;

    data = new double *[number_of_rows];
    for (int i = 0; i < number_of_rows; i++)
    {
        data[i] = new double[number_of_columns];
    }
    for (int i = 0; i < number_of_rows; i++)
    {
        for (int j = 0; j < number_of_columns; j++)
        {
            data[i][j] = other.data[i][j];
        }
    }
    return *this;
}

// Here is where the machine learning actually starts...
// Abstract class that just creates the basic infrastructure

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}
double relu(double x)
{
    return x > 0 ? x : 0;
}

class Layer
{
  private:
  public:
    virtual Matrix forward(const Matrix &input) = 0;
    virtual Matrix backward(const Matrix &previous_layer_error) = 0;
    virtual ~Layer()
    {
    }
};

class DenseLayer : public Layer
{
  private:
    Matrix *W; // weights
    Matrix *B; // biases
    Matrix *Input;
    Matrix *Z; // OUTPUT Z = W Input + B, Z is before the non linear normalizatrion function like RELU or σ
    Matrix *dJ_dW;
    Matrix *dJ_dB;

  public:
    DenseLayer(int input_size, int output_size)
    {
        W = new Matrix(output_size, input_size);
        B = new Matrix(output_size, 1);
        Input = new Matrix(input_size, 1);
        Z = new Matrix(output_size, 1);
        dJ_dW = new Matrix(output_size, input_size);
        dJ_dB = new Matrix(output_size, 1);
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
    Matrix normalization_funct(double (*func)(double)) const
    {
        Matrix ReturnMatrix(Z->getRows(), Z->getCols());
        for (int i = 0; i < ReturnMatrix.getRows(); i++)
            for (int j = 0; j < ReturnMatrix.getCols(); j++)
                ReturnMatrix.getData()[i][j] = func(Z->getData()[i][j]);
        return ReturnMatrix;
    }
    Matrix forward(const Matrix &input) override
    {
        *Input = input;
        *Z = ((*W) * (*Input)) + (*B);
        return normalization_funct(sigmoid);
    }
    Matrix backward(const Matrix &previous_layer_error) override
    {
        // It is true that delta of the level is the dJ_dB
        *dJ_dB = previous_layer_error.Hadamard(normalization_funct(sigmoid_derivative));
        *dJ_dW = (*dJ_dB) * Input->traspose();
        return (W->traspose() * (*dJ_dB));
    }
};

// Now for the final integration

class NeuralNetwork
{
  private:
    vector<int> topology;
    vector<Layer *> Layer_Vector;

  public:
    NeuralNetwork(const std::vector<int> top) : topology(top)
    {
        for (size_t i = 0; i < topology.size() - 1; i++)
            Layer_Vector.push_back(new DenseLayer(topology[i], topology[i + 1]));
    }
    Matrix forward(Matrix input)
    {
        Matrix current_act = input;
        for (size_t i = 0; i < Layer_Vector.size(); i++)
            current_act = Layer_Vector.at(i)->forward(current_act);
        return current_act;
    }
    void backward(const Matrix &output_error)
    {
        Matrix current_error = output_error;
        for (int i = Layer_Vector.size() - 1; i >= 0; i--)
            current_error = Layer_Vector[i]->backward(current_error);
    }
};
int main()
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

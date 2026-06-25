#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
const double learning_rate = 0.2;

template <typename T> class Matrix
{
  private:
    int number_of_rows, number_of_columns;
    T **data;

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
    T **getData()
    {
        return data;
    }
    T **getDataConst() const
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
    Matrix operator*(const double number)
    {
        Matrix ReturnMatrix(this->number_of_rows, this->number_of_columns);
        for (int i = 0; i < ReturnMatrix.number_of_rows; i++)
            for (int j = 0; j < ReturnMatrix.number_of_columns; j++)
                ReturnMatrix.getData()[i][j] = number * this->getData()[i][j];
        return ReturnMatrix;
    }
    Matrix &operator=(const Matrix<T> &other);
    Matrix operator+(const Matrix<T> &B)
    {
        Matrix<T> ReturnMatrix(this->number_of_rows, this->number_of_columns);
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

template <typename T> Matrix<T>::Matrix(int rows, int columns) : number_of_rows(rows), number_of_columns(columns)
{
    data = new T *[number_of_rows];
    for (int i = 0; i < number_of_rows; i++)
        data[i] = new T[number_of_columns]();
}
template <typename T>
Matrix<T>::Matrix(const Matrix &other)
    : number_of_rows(other.number_of_rows), number_of_columns(other.number_of_columns)
{
    data = new T *[number_of_rows];
    for (int i = 0; i < number_of_rows; i++)
        data[i] = new T[number_of_columns];
    for (int i = 0; i < number_of_rows; i++)
        for (int j = 0; j < number_of_columns; j++)
            data[i][j] = other.getDataConst()[i][j];
}
template <typename T> Matrix<T>::~Matrix()
{
    for (int i = 0; i < number_of_rows; i++)
        delete[] data[i];
    delete[] data;
}
template <typename T> Matrix<T> Matrix<T>::traspose()
{
    Matrix ReturnMatrix(this->getCols(), this->getRows());
    for (int i = 0; i < ReturnMatrix.number_of_rows; i++)
        for (int j = 0; j < ReturnMatrix.number_of_columns; j++)
            (ReturnMatrix.getData())[i][j] = this->getData()[j][i];
    return ReturnMatrix;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix &other)
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

    data = new T *[number_of_rows];
    for (int i = 0; i < number_of_rows; i++)
    {
        data[i] = new T[number_of_columns];
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
    vector<int> topology;
    vector<Layer<T> *> Layer_Vector;

  public:
    NeuralNetwork(const std::vector<int> top) : topology(top)
    {
        for (size_t i = 0; i < topology.size() - 1; i++)
            Layer_Vector.push_back(new DenseLayer<T>(topology[i], topology[i + 1]));
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

// This is the parser part that was a bit vibe coded...
template <typename T>
bool load_mnist_csv(const string &filename, vector<Matrix<T>> &images, vector<Matrix<T>> &labels, int max_rows)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cout << "Error: Could not open file " << filename << endl;
        return false;
    }

    string line;
    if (!getline(file, line))
        return false;

    int row_count = 0;
    while (getline(file, line) && row_count < max_rows)
    {
        stringstream ss(line);
        string token;

        if (getline(ss, token, ','))
        {
            int label_val = stoi(token);
            Matrix<T> target(10, 1);
            target.getData()[label_val][0] = 1.0; // 1.0 στη σωστή θέση, 0.0 παντού αλλού
            labels.push_back(target);
        }

        // Φτιάχνουμε τον Input Vector (784x1) για τα pixels
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
    cout << "Successfully loaded " << row_count << " rows from " << filename << endl;
    return true;
}
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

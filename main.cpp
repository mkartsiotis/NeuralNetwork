#include <iostream>
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
    Matrix operator*(Matrix &B)
    {
        Matrix ReturnMatrix(this->number_of_rows, B.number_of_columns);
        if (B.number_of_rows != this->number_of_columns)
            return ReturnMatrix;
        for (int i = 0; i < ReturnMatrix.number_of_rows; i++)
            for (int j = 0; j < ReturnMatrix.number_of_columns; j++)
            {
                double element = 0;
                for (int k = 0; k < this->number_of_columns; k++)
                    element += (this->getData())[i][k] * (B.getData())[k][j];
                (ReturnMatrix.getData())[i][j] = element;
            }
        return ReturnMatrix;
    }
    Matrix &operator=(const Matrix &other);
    Matrix operator+(Matrix &B)
    {
        Matrix ReturnMatrix(this->number_of_rows, this->number_of_columns);
        if (B.number_of_rows != this->number_of_rows && B.number_of_columns != this->number_of_columns)
            return ReturnMatrix;
        for (int i = 0; i < ReturnMatrix.number_of_rows; i++)
            for (int j = 0; j < ReturnMatrix.number_of_columns; j++)
                (ReturnMatrix.getData())[i][j] = this->getData()[i][j] + B.getData()[i][j];
        return ReturnMatrix;
    }
    Matrix traspose();
    ~Matrix();
};

Matrix::Matrix(int rows, int columns) : number_of_rows(rows), number_of_columns(columns)
{
    data = new double *[number_of_rows];
    for (int i = 0; i < number_of_rows; i++)
        data[i] = new double[number_of_columns];
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

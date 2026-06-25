#ifndef MATRIX_HPP
#define MATRIX_HPP

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
#endif // !MATRIX_HPP

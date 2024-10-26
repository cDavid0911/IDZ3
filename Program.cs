using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Statistics;

var matrix = DenseMatrix.OfArray(new double[,] {
    {12500, 130, 12, 2.3},
    {13700, 120, 10, 1.9},
    {9200, 300, 15, 1.8},
    {11400, 180, 13, 2.1},
    {15800, 150, 14, 2.6},
    {12300, 80, 8, 1.7},
    {16300, 170, 10, 2.4},
    {10200, 210, 11, 1.9},
    {11000, 250, 7, 1.9},
    {12700, 150, 9, 1.7},
    {15000, 90, 4, 2.2},
    {10500, 230, 13, 2.4},
    {17200, 120, 8, 2.3},
    {16000, 110, 9, 2.5},
    {17100, 120, 6, 2.6}
});

var matrixX = DenseMatrix.OfArray(new double[,] {
    {1, 130, 12, 2.3},
    {1, 120, 10, 1.9},
    {1, 300, 15, 1.8},
    {1, 180, 13, 2.1},
    {1, 150, 14, 2.6},
    {1, 80, 8, 1.7},
    {1, 170, 10, 2.4},
    {1, 210, 11, 1.9},
    {1, 250, 7, 1.9},
    {1, 150, 9, 1.7},
    {1, 90, 4, 2.2},
    {1, 230, 13, 2.4},
    {1, 120, 8, 2.3},
    {1, 110, 9, 2.5},
    {1, 120, 6, 2.6}
});


var vectorY = DenseVector.OfArray(new double[]
        {
            12500, 13700, 9200, 11400, 15800,
            12300, 16300, 10200, 11000, 12700,
            15000, 10500, 17200, 16000, 17100
        });

var p = 3;
var n = 15;

// Транспонирование
var transposedMatrixX = matrixX.Transpose();

var XtX = transposedMatrixX * matrixX;

var inverseXtX = XtX.Inverse();

var teta = inverseXtX * transposedMatrixX * vectorY;

var ySKrishkoi = matrixX * teta;

var yMean = vectorY.Mean();

var F = (Calc1(DenseVector.OfVector(ySKrishkoi), yMean) / p) /
        (Calc2(DenseVector.OfVector(ySKrishkoi), vectorY) / (n - p - 1));

var sInSquare = Calc2(vectorY, DenseVector.OfVector(ySKrishkoi)) / (n - p - 1);

var s = Math.Sqrt(sInSquare);

var intervalValue = s * 2.20 * Math.Sqrt(inverseXtX[1,1]);

var interval = DenseVector.OfArray(new double[]
        {
            teta[0] - intervalValue, teta[0] + intervalValue,
        });

var intervalHi = DenseVector.OfArray(new double[]
        {
            sInSquare*(n-p-1)/21.92, sInSquare*(n-p-1)/3.82
        });

var vectorX = DenseVector.OfArray(new double[]
        {
            1, 130, 12, 2.3
        });

var f = vectorX * teta;

var z = vectorX;

var intervalf = DenseVector.OfArray(new double[]
        {
            f - s*2.20*Math.Sqrt((inverseXtX * z) * z ),
            f + s*2.20*Math.Sqrt((inverseXtX * z) * z )
        });

var t1 = teta[0]/(s * Math.Sqrt(inverseXtX[1, 1]));

var R = CalcR(matrix, p, n);

var R00 = CalculateAlgebraicComplement(R, 0, 0);
var R01 = CalculateAlgebraicComplement(R, 0, 1);
var R11 = CalculateAlgebraicComplement(R, 1, 1);
var ro01 = -R01 / (Math.Sqrt(R00 * R11));

// Вывод на консоль
Console.WriteLine("Original Matrix:\n" + matrixX);
Console.WriteLine("\n\n");
Console.WriteLine("Транспонована помножена на початкову\n" + XtX);
Console.WriteLine("\n\n");
Console.WriteLine("Обернена до попередньої\n" + inverseXtX);
Console.WriteLine("\n\n");
Console.WriteLine("Оцінка\n" + teta);
Console.WriteLine("\n\n");
Console.WriteLine("Вектор \n" + ySKrishkoi);
Console.WriteLine("\n\n");
Console.WriteLine("Середнє накопичення\n" + yMean);
Console.WriteLine("\n\n");
Console.WriteLine("Статистика F\n" + F);
Console.WriteLine("\n\n");
Console.WriteLine("Незсунута оцінка дисперсії\n" + sInSquare);
Console.WriteLine("\n\n");
Console.WriteLine("Корінь з неї F\n" + s);
Console.WriteLine("\n\n");
Console.WriteLine("Інтервал для тета1\n" + interval);
Console.WriteLine("\n\n");
Console.WriteLine("Інтервал для дисперсії залишків\n" + intervalHi);
Console.WriteLine("\n\n");
Console.WriteLine("f\n" + f);
Console.WriteLine("\n\n");
Console.WriteLine("Інтервал для функції регресії в зазначеній точці\n" + intervalf);
Console.WriteLine("\n\n");
Console.WriteLine("t1\n" + t1);
Console.WriteLine("\n\n");
Console.WriteLine("Кореляційна матриця R\n" + R);
Console.WriteLine("\n\n");
Console.WriteLine("Алг доповнення R00\n" + R00);
Console.WriteLine("\n\n");
Console.WriteLine("Алг доповнення R01\n" + R01);
Console.WriteLine("\n\n");
Console.WriteLine("Алг доповнення R11\n" + R11);
Console.WriteLine("\n\n");
Console.WriteLine("Вибіркова оцінка частинного коефіцієнту кореляції\n" + ro01);
Console.WriteLine("\n\n");
Console.WriteLine("Статистика t\n" + (ro01 * Math.Sqrt(n-p-1))/Math.Sqrt(1- ro01* ro01));





double Calc1(DenseVector x, double y)
{
    double res = 0;

    foreach (var value in x)
    {
        res += (value - y) * (value - y);
    }
    return res;
}

double Calc2(DenseVector x, DenseVector y)
{
    if (x.Count != y.Count) throw new ArgumentException("wrong input");
    double res = 0;

    for(int i =0; i < x.Count; i++)
    {
        res += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return res;
}

DenseMatrix CalcR(DenseMatrix y, int p, int n)
{
    var x = y.Transpose();
    var res = DenseMatrix.Create(p+1,p+1,0);

    for (int k = 0; k < p + 1; k++)
    {
        for (int j = 0; j < p + 1; j++)
        {
            double val1 = 0;
            double val2 = 0;
            double val3 = 0;
            double val4 = 0;
            double val5 = 0;
            for (int i = 0; i < n; i++)
            {
                val1 += x[k, i] * x[j, i];
                val2 += x[k, i];
                val3 += x[j, i];
                val4 += x[k, i] * x[k, i];
                val5 += x[j, i] * x[j, i];
            }
            res[k, j] = (n * val1 - val2 * val3) / (Math.Sqrt((n * val4 - val2 * val2) * (n * val5 - val3 * val3)));
        }
    }
    return res;
}

static double CalculateAlgebraicComplement(DenseMatrix matrix, int row, int column)
{
    // Проверяем на корректность индексов
    if (row < 0 || row >= matrix.RowCount || column < 0 || column >= matrix.ColumnCount)
        throw new ArgumentOutOfRangeException("Индексы выходят за пределы матрицы.");

    // Вычисляем минор для элемента (row, column)
    double minor = CalculateMinor(matrix, row, column);

    // Возвращаем алгебраическое дополнение
    return ((row + column) % 2 == 0 ? 1 : -1) * minor;
}

static double CalculateMinor(DenseMatrix matrix, int row, int column)
{
    int rows = matrix.RowCount;
    int columns = matrix.ColumnCount;
    var minorMatrix = DenseMatrix.Create(rows - 1, columns - 1, 0);

    for (int i = 0, mi = 0; i < rows; i++)
    {
        for (int j = 0, nj = 0; j < columns; j++)
        {
            if (i != row && j != column)
            {
                minorMatrix[mi, nj++] = matrix[i, j];
                if (nj == columns - 1)
                {
                    nj = 0;
                    mi++;
                }
            }
        }
    }

    return minorMatrix.Determinant(); // Возвращаем определитель минорной матрицы
}
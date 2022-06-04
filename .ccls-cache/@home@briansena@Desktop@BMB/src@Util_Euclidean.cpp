/**
 * @file Euclidean.cpp
 * @version 2.5
 * @date 09/05/2022
 * @author Brian Sena Simons 3ºA-A2
 * @brief Diferentes implementaciones de la distancia Euclidea según necesidades
 * Mejor utilizada con los demás includes como mytools.h
 * @code
 * [...]
    RowVectorXd Weights = VectorXd::Constant(mat.cols(),1);
    min = minEuclideanDistance(Weights,mat.row(0), mat.block(row,0,upper_row - row,mat.cols()),index);
    cout << "Prueba de obtención de clase concreta" << endl;
    cout << "Obteniendo filas tipo g " << endl;
    MatrixXd typeG = getClass(mat,label,'g');
    cout << "Filas: " << typeG.rows() << " columnas: " << typeG.cols()  << endl;
    cout << "###################################" << endl;
 *[...]
 * @endcode
 **/
#include "../tools/Euclidean.h"
#include "../inc/eigen-3.4.0/Eigen/Dense"
#include "../inc/random.hpp"
#include <fstream>
#include <math.h>
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;

/**
 * Usando la librería Eigen sabemos que podemos restarle a una fila un vector
 * de tipe RowVectorXd si activamos el calculo por filas con .rowwise() y además
 * ese resultado podemos concatenarlo en vez de almacenarlo y llamar ahora
 * a .squaredNorm() que calcula la distancia euclidea al cuadrado. Sin embargo,
 * también hemos de decir que la queremos por filas y no por columnas. Luego
 * a continuación hacemos una llamada a minCoeff() que busca el valor mínimo
 * resultante de las operaciones realizadas.
 */
double minEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,Eigen::MatrixXd::Index &maxRow){
    return (data.rowwise() - fixed).rowwise().squaredNorm().minCoeff(&maxRow);
}

/**
 * Hacemos el calculo de la distancia euclidea al cuadrado pero ahora multiplicamos
 * cada parámetro por su respectivo peso que pondera sus características.
 * @see minEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,Eigen::MatrixXd::Index &maxRow)
 */
double minEuclideanDistance(Eigen::RowVectorXd weights,Eigen::RowVectorXd fixed, Eigen::MatrixXd data,Eigen::MatrixXd::Index &maxRow){
    data = (data.rowwise() - fixed).array().square().rowwise() * weights.array();
    return data.rowwise().sum().minCoeff(&maxRow);
}

/**
 * Hacemos el cálculo de la distancia euclidea al cuadrado utilizando la librería
 * Eigen y ese resultado lo almacenamos en un vector resultante y iteramos sobre
 * él en busca del segundo mínimo valor evitando que ese sea justo el de la
 * posición especificada "pos".
 * @see minEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,Eigen::MatrixXd::Index &maxRow)
 */
double ManualEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,unsigned int pos, unsigned int &maxRow){
    RowVectorXd res = (data.rowwise() - fixed).rowwise().squaredNorm();
    float min = 99999;
    for(unsigned int i=0;i<res.cols();i++){
        if(*(res.data()+i)<min && i!=pos){
            min = *(res.data()+i);
            maxRow = i;
        }
    }
    return min;
}

/**
 * Hacemos el cálculo de la distancia euclidea al cuadrado utilizando la librería
 * Eigen ponderado por los pesos característicos calculados
 * y ese resultado lo almacenamos en un vector resultante y iteramos sobre
 * él en busca del segundo mínimo valor evitando que ese sea justo el de la
 * posición especificada "pos".
 * @see ManualEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,unsigned int pos, unsigned int &maxRow);
 */
double ManualEuclideanDistance(Eigen::RowVectorXd weights,Eigen::RowVectorXd fixed, Eigen::MatrixXd data,unsigned int pos, unsigned int &maxRow){
    data = (data.rowwise() - fixed).array().square().rowwise() * weights.array();
    RowVectorXd res = data.rowwise().sum();
    float min = 99999;
    for(unsigned int i=0;i<res.cols();i++){
        if(*(res.data()+i)<min && i!=pos){
            min = *(res.data()+i);
            maxRow = i;
        }
    }
    return min;
}

/**
 * Hacemos el cálculo de la distancia euclidea al cuadrado utilizando la librería
 * Eigen y ese resultado lo almacenamos en un vector resultante y iteramos sobre
 * él en busca del segundo mínimo valor evitando que ese sea justo el de la
 * posición especificada "pos" y verificando que sea de tipo "type" con ayuda
 * del vector de etiquetas "label"
 * @see ManualEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,unsigned int pos, unsigned int &maxRow);
 */
double ManualEuclideanDistanceType(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,
        vector<char> label, char type,unsigned int pos, unsigned int &maxRow){
    RowVectorXd res = (data.rowwise() - fixed).rowwise().squaredNorm();
    float min = 99999;
    for(unsigned int i=0;i<res.cols();i++){
        if(*(res.data()+i)<min && i != pos && type==label[i]){
            min = *(res.data()+i);
            maxRow = i;
        }
    }
    return min;
}

/**
 * Hacemos el cálculo de la distancia euclidea al cuadrado utilizando la librería
 * Eigen multiplicado por el vector de características calculado,
 * y ese resultado lo almacenamos en un vector resultante y iteramos sobre
 * él en busca del segundo mínimo valor evitando que ese sea justo el de la
 * posición especificada "pos" y verificando que sea de tipo "type" con ayuda
 * del vector de etiquetas "label"
 * @see double ManualEuclideanDistanceType(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,
        vector<char> label, char type,unsigned int pos, unsigned int &maxRow);
 */
double ManualEuclideanDistanceType(Eigen::RowVectorXd weights,Eigen::RowVectorXd fixed, Eigen::MatrixXd data,
        vector<char> label, char type, unsigned int pos, unsigned int &maxRow){
    data = (data.rowwise() - fixed).array().square().rowwise() * weights.array();
    RowVectorXd res = data.rowwise().sum();
    float min = 99999;
    for(unsigned int i=0;i<res.cols();i++){
        if(*(res.data()+i)<min && i != pos && type==label[i]){
            min = *(res.data()+i);
            maxRow = i;
        }
    }
    return min;
}

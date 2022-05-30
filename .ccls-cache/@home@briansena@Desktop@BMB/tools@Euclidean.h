/**
 * @file Euclidean.h
 * @version 2.5
 * @date 09/04/2022
 * @author Brian Sena Simons 3ºA-A2
 * @code
 * int main(){
 *  MatrixXd mat(3,2);
 *  mat << 1, 1,
 *         2, 2,
 *         4, 4;
 *
 *  RowVectorXd fil = removeRow(mat,2);
 *  MatriXd::Index pos;
 *  double min = minEuclideanDistance(fil,mat,pos);
 *  cout << "Mínimo vecino por fila en: " << pos << " con valor: " << min << endl;
 * }
 * @endcode
 **/
#ifndef EUCLIDEAN_H
#define EUCLIDEAN_H

#include "../inc/eigen-3.4.0/Eigen/Dense"
#include "../inc/random.hpp"
#include <string.h>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;
/**
 * @brief minEuclideanDistance utiliza la librería Eigen para calcular el
 * menor vecino por fila desde una fila "fixed" NO contenida en data.
 * @param fixed Fila a buscar NO contenida en data.
 * @param data Matrix de datos a verificar vecinos.
 * @param maxRow es el ínidice de la fila más cercana encontrada.
 * @return el valor mínimo.
 **/
double minEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,Eigen::MatrixXd::Index &maxRow);

/**
 * @brief minEuclideanDistance utiliza la librería Eigen para calcular el
 * menor vecino por fila desde una fila "fixed" NO contenida en data multiplicando
 * las distancias por el vector de características que las pondera.
 * @param weights es el vector de pesos característicos a multiplicar
 * @param fixed Fila a buscar NO contenida en data.
 * @param data Matrix de datos a verificar vecinos.
 * @param maxRow es el ínidice de la fila más cercana encontrada.
 * @return el valor mínimo.
 * @see minEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,Eigen::MatrixXd::Index &maxRow);
 **/
double minEuclideanDistance(Eigen::RowVectorXd weigths,Eigen::RowVectorXd fixed, Eigen::MatrixXd data,Eigen::MatrixXd::Index &maxRow);

/**
 * @brief ManualEuclideanDistance no-utiliza completamente la librería Eigen
 * para calcular el menor vecino por fila, ya que ahora permitimos que fixed,
 * esté contenido en data, y lo que hacemos es elegir el segundo mínimo manualmente.
 * @param weights es el vector de pesos característicos a multiplicar
 * @param fixed Fila a buscar contenida en data.
 * @param data Matrix de datos a verificar vecinos.
 * @param pos Fila de la que provee fixed.
 * @param maxRow es el ínidice de la fila más cercana encontrada.
 * @return el valor mínimo.
 * @see minEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,Eigen::MatrixXd::Index &maxRow);
 **/
double ManualEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,unsigned int pos,unsigned int &maxRow);

/**
 * @brief ManualEuclideanDistance no-utiliza completamente la librería Eigen
 * para calcular el menor vecino por fila, ya que ahora permitimos que fixed,
 * esté contenido en data, y lo que hacemos es elegir el segundo mínimo ponderado
 * por el vector de características manualmente.
 * @param weights es el vector de pesos característicos a multiplicar
 * @param fixed Fila a buscar contenida en data.
 * @param data Matrix de datos a verificar vecinos.
 * @param pos Fila de la que provee fixed.
 * @param maxRow es el ínidice de la fila más cercana encontrada.
 * @return el valor mínimo.
 * @see double ManualEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,unsigned int pos,unsigned int &maxRow);
 **/
double ManualEuclideanDistance(Eigen::RowVectorXd weigths,Eigen::RowVectorXd fixed, Eigen::MatrixXd data,unsigned int pos, unsigned int &maxRow);

/**
 * @brief ManualEuclideanDistanceType no-utiliza completamente la librería Eigen
 * para calcular el menor vecino por fila, ya que ahora permitimos que fixed,
 * esté contenido en data, y lo que hacemos es elegir el segundo mínimo
 * perteneciente a una clase específica manualmente.
 * @param weights es el vector de pesos característicos a multiplicar
 * @param fixed Fila a buscar contenida en data.
 * @param data Matrix de datos a verificar vecinos.
 * @param label Vector de etiquetas.
 * @param type Clase a buscar.
 * @param pos Fila de la que provee fixed.
 * @param maxRow es el ínidice de la fila más cercana encontrada.
 * @return el valor mínimo.
 * @see double ManualEuclideanDistance(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,unsigned int pos,unsigned int &maxRow);
 **/
double ManualEuclideanDistanceType(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,vector<char>label, char type, unsigned int pos, unsigned int &maxRow);

/**
 * @brief ManualEuclideanDistanceType no-utiliza completamente la librería Eigen
 * para calcular el menor vecino por fila, ya que ahora permitimos que fixed,
 * esté contenido en data, y lo que hacemos es elegir el segundo mínimo ponderado
 * por el vector de características y perteneciente a una clase específica manualmente.
 * @param weights es el vector de pesos característicos a multiplicar
 * @param fixed Fila a buscar contenida en data.
 * @param data Matrix de datos a verificar vecinos.
 * @param label Vector de etiquetas.
 * @param type Clase a buscar.
 * @param pos Fila de la que provee fixed.
 * @param maxRow es el ínidice de la fila más cercana encontrada.
 * @return el valor mínimo.
 * @see double ManualEuclideanDistanceType(Eigen::RowVectorXd fixed, Eigen::MatrixXd data,vector<char>label, char type, unsigned int pos, unsigned int &maxRow);
 **/
double ManualEuclideanDistanceType(Eigen::RowVectorXd weigths,Eigen::RowVectorXd fixed, Eigen::MatrixXd data,vector<char>label,char type,unsigned int pos, unsigned int &maxRow);

#endif

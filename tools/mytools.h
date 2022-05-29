/**
 * @file mytools.h
 * @version 3.0
 * @date 05/04/2022
 * @author Brian Sena Simons 3ºA-A2
 * @brief Herramientas definidas para la práctica.
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
#ifndef MYTOOLS_H
#define MYTOOLS_H

#include "../inc/eigen-3.4.0/Eigen/Dense"
#include "../inc/random.hpp"
#include <string.h>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;

/**
 * @brief fillRange rellena un vector de valores desde 0 hasta upperlimit;
 * @param toFill es el vector a rellenar
 * @param upperlimit es el valor tope del vector
 **/
void fillRange(vector<int>& toFill,unsigned int upperlimit);

/**
 * @brief shuffleData utiliza el archivo "random.hpp" y la librería Eigen
 * para crear un vector de índices, al cual barajamos y luego utilizamos ese
 * vector para intercambiar posiciones. Dónde el número que aparezca en la posición
 * 0 es el intercambio de 0 con ese número y así.
 * @param mat Matrix a permutar filas.
 * @parm label Vector de etiquetas a intercambiar valores.
 * @param seed Semilla para la función Random
 **/
void shuffleData(MatrixXd& mat,vector<char>& label,long int seed);

/*
 * @brief shuffleFit utiliza el archivo "random.hpp" y la librería Eigen para
 * barajar tanto la matriz "mat" como el vector "fitness" de la misma forma
 * manteniendo las relaciones entre los índices de cada.
 * @param mat Matrix a permutar filas
 * @param fitness Vector de puntación a permutar columnas
 * @param seed Semilla a utilizar para la función Random
 */
void shuffleFit(MatrixXd& mat,RowVectorXd& fitness, long int seed);
void shuffleFit(MatrixXd& mat, MatrixXd& GenData, long int seed);

/*
 * @brief Es la función más trivial de las que tenemos de barajar,
 * dónde el objetivo es apenas permutar las filas de la matriz sin tener
 * nada en cuenta;
 * @param mat Matriz a permutar las filas
 * @param seed Semilla para la función Random
 */
void shuffleOnly(MatrixXd& mat, long int seed);

/**
 * @brief removeRow nos permite quitar una fila de una matrix de entrada (sin
 * modificar la original) y devuelve la matrix resultante modificada.
 * @param matrix Matrix con los datos completos.
 * @param rowToRemove Fila a quitar.
 * @return Matrix modificada resultante
 **/
Eigen::MatrixXd removeRow(Eigen::MatrixXd matrix, unsigned int rowToRemove);

/**
 * @brief removeRow nos permite quitar una columna de una matrix de entrada (sin
 * modificar la original) y devuelve la matrix resultante modificada.
 * @param matrix Matrix con los datos completos.
 * @param colToRemove Columna a quitar.
 * @return Matrix modificada resultante
 **/
Eigen::MatrixXd removeCol(Eigen::MatrixXd matrix, unsigned int colToRemove);

/*
 * @brief Iteramos sobre el vector de puntuación y devolvemos un vector de
 * tamaño size con los índices de los "size" primeros mejores valores.
 * @param Fitness Vector a iterar.
 * @param indexGrid Índices resultantes.
 * @param size Número de posiciones a computar.
 */
void getBest(RowVectorXd Fitness,vector<int>& indexGrid,unsigned int size);
void getBest(MatrixXd GenData,vector<int>& indexGrid,unsigned int size);

/*
 * @brief Calcula el path del ejecutable para poder obtener la dirección de la
 * carpeta results.
 */
string get_selfpath();
/*
 * @brief Simplesmente dibuja la barra de progreso del código.
 */
void progress_bar(float progress);

#endif

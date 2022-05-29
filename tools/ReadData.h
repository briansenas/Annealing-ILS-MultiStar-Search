/**
 * @file ReadData.h
 * @version 2.3
 * @date 09/05/2022
 * @author Brian Sena Simons 3ºA-A2
 * @brief Lectura de datos definidas para la práctica.
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
#ifndef READDATA_H
#define READDATA_H

#include "../inc/eigen-3.4.0/Eigen/Dense"
#include "mytools.h"
#include "../inc/random.hpp"
#include <string.h>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;
/**
 * @brief getClass lo que nos devuelve es una matrix con los datos pertenecientes
 * a una clase específica.
 * @param data Matrix de datos completos.
 * @param label Vector de etiquetas.
 * @param type Clase específica a buscar.
 * @return Matrix con los datos de esa clase
 **/
Eigen::MatrixXd getClass(Eigen::MatrixXd data, vector<char> label, char type);

Eigen::MatrixXd getClassLabelled(Eigen::MatrixXd data, vector<char> Label, vector<char>& newLabel, char type);

/**
 * @brief getFold divide la matrix de datos "data" en dos matrices "tranning" y
 * "test" con los datos pertenecientes a uno de los 5-folds que tenemos de
 * desplazar un 20% al grupo de test sobre el conjunto total
 * @param data Matrix de datos completos.
 * @param label Vector de etiquetas.
 * @param traning Matrix con los datos para entrenar.
 * @param Tlabel Vector con las etiquetas de entreno.
 * @param test Matrix con los datos para el test.
 * @param Ttlabel Vector con las etiquetas de test.
 * @param num Fold a utilizar (0:(0-80%;80%-100%), 1:(0:60%+80%-100%;60%-80%)...)
 **/
void getFold(Eigen::MatrixXd data, vector<char> Label, Eigen::MatrixXd &training, vector<char> &TLabel,
        Eigen::MatrixXd &test, vector<char>& TtLabel, unsigned int num=1);

/**
 * @brief getFoldbyLoop divide la matrix de datos "data" en dos matrices "tranning" y
 * "test" con los datos pertenecientes a uno de los 5-folds que tenemos de
 * desplazar un 20% al grupo de test sobre el conjunto total. A diferencia de
 * getFold() intenta utilizar el "for-loop" de las etiquetas para asignar las filas.
 * Sin embargo posee peor rendimiento.
 * @param data Matrix de datos completos.
 * @param label Vector de etiquetas.
 * @param traning Matrix con los datos para entrenar.
 * @param Tlabel Vector con las etiquetas de entreno.
 * @param test Matrix con los datos para el test.
 * @param Ttlabel Vector con las etiquetas de test.
 * @param num Fold a utilizar (0:(0-80%;80%-100%), 1:(0:60%+80%-100%;60%-80%)...)
 * @see getFold()
 **/
void getFoldbyLoop(Eigen::MatrixXd data, vector<char> Label, Eigen::MatrixXd &training, vector<char> &TLabel,
        Eigen::MatrixXd &test, vector<char>& TtLabel, unsigned int num);

/*
 * La idea es obtener una distribución balanceada de datos para el
 * conjunto de entreno / test.
 * @param group1 Grupo con características de clase 1;
 * @param label1 Vector con las etiquetas de clase 1;
 * @param group2 Grupo con características de clase 2;
 * @param label2 Vector con las etiquetas de clase 2;
 * @param training Matrix resultante para el conjunto de entreno.
 * @param Tlabel Vector con las etiquetas resultante para el entreno
 * @param test Matrix resultante con el conjunto de test,
 * @param Ttlabel Vector con las etiquetas resultantes para el test.
 * @param num Número de correspondencia de fold.
 * @param seed Semilla para el barajado final
 *@see getFold();
 **/
void getBalancedFold(Eigen::MatrixXd group1, vector<char> label1,Eigen::MatrixXd group2,
        vector<char> label2, Eigen::MatrixXd &training, vector<char> &TLabel,
        Eigen::MatrixXd &test, vector<char>& TtLabel, unsigned int num, long int seed);

/**
 * @brief readValues es la función utilizada para leer los archivos ".arrf" de
 * la práctia en una Matrix de la libreria "Eigen" y un vector de etiquetas "Label".
 * @param filename Nombre del archivo que vamos a leer
 * @param label Vector de etiquetas a rellenar.
 * @return Matrix con todos los valores.
 **/
Eigen::MatrixXd readValues(string filename, vector<char>& label);
#endif

/**
 * @file ReadData.cpp
 * @version 2.5
 * @date 09/05/2022
 * @author Brian Sena Simons 3ºA-A2
 * @brief Funciones necesarias para la lectura y manejo de los datos y etiquetas.
 * Es mejor utilizada con los demás includes como mytools.h
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
#include "../tools/ReadData.h"
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
 * La idea es iterar sobre el vector de etiquetas y cuando su valor coincida con
 * el especificado por "type" introducimos en la matrix "res" la fila correspondiente
 * para así calcular la matrix con apenas un tipo de clase.
 */
Eigen::MatrixXd getClass(Eigen::MatrixXd data, vector<char> Label, char type){
    Eigen::MatrixXd res;
    res.resize(1,data.cols());
    for(unsigned int i=0,size=Label.size();i<size;i++){
        if(Label[i] == type){
            res.row(res.rows()-1) = data.row(i);
            res.conservativeResize(res.rows()+1,res.cols());
        }
    }
    res.conservativeResize(res.rows()-1,res.cols());
    return res;
}

Eigen::MatrixXd getClassLabelled(Eigen::MatrixXd data, vector<char> Label, vector<char>& newLabel, char type){
    Eigen::MatrixXd res;
    newLabel.clear();
    res.resize(1,data.cols());
    for(unsigned int i=0,size=Label.size();i<size;i++){
        if(Label[i] == type){
            res.row(res.rows()-1) = data.row(i);
            res.conservativeResize(res.rows()+1,res.cols());
            newLabel.push_back(type);
        }
    }
    res.conservativeResize(res.rows()-1,res.cols());
    return res;
}
/**
 * La idea es calcular el 20% de la Matrix total y luego con ese valor ir
 * estableciendo el "boundary" superior a copiar para la primera (o toda la parte
 * del dataset de entreno cuando num=0, boundary = 80%) y luego copiar lo que
 * sigue a boundary hasta un 20% más (boundary + size) en test y luego copiar devuelta
 * lo que queda en tranning.
 * Utilizamos la librería de Eigen para un mejor rendimiento.
 */
void getFold(Eigen::MatrixXd data, vector<char> Label, Eigen::MatrixXd &training, vector<char> &TLabel,
        Eigen::MatrixXd &test, vector<char>& TtLabel, unsigned int num){
    unsigned int size = round(data.rows()*0.2);
    TLabel.clear(); TtLabel.clear();
    switch(num){
        case 0:{
                training = data.block(0,0,data.rows()-size,data.cols());
                TLabel = vector<char>(Label.begin(),Label.begin()+data.rows()-size);
                test = data.block(data.rows()-size,0,size,data.cols());
                TtLabel = vector<char>(Label.begin()+data.rows()-size, Label.end());
                break;
        }
        default:
            {
                unsigned int boundary = data.rows()-(num+1)*size;
                if(boundary<(size/2) or boundary>=data.rows())
                    boundary=0;
                for(unsigned int i=0;i<Label.size();i++){
                    if(i<boundary or i>=boundary+size)
                        TLabel.push_back(Label[i]);
                    else
                        TtLabel.push_back(Label[i]);
                }
                if(boundary!=0){
                    training.resize(data.rows()-size,data.cols());
                    training.block(0,0,boundary,data.cols()) = data.block(0,0,boundary,data.cols());
                    training.block(boundary,0,data.rows()-(boundary+size),data.cols()) = data.block(boundary+size,0,data.rows()-(boundary+size),data.cols());
                    test = data.block(boundary,0,size,data.cols());
                }else{
                    test = data.block(0,0,size,data.cols());
                    training = data.block(size,0,data.rows()-size,data.cols());
                }

                break;
            }
    }
}


/**
 * La idea es calcular el 20% de la Matrix total y luego con ese valor ir
 * estableciendo el "boundary" superior a copiar para la primera (o toda la parte
 * del dataset de entreno cuando num=0, boundary = 80%) y luego copiar lo que
 * sigue a boundary hasta un 20% más (boundary + size) en test y luego copiar devuelta
 * lo que queda en tranning.
 * Pero en este caso no utilizamos las facilidades de la librería Eigen si no
 * que apenas calculamos el tamaño total de las matrices a rellenar y vamos
 * iterando a la vez que creamos las etiquetas adecuadas.
 */
void getFoldbyLoop(Eigen::MatrixXd data, vector<char> Label, Eigen::MatrixXd &training, vector<char> &TLabel,
        Eigen::MatrixXd &test, vector<char>& TtLabel, unsigned int num){
    unsigned int size = round(data.rows()*0.2), boundary = data.rows()-(num+1)*size;
    unsigned int Tcont = 0, Ttcont=0;
    training.resize(data.rows()-size,data.cols());
    test.resize(size,data.cols());
    TLabel.clear(); TtLabel.clear();
    if(boundary<=(size/2) or boundary>=data.rows()){
        boundary=0;
    }
    for(unsigned int i=0;i<Label.size();i++){
        if(i<boundary or i>=boundary+size){
            training.row(Tcont++) = data.row(i);
            TLabel.push_back(Label[i]);
        }else{
            test.row(Ttcont++) = data.row(i);
            TtLabel.push_back(Label[i]);
        }
    }
}

/**
 * Hace lo mismo que getFold() pero con un equilibrado de clase de por medio;
 * Se utiliza getClass() para obtener matrices correspondente a cada clase,
 * se pasa a este método y lo para cada conjunto utiliza getFold() para obtener
 * su correspondente 80/20 y luego lo añade todo secuencialmente en el conjunto
 * de entreno y testeo que luego a final son barajados.
 */

void getBalancedFold(Eigen::MatrixXd group1, vector<char> label1,Eigen::MatrixXd group2,
        vector<char> label2, Eigen::MatrixXd &training, vector<char> &TLabel,
        Eigen::MatrixXd &test, vector<char>& TtLabel, unsigned int num, long int seed){
    TLabel.clear();
    TtLabel.clear();

    MatrixXd train1, test1;
    vector<char> tlab1, ttlab1;

    MatrixXd train2, test2;
    vector<char> tlab2, ttlab2;

    getFold(group1,label1,train1,tlab1,test1,ttlab1,num);
    getFold(group2,label2,train2,tlab2,test2,ttlab2,num);

    training.resize(train1.rows()+train2.rows(),group1.cols());
    training << train1, train2;
    test.resize(test1.rows()+test2.rows(), group1.cols());
    test << test1, test2;

    TLabel.insert(TLabel.begin(), tlab1.begin(), tlab1.end());
    TLabel.insert(TLabel.end(), tlab2.begin(), tlab2.end());
    TtLabel.insert(TtLabel.begin(), ttlab1.begin(), ttlab1.end());
    TtLabel.insert(TtLabel.end(), ttlab2.begin(), ttlab2.end());

    shuffleData(training, TLabel,seed);
    shuffleData(test, TtLabel, seed);
}

/**
 * Leemos y utilizamos los "@attribute" del archivo para calcular el tamaño de
 * las columnas de la matrix a rellenar. Luego al leer "@data" lo que haremos
 * es empezar el bucle de rellenado. Dónde en este lo que hacemos es leer
 * un número y una coma, insertar el número en un vector hasta que se alcanze
 * la cantidad de valores calculadas. Luego lo insertamos en la última fila
 * de la matrix y la incrementamos de tamaño para la siguiente iteración. Así
 * hasta que terminemos de leer el archivo.
 **/
Eigen::MatrixXd readValues(string filename, vector<char> & label){
    ifstream file;
    file.open(filename);

    int attr = 0;
    double number;
    char delimiter;
    int cont = 0;
    Eigen::MatrixXd mat;
    Eigen::VectorXd vec;
    if(file.is_open()){
        string tmpString;
        // Contar
        while((file >> tmpString) and tmpString != "@data"){
            if(tmpString=="@attribute"){
                getline(file,tmpString);
                attr++;
            }
        }
        attr--;
        mat.resize(1,attr);
        vec.resize(attr);
        // Leer
    double highest = -9999;
    double lowest = 9999;
        while(!file.eof()){
            if(cont<attr){
                file >> number >> delimiter;
                if(number > highest){
                    highest = number;
                }
                if(number < lowest){
                    lowest = number;
                }
                vec[cont] = number;
            }else{
                cont = -1;
                for(size_t i=0, size = vec.size();i<size;i++)
                    *(vec.data() + i) = (*(vec.data() + i) - lowest) / (highest - lowest);
                lowest = 9999; highest = -9999;
                mat.row(mat.rows()-1) = vec;
                mat.conservativeResize(mat.rows()+1,mat.cols());
                char tag; file >> tag;
                label.push_back(tag);
            }
            cont++;
        }
        mat.conservativeResize(mat.rows()-1,mat.cols());
    }
    file.close();
    return mat;
}


#include "../inc/eigen-3.4.0/Eigen/Dense"
// From https://github.com/effolkronium/random
#include "../inc/random.hpp"

#include "../tools/mytools.h"
#include "../tools/Euclidean.h"
#include "../tools/ReadData.h"
#include "../tools/Genetics.h"
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <fstream>
#include <unistd.h>


using namespace std;
using namespace Eigen;
using namespace std::chrono;
using Random = effolkronium::random_static;


int main(int argc, char** argv){
    if(argc<=6) {
        cerr << "[ERROR]: Couldn't resolve file name;" << endl;
        cerr << "[EXECUTION]: ./main (filename) (label1) (label2) (0-print,1-write) (seed)[int] (0-Normal, 1-ShuffleData,2-BalanceData)" << endl;
        exit(-1);
    }
    string filename = argv[1];
    char type1 = *argv[2];
    char type2 = *argv[3];
    int streambus = atoi(argv[4]);
    long int seed = atol(argv[5]);
    int shuffle = atoi(argv[6]);
    srand(seed);
    Random::seed(seed);
    bool printing = (streambus>=1)?false:true;
    ofstream plot,myfile;
    string writefile = "", plot_path = "", output;
    string path = get_selfpath();
    path = path.substr(0,path.find_last_of("/\\") + 1);
    if(streambus>=1) {
        //https://www.codegrepper.com/code-examples/cpp/c%2B%2B+get+filename+from+path
        // get filename
        std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
        // remove extension from filename
        std::string::size_type const p(base_filename.find_last_of('.'));
        std::string file_without_extension = base_filename.substr(0, p);

        string datafilename = "ILS_"+file_without_extension+to_string(seed)+"_"+to_string(shuffle);
        writefile = path+"../results/"+datafilename;
        writefile += ".txt";
        myfile.open(writefile,ios::out|ios::trunc);
        if(!myfile.is_open()){
            cerr << "[ERROR]: Couldn't open file, printing enabled" << endl;
            printing = true;
        }
    }
    filename = path+"../datos/"+filename;

    vector<char> label;
    MatrixXd allData = readValues(filename,label);

    std::normal_distribution<double> distribution(0.0, sqrt(0.4));
    int cols = allData.cols();
    unsigned int iter = 0, maxIter = 15, max_evaluations = 1000, eval_num;
    unsigned int maxTilBetter = 20*cols, mutaciones = 0.1*cols, i;

    float alpha = 0.5;
    vector<float> fitness;
    high_resolution_clock::time_point momentoInicio, momentoFin;
    milliseconds tiempo;

    MatrixXd data, test, group1, group2;
    vector<char> Tlabel, Ttlabel, label_group1, label_group2;
    RowVectorXd Solution(cols), NewSolution(cols), BestSolution(cols), score(2),best_score(2);
    vector<int> indexGrid;
    fillRange(indexGrid,allData.cols());
    /// Inicializamos todas las variables que vamos a necesitar para almacenar información
    if(shuffle==1){
        cout << "[WARNING]: Data has been shuffled; " << endl;
        if(streambus>=1)
            myfile << "[WARNING]: Data has been shuffled; " << endl;
        shuffleData(allData,label,seed);
    }
    if(shuffle==2){
        cout << "[WARNING]: Data has been balanced and shuffled (inevitably); " << endl;
        if(streambus>=1)
            myfile << "[WARNING]: Data has been balanced and shuffled (inevitably); " << endl;
        group1 = getClassLabelled(allData,label, label_group1, type1);
        group2 = getClassLabelled(allData,label, label_group2, type2);
    }

    if(myfile.is_open()){
        myfile << " ### Algoritmo Búsqueda Local Reiterada ###\n ";
        myfile << "F\tclasific\treducir \tfitness \ttime\n";
    }

    for(int x=0,folds=5;x<folds;x++){
        if(shuffle!=2)
            getFold(allData,label,data,Tlabel,test,Ttlabel,x);
        else
            getBalancedFold(group1,label_group1,group2,label_group2,data,Tlabel, test, Ttlabel,x,seed);

        momentoInicio = high_resolution_clock::now();
        Solution = (RowVectorXd::Random(cols) + RowVectorXd::Constant(cols,1))/2.0;
        BestSolution = LocalSearch(data,Tlabel, Solution, eval_num,
                max_evaluations,maxTilBetter, fitness, alpha);
        best_score = get1Fit(data,Tlabel,BestSolution,alpha);
        for(iter=1;iter<maxIter;iter++){
            NewSolution = BestSolution;
            Random::shuffle(indexGrid);
            for(i=0;i<mutaciones;i++){
                NewSolution[indexGrid[i]] += Random::get(distribution);
            }
            NewSolution = LocalSearch(data,Tlabel, NewSolution, eval_num,
                    max_evaluations,maxTilBetter, fitness, alpha);
            score = get1Fit(data,Tlabel,NewSolution);
            if(best_score.sum()<score.sum()){
                BestSolution = NewSolution;
                best_score = score;
            }
            fitness.clear();
            progress_bar(float((x*maxIter+iter)) / float((folds*maxIter)));
        }
        momentoFin = high_resolution_clock::now();
        tiempo = duration_cast<milliseconds>(momentoFin - momentoInicio);
        best_score = get1Fit(test,Ttlabel, BestSolution,alpha);
        if(printing){
            cout << "###################################\n" ;
            cout << "[BEST FITNESS]: " << best_score(0)/alpha << "\t" << best_score(1)/(1-alpha) << "\t" << best_score.sum() << "\t";
            cout << tiempo.count() << endl;
        }else{
            output = to_string(x) + "\t" + to_string(best_score(0)/alpha)
                    + "\t" +to_string(best_score(1)/(1-alpha)) + "\t" +
                    to_string(best_score.sum()) + "\t" + to_string(tiempo.count()) + "\n";
            myfile << std::setw(30) << output;
        }
    }
    cout << endl;
    return 0;
}

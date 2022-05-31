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

void SimulatedAnnealing(MatrixXd data, vector<char>Tlabel, RowVectorXd& NewSolution,
RowVectorXd& BestSolution, RowVectorXd& best_score, float max_evaluations,
float max_vecinos=0,float max_exitos=0,float mu=0.3,float phi=0.3);

int main(int argc, char** argv){
    if(argc<=7) {
        cerr << "[ERROR]: Couldn't resolve file name;" << endl;
        cerr << "[EXECUTION]: ./main (filename) (label1) (label2) (0-print,1-write) (seed)[int] (0-Normal, 1-ShuffleData,2-BalanceData) (0=LS,1=SA)" << endl;
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
    int searchtype  = atoi(argv[7]);
    if(searchtype==0)
        cout << "[WARNING]: LocalSearch method was choosen" << endl;
    else
        cout << "[WARNING]: SimulatedAnnealing method was choosen" << endl;
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

        string datafilename = "ILS_"+file_without_extension+to_string(seed)+"_"+to_string(shuffle)+(((searchtype==0)?"LS":"SA"));
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
    RowVectorXd Solution(cols), NewSolution(cols), BestSolution(cols), SABest(cols), score(2),best_score(2),sabest(2),old_score(2);
    vector<int> indexGrid;
    fillRange(indexGrid,allData.cols());

    unsigned int max_vecinos = 10*cols, max_exitos = 0.1*max_vecinos;
    float mu = 0.3, phi=0.3;

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
        if(searchtype==0){
            BestSolution = LocalSearch(data,Tlabel, Solution, eval_num,
                    max_evaluations,maxTilBetter, fitness, alpha);
            best_score = get1Fit(data,Tlabel,BestSolution,alpha);
        }else{
            SimulatedAnnealing(data,Tlabel,Solution, BestSolution,
                    best_score, max_evaluations,max_vecinos,max_exitos,mu,phi);
        }
        for(iter=1;iter<maxIter;iter++){
            NewSolution = BestSolution;
            Random::shuffle(indexGrid);
            for(i=0;i<mutaciones;i++){
                NewSolution[indexGrid[i]] += Random::get(distribution);
            }
            if(searchtype==0){
                fitness.clear();
                NewSolution = LocalSearch(data,Tlabel, NewSolution, eval_num,
                        max_evaluations,maxTilBetter, fitness, alpha);
                score = get1Fit(data,Tlabel,NewSolution);
                if(best_score.sum()<score.sum()){
                    BestSolution = NewSolution;
                    best_score = score;
                }
            }else{
                SimulatedAnnealing(data,Tlabel,NewSolution, BestSolution,
                        best_score, max_evaluations,max_vecinos,max_exitos,mu,phi);
            }
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

void SimulatedAnnealing(MatrixXd data, vector<char>Tlabel, RowVectorXd& NewSolution,
RowVectorXd& BestSolution, RowVectorXd& best_score, float max_evaluations,
float max_vecinos,float max_exitos,float mu,float phi){

    unsigned int cols = NewSolution.cols();
    if(max_vecinos==0)
        max_vecinos = 10*cols;
    if(max_exitos==0)
        max_exitos = 0.1*max_vecinos;


    std::normal_distribution<double> distribution(0.0, sqrt(0.4));

    float T_0=0.1, T_f=0.001;
    RowVectorXd score(2), old_score(2), sabest(2),
                Solution(cols),SABest(cols);

    old_score = get1Fit(data,Tlabel,NewSolution);
    T_0 = ( mu * old_score.sum()) / (-1*log(phi) );
    if(T_0 == 0)
        T_0 = 0.1;
    if(T_0 <= T_f)
        T_f = T_0 * pow(10,-3);

    float M = max_evaluations/max_vecinos;
    float beta = (T_0 - T_f) / (M*T_0*T_f),diff;
    unsigned int vecinos = 0, exitos = 0, evaluations=0, index;
    SABest = Solution = NewSolution;
    sabest = old_score;
    evaluations = 1;
    while(evaluations<=max_evaluations){
        vecinos = exitos = 0;
        while(vecinos<=max_vecinos && exitos<=max_exitos && evaluations<=max_evaluations){
            index = Random::get<unsigned>(0,cols-1);
            // Apply move
            NewSolution[index] += Random::get(distribution);
            score = get1Fit(data,Tlabel,NewSolution,0.5);
            vecinos++;evaluations++;
            diff = old_score.sum()-score.sum();
            if((diff<0.0) || (Random::get(0.0,1.0) <= 1.0/exp(diff/T_0))){
                Solution[index] = NewSolution[index];
                old_score = score;
                vecinos = 0;
                exitos++;
                if(sabest.sum() < score.sum()){
                    SABest = Solution;
                    sabest = score;
                }
            }
            NewSolution[index] = Solution[index];
        }// END WHILE2
        T_0 = (T_0) / (1+beta*T_0);
        if(exitos==0) break;
    }
    if(best_score.sum()<sabest.sum()){
        BestSolution = SABest;
        best_score = sabest;
    }
}

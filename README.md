# ** _Brian Sena Simons 3ºA - A2_ **
# Práctica 3 - MetaHeurística

### - Utilize el "cmake CMakeLists.txt && make" para compilar;
### - Los algoritmos pueden ser ejecutados utilizando los scripts en ./scripts {Ej. runAll.sh}
- Para los scripts varian:
    - ILS -> Es necesario tipo de salida 0=imprimir,1=guardar\_en\_resultados, una semilla y si barajamos los datos(0=normal,1=barajado,2=balanceado)
    - Análogamente para los demás scripts.

### - Los resultados se guardan en ./results;
## Ejecución individual:
### Algoritmo ILS:
- ./ILS (filename) (label1)\[char\] (label2)\[char\] (0=Print/1=WriteFile/2=Write+Plot\_data) (seed)[-∞,∞]
(0=No/1=Shuffle/2=Balanced) (0=LocalSearch,1=SimulatedAnnealing)
    - Arg1 = filename -> nombre del archivo de datos.
    - Arg2 = label1 -> etiqueta del primer grupo del conjunto de datos de tipo char.
    - Arg3 = label2 -> etiqueta del segundo grupo del conjunto de datos de tipo char.
    - Arg4 = Bus -> 0 es imprimir por pantalla, 1 es escribir a ./results y 2 es escribir también a ./resutls/plots.
    - Arg5 = seed -> semilla para el generador aleatorio.
    - Arg6 = Preprocesar los datos -> 0 es Normal, 1 es Barajado y 2 es equilibrio de clase.
    - Arg7 = Searchtype -> 0 es utilizar la búsqueda local, 1 es utilizar enfrimiamiento simulado.

- Análogamente para los demás algoritmos con excepción del argumento 7.

## Descripción breve del Problema
La idea es comparar distintos tipos de algoritmos para clasificar datos pertenecientes
a una base de datos públicas que nos provee el profesorado. Partiremos primero
de la implementación del típico algoritmo de clasficación K-NN dónde K representa
el número de vecinos a mirar y la idea conceptual es buscar los K vecinos más
cercanos para realizar una predicción sobre que clase pertenece el objeto a predecir.

Una vez implementado el algoritmo 1NN intentaremos mejorar el porcentaje de aciertos
utilizando técnicas de ponderación de características mediante un vector de pesos.
El grueso de la práctica está en el cálculo de esos pesos. En esta parte de la
práctica compararemos los algoritmos empleados anteriormente (1NN, Greedy,
búsqueda local) con unas variaciones que exploran las características de la búsqueda local.

Tendremos la versión del enfriamiento simulado que se traduce en un comportamiento
explorativo incrementado respecto a la búsqueda local que se degrada en el tiempo
con intención de converger al máximo global con mayor facilidad al haber explorado
más la función. (Simulado las partículas en caliente). Luego tendremos versiones
que abarcan la característica que define el punto de partida de la búsqueda local,
por ejemplo la búsqueda multi-arrance básica, y el descenso iterativo de la mejor
solución mutada.


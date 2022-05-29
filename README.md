# ** _Brian Sena Simons 3ºA - A2_ **
# Práctica 2 - MetaHeurística

### - Utilize el "cmake CMakeLists.txt && make" para compilar;
### - Los algoritmos pueden ser ejecutados utilizando los scripts en ./scripts {Ej. runAll.sh}
- Para los scripts varian:
    - EST -> Es necesario una semilla, si barajamos los datos(0-2)[ARG6] y el tipo de cruce(0-1)[ARG7]
    - GEN -> Es necesario una semilla, si barajamos los datos(0-2)[ARG6] y el tipo de cruce(0-1)[ARG7] y selección(0-2)[ARG9].
- Si se quiere ejecutar con búsqueda local es necesario añadir cada cuantas generaciones, el porcentage de población de 0.0 al 1.0 y si es 0=aleatorio 1=LosMejores. (ARG11-13).
- Ejecutar un ./runAGEST-all.sh o el otro tarda alrededor de 4 minutos.

        -Ejemplo estacionario sin barajar con cruce artimético:  ./runAGEST-all.sh 150421 0 1
        -Ejemplo generacional sin barajar con cruce artimético de los mejores padres:  ./runAGGEN-all.sh 150421 0 1 2
        -Ejemplo estacionario sin barajar con cruce artimético memético:  ./runAGEST-ls-all.sh 150421 0 1 10 0.1 0
        -Ejemplo generacional sin barajar con cruce artimético de los mejores padres memético:  ./runAGGEN-ls-all.sh 150421 0 1 2 10 0.1 0

### - Los resultados se guardan en ./results;
## Ejecución individual:
### Algoritmo Generacional:
- ./AGGEN (filename) (label1)\[char\] (label2)\[char\] (0=Print/1=WriteFile/2=Write+Plot\_data) (seed)[-∞,∞]
(0=No/1=Shuffle/2=Balanced) (0=BLX/1=ARITHMETIC) (POP.SIZE)\[0,∞\] (0=RandomOnly,1=RandomKeepBestCross/1=TopKeepBestCross) (0=No/1=LocalSearch)
{LOCALSEARCH OPTIONAL: (HowOften)\[0,inf\] (POP.Percentage)\[0.0,1.0\] (0=RandomSearch/1=OnlyBestSearch)};

    - Arg1 = filename -> nombre del archivo de datos.
    - Arg2 = label1 -> etiqueta del primer grupo del conjunto de datos de tipo char.
    - Arg3 = label2 -> etiqueta del segundo grupo del conjunto de datos de tipo char.
    - Arg4 = Bus -> 0 es imprimir por pantalla, 1 es escribir a ./results y 2 es escribir también a ./resutls/plots.
    - Arg5 = seed -> semilla para el generador aleatorio.
    - Arg6 = Preprocesar los datos -> 0 es Normal, 1 es Barajado y 2 es equilibrio de clase.
    - Arg7 = Cruce -> 0 es el cruce BLX-alpha y 1 es el cruce Aritmético
    - Arg8 = Pop. Size -> Tamaño de la población.
    - Arg9 = Quiénes cruzan -> 0 cruzamos aleatoriamente y lo que falte por llenar de la población será los que no cruzaron,
            1 cruzamos aleatoriamente y nos quedamos también con el mejor padre, 1 cruzamos apenas lo mejores padres.
            (TODOS MANTIENEN EL ELITISMO).
    - Arg10 = Búsqueda Local -> 0 es sin búsqueda y 1 es con búsqueda.
    - Argumentos opcionales:
        - Arg11 = HowOften -> Cada cuántas generaciones hacemos la búsqueda, Arg10 tiene que ser 1.
        - Arg12 = Pop.Percentage -> A qué porcentaje de la población le aplico la búsqueda.
        - Arg13 = A quiénes -> 0 es búsqueda a los seleccionados aleatoriamente y 1 a los mejores.

- ./AGEST (filename) (label1)\[char\] (label2)\[char\] (0=Print/1=WriteFile/2=Write+Plot\_data) (seed)[-∞,∞]
(0=No/1=Shuffle/2=Balanced) (0=BLX/1=ARITHMETIC) (POP.SIZE)\[0,∞\] (0=No/1=LocalSearch)
{LOCALSEARCH OPTIONAL: (HowOften)\[0,inf\] (POP.Percentage)\[0.0,1.0\] (0=RandomSearch/1=OnlyBestSearch)};

    - Es lo mismo pero sin el ARG9, ya que se elige dos padres al azar y se cruzan directamente;

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
búsqueda local) con unas variaciones genéticas y meméticas.

La primera de ella consiste en generar toda una nueva población de datos a
partir de soluciones aleatorias utilizando operadores de cruces como lo
es el BLX-alpha, que calcula un intervalo de valores a generar para explorar
el vecindario, y el cruce aritmético que es una media ponderada de la característica
a calcular (Columna o gen en concreto al que se aplica). Luego aplicaremos una
mutación con una Probabilidad de 0.1 a uno o varios genes de una solución.

La segunda implemetación consiste en generar dos nuevas soluciones a partir de
dos padres aleatorios, mutarlas y que compitan para entrar de vuelta a la población
original, es decir, deben de tener mejor valor de función.

La última implementación, la versión memética, consiste en submeter esos algoritmos
a la búsqueda local cada cierto número de generación y observar el comportamiento.
Una vez desarrollado toda la práctica y obtengamos todos los datos podemos
proceder a realizar un análisis profundo de las diferencias.

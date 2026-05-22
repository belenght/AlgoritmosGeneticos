import random
import matplotlib.pyplot as plt


#  CONSTANTES DEL PROBLEMA
BITS          = 30           # longitud de cada cromosoma
COEF          = 2**30 - 1    # coeficiente de la función objetivo
TAM_POBLACION = 10           # cantidad de individuos
PC            = 0.75         # probabilidad de crossover
PM            = 0.05         # probabilidad de mutación


# ============================================================
#  PASO 1 — CREAR LA POBLACIÓN INICIAL
# ============================================================

def crear_poblacion():
    """Genera TAM_POBLACION cromosomas al azar, cada uno con BITS bits."""
    poblacion = []
    for _ in range(TAM_POBLACION):
        cromosoma = [random.randint(0, 1) for _ in range(BITS)]
        poblacion.append(cromosoma)
    return poblacion


# ============================================================
#  PASO 2 — EVALUAR LA POBLACIÓN
# ============================================================

def binario_a_decimal(cromosoma):
    """Convierte una lista de bits a entero decimal."""
    return int("".join(str(b) for b in cromosoma), 2)


def funcion_objetivo(x):
    """f(x) = (x / coef)^2  →  siempre entre 0 y 1."""
    return (x / COEF) ** 2


def evaluar_poblacion(poblacion):
    """
    Calcula para cada cromosoma:
      - x     : valor decimal
      - f(x)  : valor de la función objetivo
      - fitness: proporción respecto a la suma total de f(x)
    Retorna listas paralelas: valores_x, valores_fx, fitness
    """
    valores_x  = [binario_a_decimal(c) for c in poblacion]
    valores_fx = [funcion_objetivo(x) for x in valores_x]

    suma_total = sum(valores_fx)
    # Evitamos división por cero (caso rarísimo donde todos dan 0)
    if suma_total == 0:
        fitness = [1 / TAM_POBLACION] * TAM_POBLACION
    else:
        fitness = [fx / suma_total for fx in valores_fx]

    return valores_x, valores_fx, fitness


# ============================================================
#  PASO 3 — SELECCIÓN POR RULETA
# ============================================================

def seleccion_ruleta(poblacion, fitness):
    """
    Construye la ruleta acumulada y la gira TAM_POBLACION veces.
    Los cromosomas con más fitness tienen más chances de ser elegidos.
    """
    # Armamos la ruleta acumulada
    acumulado = []
    suma = 0
    for f in fitness:
        suma += f
        acumulado.append(suma)

    padres = []
    for _ in range(TAM_POBLACION):
        r = random.random()
        for i, limite in enumerate(acumulado):
            if r <= limite:
                # Copiamos el cromosoma para no modificar el original
                padres.append(list(poblacion[i]))
                break

    return padres


# ============================================================
#  PASO 4 — CROSSOVER DE 1 PUNTO
# ============================================================

def crossover(padres):
    """
    Toma los padres de a pares y los cruza con probabilidad PC.
    Crossover de 1 punto: se elige un punto de corte al azar,
    los hijos heredan un segmento de cada padre.
    """
    hijos = []

    # Recorremos de a pares: (0,1), (2,3), (4,5), ...
    for i in range(0, TAM_POBLACION, 2):
        padre1 = list(padres[i])
        padre2 = list(padres[i + 1])

        if random.random() < PC:
            # Elegimos el punto de corte al azar (entre 1 y BITS-1)
            punto = random.randint(1, BITS - 1)

            # Hijo 1: primeros 'punto' genes del padre1 + resto del padre2
            hijo1 = padre1[:punto] + padre2[punto:]
            # Hijo 2: primeros 'punto' genes del padre2 + resto del padre1
            hijo2 = padre2[:punto] + padre1[punto:]
        else:
            # Sin crossover: los hijos son copias de los padres
            hijo1 = padre1
            hijo2 = padre2

        hijos.append(hijo1)
        hijos.append(hijo2)

    return hijos


# ============================================================
#  PASO 5 — MUTACIÓN INVERTIDA
# ============================================================

def mutacion(hijos):
    """
    Para cada gen de cada hijo, con probabilidad PM lo invertimos:
    0 → 1  o  1 → 0
    Esto introduce diversidad y evita quedar atascados en óptimos locales.
    """
    for cromosoma in hijos:
        for j in range(BITS):
            if random.random() < PM:
                cromosoma[j] = 1 - cromosoma[j]   # flip del bit
    return hijos


# ============================================================
#  CICLO PRINCIPAL DEL ALGORITMO GENÉTICO
# ============================================================

def ejecutar_ag(n_generaciones):
    """
    Corre el algoritmo genético completo por n_generaciones ciclos.
    Retorna:
      - historico: dict con listas de maximos, minimos y promedios por generación
      - mejor_cromosoma: el cromosoma con el f(x) más alto de toda la corrida
      - mejor_x: su valor decimal
      - mejor_fx: su f(x)
    """
    # --- PASO 1: Población inicial ---
    poblacion = crear_poblacion()

    historico = {
        "maximos":   [],
        "minimos":   [],
        "promedios": []
    }

    mejor_fx          = -1
    mejor_cromosoma   = None
    mejor_x           = None

    for generacion in range(1, n_generaciones + 1):

        # --- PASO 2: Evaluación ---
        valores_x, valores_fx, fitness = evaluar_poblacion(poblacion)

        # Estadísticas de esta generación
        maximo   = max(valores_fx)
        minimo   = min(valores_fx)
        promedio = sum(valores_fx) / len(valores_fx)

        historico["maximos"].append(maximo)
        historico["minimos"].append(minimo)
        historico["promedios"].append(promedio)

        # Actualizamos el mejor global de toda la corrida
        idx_max = valores_fx.index(maximo)
        if maximo > mejor_fx:
            mejor_fx        = maximo
            mejor_cromosoma = list(poblacion[idx_max])
            mejor_x         = valores_x[idx_max]

        # --- PASO 3: Selección ---
        padres = seleccion_ruleta(poblacion, fitness)

        # --- PASO 4: Crossover ---
        hijos = crossover(padres)

        # --- PASO 5: Mutación ---
        hijos = mutacion(hijos)

        # --- PASO 6: Nueva generación ---
        poblacion = hijos

    return historico, mejor_cromosoma, mejor_x, mejor_fx


# ============================================================
#  MOSTRAR TABLA DE RESULTADOS
# ============================================================

def mostrar_tabla(historico, n_generaciones):
    """Imprime la tabla de máximos, mínimos y promedios por generación."""
    print(f"\n{'='*62}")
    print(f"  TABLA DE RESULTADOS — {n_generaciones} generaciones")
    print(f"{'='*62}")
    print(f"  {'Gen':>4}  {'Máximo':>10}  {'Mínimo':>10}  {'Promedio':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")

    for i in range(n_generaciones):
        print(f"  {i+1:>4}  "
              f"{historico['maximos'][i]:>10.6f}  "
              f"{historico['minimos'][i]:>10.6f}  "
              f"{historico['promedios'][i]:>10.6f}")

    print(f"{'='*62}")
    print(f"  Máximo global  : {max(historico['maximos']):.6f}")
    print(f"  Mínimo global  : {min(historico['minimos']):.6f}")
    print(f"  Promedio global: {sum(historico['promedios'])/len(historico['promedios']):.6f}")
    print(f"{'='*62}")


# ============================================================
#  GRAFICAR RESULTADOS
# ============================================================

def graficar(historico, n_generaciones):
    """Genera la gráfica de máximos, mínimos y promedios por generación."""
    generaciones = list(range(1, n_generaciones + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(generaciones, historico["maximos"],   label="Máximo",   color="green",  linewidth=2)
    plt.plot(generaciones, historico["promedios"], label="Promedio", color="blue",   linewidth=2, linestyle="--")
    plt.plot(generaciones, historico["minimos"],   label="Mínimo",   color="red",    linewidth=2, linestyle=":")

    plt.title(f"Algoritmo Genético — f(x) por generación ({n_generaciones} iteraciones)", fontsize=13)
    plt.xlabel("Generación")
    plt.ylabel("f(x) = (x/coef)²")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"grafica_{n_generaciones}_generaciones.png", dpi=150)
    plt.show()
    print(f"  Gráfica guardada: grafica_{n_generaciones}_generaciones.png")


# ============================================================
#  PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("\n" + "█" * 62)
    print("  ALGORITMO GENÉTICO CANÓNICO")
    print(f"  f(x) = (x/coef)²  —  dominio [0, 2^30 - 1]")
    print(f"  Población: {TAM_POBLACION}  |  PC: {PC}  |  PM: {PM}")
    print("█" * 62)

    # Corremos el AG para 20, 100 y 200 generaciones
    for n_gen in [20, 100, 200]:

        print(f"\n{'▶'*3}  Corrida de {n_gen} generaciones...")

        historico, mejor_crom, mejor_x, mejor_fx = ejecutar_ag(n_gen)

        # Mostramos la tabla
        mostrar_tabla(historico, n_gen)

        # Mostramos el mejor resultado
        print(f"\n  ★ MEJOR RESULTADO ({n_gen} generaciones):")
        print(f"    Cromosoma : {''.join(str(b) for b in mejor_crom)}")
        print(f"    x (decimal): {mejor_x}")
        print(f"    f(x)       : {mejor_fx:.8f}")

        # Graficamos
        graficar(historico, n_gen)

    print("\n✔ Programa finalizado.")
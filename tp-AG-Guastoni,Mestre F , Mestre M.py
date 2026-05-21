import random


BITS = 30                        # longitud de cada cromosoma
COEF = 2**30 - 1                 # coeficiente de la función objetivo
TAM_POBLACION = 10               # cantidad de individuos


# ============================================================
#  PASO 1 — CREAR LA POBLACIÓN INICIAL
# ============================================================
#
#  Necesitamos 10 cromosomas al azar.
#  Cada cromosoma es una lista de 30 bits (0s y 1s).
#
#  Ejemplo de un cromosoma:
#  [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1]

def crear_poblacion():
    poblacion = []

    for _ in range(TAM_POBLACION):
        # Creamos un cromosoma: lista de 30 bits elegidos al azar (0 o 1)
        cromosoma = [random.randint(0, 1) for _ in range(BITS)]
        poblacion.append(cromosoma)

    return poblacion


# ============================================================
#  PASO 2 — EVALUAR LA POBLACIÓN
# ============================================================
#
#  Para cada cromosoma necesitamos calcular:
#    1. x      → convertir el binario a decimal
#    2. f(x)   → aplicar la función objetivo f(x) = (x/coef)²
#    3. fitness → qué proporción del total representa este individuo
#
#  El fitness nos dice qué tan "bueno" es cada uno
#  comparado con los demás de su generación.


def binario_a_decimal(cromosoma):
    # Convertimos la lista de bits a un número entero
    # Ejemplo: [1,1,0,1] → "1101" → 13
    bits_como_string = "".join(str(bit) for bit in cromosoma)
    return int(bits_como_string, 2)


def funcion_objetivo(x):
    # f(x) = (x / coef)²
    # Como x está entre 0 y coef, f(x) siempre da entre 0 y 1
    return (x / COEF) ** 2


def evaluar_poblacion(poblacion):
    # Primero calculamos x y f(x) para cada cromosoma
    valores_x  = []
    valores_fx = []

    for cromosoma in poblacion:
        x  = binario_a_decimal(cromosoma)
        fx = funcion_objetivo(x)
        valores_x.append(x)
        valores_fx.append(fx)

    # Ahora calculamos el fitness de cada uno
    # fitness(i) = f(xi) / suma_de_todos_los_f(x)
    suma_total = sum(valores_fx)

    fitness = []
    for fx in valores_fx:
        fitness.append(fx / suma_total)

    return valores_x, valores_fx, fitness


# ============================================================
#  PASO 3 — SELECCIÓN POR RULETA
# ============================================================
#
#  Imaginá una ruleta donde cada cromosoma ocupa un sector
#  proporcional a su fitness. El que más fitness tiene,
#  más grande es su sector → más probabilidad de ser elegido.
#
#  Para implementarlo usamos la "ruleta acumulada":
#
#  Ejemplo con 3 individuos:
#    fitness = [0.50, 0.30, 0.20]
#    acumulado = [0.50, 0.80, 1.00]
#
#  Tiramos un número random entre 0 y 1.
#  Si cae entre 0.00 y 0.50 → elegimos al individuo 1
#  Si cae entre 0.50 y 0.80 → elegimos al individuo 2
#  Si cae entre 0.80 y 1.00 → elegimos al individuo 3


def seleccion_ruleta(poblacion, fitness):
    # Armamos la ruleta acumulada
    acumulado = []
    suma = 0
    for f in fitness:
        suma += f
        acumulado.append(suma)

    # Giramos la ruleta TAM_POBLACION veces para elegir los padres
    padres = []

    for _ in range(TAM_POBLACION):
        r = random.random()   # número random entre 0 y 1

        # Buscamos en qué sector cayó el número
        for i, limite in enumerate(acumulado):
            if r <= limite:
                padres.append(poblacion[i])   # elegimos ese cromosoma
                break

    return padres


# ============================================================
#  PROGRAMA PRINCIPAL — probamos los 3 pasos
# ============================================================

if __name__ == "__main__":

    # --- PASO 1 ---
    print("=" * 55)
    print("PASO 1 — Población inicial")
    print("=" * 55)

    poblacion = crear_poblacion()

    for i, cromosoma in enumerate(poblacion):
        print(f"  Cromosoma {i+1:02d}: {''.join(str(b) for b in cromosoma)}")

    # --- PASO 2 ---
    print()
    print("=" * 55)
    print("PASO 2 — Evaluación")
    print("=" * 55)
    print(f"  {'#':<4} {'x (decimal)':<15} {'f(x)':<12} {'fitness'}")
    print(f"  {'-'*4} {'-'*15} {'-'*12} {'-'*10}")

    valores_x, valores_fx, fitness = evaluar_poblacion(poblacion)

    for i in range(TAM_POBLACION):
        print(f"  {i+1:<4} {valores_x[i]:<15} {valores_fx[i]:<12.6f} {fitness[i]:.6f}")

    print()
    print(f"  Máximo  f(x): {max(valores_fx):.6f}")
    print(f"  Mínimo  f(x): {min(valores_fx):.6f}")
    print(f"  Promedio f(x): {sum(valores_fx)/len(valores_fx):.6f}")

    # --- PASO 3 ---
    print()
    print("=" * 55)
    print("PASO 3 — Selección por Ruleta")
    print("=" * 55)

    padres = seleccion_ruleta(poblacion, fitness)

    for i, padre in enumerate(padres):
        print(f"  Padre {i+1:02d}: {''.join(str(b) for b in padre)}")

    print()
    print("Listo! Estos son los padres que van al crossover (Paso 4).")
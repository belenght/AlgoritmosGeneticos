"""
TP N°1 - Algoritmo Genético Canónico
Función objetivo: f(x) = (x / coef)^2
Dominio: [0, 2^30 - 1]
coef = 2^30 - 1

Opciones:
  A - Selección por Ruleta
  B - Selección por Torneo
  C - Elitismo (2 élites)
"""

import random
import math
import time
import statistics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tabulate import tabulate   # pip install tabulate

# ─────────────────────────── PARÁMETROS GLOBALES ────────────────────────────

BITS          = 30
COEF          = 2**BITS - 1          # 1 073 741 823
DOMINIO_MAX   = 2**BITS - 1
PC            = 0.75                 # probabilidad de crossover
PM            = 0.05                 # probabilidad de mutación
TAM_POB       = 10                   # individuos por generación
NUM_GEN       = 20                   # generaciones por corrida
ELITE_SIZE    = 2                    # tamaño de la élite (opción C)
TORNEO_K      = 3                    # participantes por torneo (opción B)


# ──────────────────────────── FUNCIÓN OBJETIVO ──────────────────────────────

def fitness(x: int) -> float:
    return (x / COEF) ** 2


# ────────────────────────── CODIFICACIÓN / DECODIFICACIÓN ───────────────────

def decodificar(cromosoma: str) -> int:
    return int(cromosoma, 2)

def codificar(valor: int) -> str:
    return format(valor, f'0{BITS}b')

def cromosoma_aleatorio() -> str:
    return ''.join(random.choice('01') for _ in range(BITS))

def poblacion_inicial() -> list[str]:
    return [cromosoma_aleatorio() for _ in range(TAM_POB)]


# ─────────────────────────── OPERADORES GENÉTICOS ───────────────────────────

# ── Selección: Ruleta ──────────────────────────────────────────────────────
def seleccion_ruleta(poblacion: list[str], fitnesses: list[float]) -> str:
    total = sum(fitnesses)
    if total == 0:
        return random.choice(poblacion)
    r = random.uniform(0, total)
    acumulado = 0
    for ind, f in zip(poblacion, fitnesses):
        acumulado += f
        if acumulado >= r:
            return ind
    return poblacion[-1]

# ── Selección: Torneo ──────────────────────────────────────────────────────
def seleccion_torneo(poblacion: list[str], fitnesses: list[float], k: int = TORNEO_K) -> str:
    indices = random.sample(range(len(poblacion)), min(k, len(poblacion)))
    ganador = max(indices, key=lambda i: fitnesses[i])
    return poblacion[ganador]

# ── Crossover de 1 punto ───────────────────────────────────────────────────
def crossover(padre1: str, padre2: str) -> tuple[str, str]:
    if random.random() < PC:
        punto = random.randint(1, BITS - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1, padre2

# ── Mutación invertida (bit-flip) ──────────────────────────────────────────
def mutar(cromosoma: str) -> str:
    resultado = []
    for bit in cromosoma:
        if random.random() < PM:
            resultado.append('1' if bit == '0' else '0')
        else:
            resultado.append(bit)
    return ''.join(resultado)


# ──────────────────────── ESTADÍSTICAS DE GENERACIÓN ────────────────────────

def estadisticas(fitnesses: list[float]) -> dict:
    n    = len(fitnesses)
    prom = sum(fitnesses) / n
    maxi = max(fitnesses)
    mini = min(fitnesses)
    std  = math.sqrt(sum((f - prom)**2 for f in fitnesses) / n)
    return {"max": maxi, "min": mini, "prom": prom, "std": std}


# ───────────────────────── ALGORITMO GENÉTICO BASE ──────────────────────────

def ejecutar_ag(metodo: str = 'ruleta', num_gen: int = NUM_GEN) -> dict:
    """
    Corre el AG completo por `num_gen` generaciones.
    metodo: 'ruleta' | 'torneo' | 'elitismo'
    Retorna dict con historiales y mejor individuo global.
    """
    pob = poblacion_inicial()

    hist_max  = []
    hist_min  = []
    hist_prom = []
    hist_std  = []

    mejor_global_crom = None
    mejor_global_fit  = -1.0

    for gen in range(num_gen):
        fits = [fitness(decodificar(c)) for c in pob]
        est  = estadisticas(fits)
        hist_max.append(est["max"])
        hist_min.append(est["min"])
        hist_prom.append(est["prom"])
        hist_std.append(est["std"])

        # actualizar mejor global
        idx_best = fits.index(max(fits))
        if fits[idx_best] > mejor_global_fit:
            mejor_global_fit  = fits[idx_best]
            mejor_global_crom = pob[idx_best]
            mejor_global_gen  = gen + 1

        # ── construir nueva población ──────────────────────────────────────
        nueva_pob = []

        # Elitismo: copiar los 2 mejores directamente
        if metodo == 'elitismo':
            elite_idx = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)[:ELITE_SIZE]
            for i in elite_idx:
                nueva_pob.append(pob[i])

        # Completar el resto de la nueva población
        while len(nueva_pob) < TAM_POB:
            # selección
            if metodo == 'torneo':
                p1 = seleccion_torneo(pob, fits)
                p2 = seleccion_torneo(pob, fits)
            else:
                # ruleta para 'ruleta' y 'elitismo'
                p1 = seleccion_ruleta(pob, fits)
                p2 = seleccion_ruleta(pob, fits)

            # crossover
            h1, h2 = crossover(p1, p2)

            # mutación
            h1 = mutar(h1)
            h2 = mutar(h2)

            nueva_pob.append(h1)
            if len(nueva_pob) < TAM_POB:
                nueva_pob.append(h2)

        pob = nueva_pob

    return {
        "hist_max":  hist_max,
        "hist_min":  hist_min,
        "hist_prom": hist_prom,
        "hist_std":  hist_std,
        "mejor_crom": mejor_global_crom,
        "mejor_fit":  mejor_global_fit,
        "mejor_gen":  mejor_global_gen,
    }


# ──────────────────────── MÚLTIPLES CORRIDAS ─────────────────────────────────

def multiples_corridas(metodo: str, n_corridas: int, num_gen: int = NUM_GEN) -> dict:
    """
    Ejecuta el AG `n_corridas` veces y agrega estadísticas.
    Retorna promedios de máx/mín/prom/std por generación + tiempos.
    """
    todos_max  = []
    todos_min  = []
    todos_prom = []
    todos_std  = []
    tiempos    = []
    mejores    = []

    for _ in range(n_corridas):
        t0  = time.perf_counter()
        res = ejecutar_ag(metodo=metodo, num_gen=num_gen)
        t1  = time.perf_counter()
        tiempos.append(t1 - t0)
        mejores.append(res["mejor_fit"])
        todos_max.append(res["hist_max"])
        todos_min.append(res["hist_min"])
        todos_prom.append(res["hist_prom"])
        todos_std.append(res["hist_std"])

    # promediar cada generación entre todas las corridas
    agg_max  = [statistics.mean(c[g] for c in todos_max)  for g in range(num_gen)]
    agg_min  = [statistics.mean(c[g] for c in todos_min)  for g in range(num_gen)]
    agg_prom = [statistics.mean(c[g] for c in todos_prom) for g in range(num_gen)]
    agg_std  = [statistics.mean(c[g] for c in todos_std)  for g in range(num_gen)]

    return {
        "agg_max":  agg_max,
        "agg_min":  agg_min,
        "agg_prom": agg_prom,
        "agg_std":  agg_std,
        "tiempo_prom": statistics.mean(tiempos),
        "tiempo_total": sum(tiempos),
        "mejor_fit_media": statistics.mean(mejores),
        "mejor_fit_std":   statistics.stdev(mejores) if len(mejores) > 1 else 0.0,
        "mejor_fit_max":   max(mejores),
        "n_corridas": n_corridas,
    }


# ────────────────────────────── GRAFICACIÓN ──────────────────────────────────

def graficar_corridas(resultados: dict, metodo: str, n_corridas: int):
    """Gráfica de máx, mín y prom por generación para un set de corridas."""
    gens = list(range(1, NUM_GEN + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Método: {metodo.capitalize()} | {n_corridas} corridas", fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(gens, resultados["agg_max"],  label="Máximo",   color='green',  marker='o', markersize=3)
    ax1.plot(gens, resultados["agg_prom"], label="Promedio", color='blue',   marker='s', markersize=3)
    ax1.plot(gens, resultados["agg_min"],  label="Mínimo",   color='red',    marker='^', markersize=3)
    ax1.set_title("Fitness por Generación (promedio de corridas)")
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.plot(gens, resultados["agg_std"], label="Desv. Estándar", color='purple', marker='D', markersize=3)
    ax2.set_title("Desviación Estándar del Fitness por Generación")
    ax2.set_xlabel("Generación")
    ax2.set_ylabel("Desviación Estándar")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    nombre = f"grafico_{metodo}_{n_corridas}corridas.png"
    plt.savefig(nombre, dpi=150)
    plt.show()
    print(f"  → Gráfico guardado: {nombre}")


def graficar_comparacion(res_ruleta, res_torneo, res_elitismo, n_corridas):
    """Comparación de los 3 métodos en la misma figura."""
    gens = list(range(1, NUM_GEN + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Comparación de métodos | {n_corridas} corridas", fontsize=14, fontweight='bold')
    metricas = [("agg_max", "Máximo"), ("agg_prom", "Promedio"), ("agg_min", "Mínimo")]

    for ax, (clave, titulo) in zip(axes, metricas):
        ax.plot(gens, res_ruleta[clave],   label='Ruleta',   color='blue')
        ax.plot(gens, res_torneo[clave],   label='Torneo',   color='green')
        ax.plot(gens, res_elitismo[clave], label='Elitismo', color='red')
        ax.set_title(titulo)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    nombre = f"comparacion_{n_corridas}corridas.png"
    plt.savefig(nombre, dpi=150)
    plt.show()
    print(f"  → Gráfico comparativo guardado: {nombre}")


# ──────────────────────────── IMPRESIÓN DE TABLAS ────────────────────────────

def imprimir_tabla_generaciones(resultados: dict, metodo: str, n_corridas: int):
    """Tabla generación a generación de máx, mín, prom, std."""
    encabezados = ["Gen", "Máximo", "Mínimo", "Promedio", "Desv. Std"]
    filas = []
    for g in range(NUM_GEN):
        filas.append([
            g + 1,
            f"{resultados['agg_max'][g]:.6f}",
            f"{resultados['agg_min'][g]:.6f}",
            f"{resultados['agg_prom'][g]:.6f}",
            f"{resultados['agg_std'][g]:.6f}",
        ])
    print(f"\n{'='*60}")
    print(f"  TABLA: {metodo.upper()} — {n_corridas} corridas")
    print(f"{'='*60}")
    print(tabulate(filas, headers=encabezados, tablefmt="rounded_outline"))


def imprimir_tabla_resumen(resultados_por_metodo: dict, corridas_lista: list):
    """Tabla resumen: mejor fitness, estabilidad y tiempo por método y corridas."""
    print(f"\n{'='*70}")
    print("  TABLA RESUMEN — Comparación de Métodos")
    print(f"{'='*70}")
    encabezados = ["Método", "Corridas", "Mejor Fit (media)", "Desv. entre corridas", "Tiempo prom (s)"]
    filas = []
    for metodo, datos_corridas in resultados_por_metodo.items():
        for nc, res in zip(corridas_lista, datos_corridas):
            filas.append([
                metodo.capitalize(),
                nc,
                f"{res['mejor_fit_media']:.6f}",
                f"{res['mejor_fit_std']:.6f}",
                f"{res['tiempo_prom']*1000:.3f} ms",
            ])
    print(tabulate(filas, headers=encabezados, tablefmt="rounded_outline"))


# ────────────────────────────── DEMO ÚNICA CORRIDA ───────────────────────────

def demo_corrida_unica(metodo: str):
    """Muestra detalle generación a generación de UNA sola corrida."""
    print(f"\n{'='*65}")
    print(f"  DEMO — Una corrida con método: {metodo.upper()}")
    print(f"{'='*65}")
    pob  = poblacion_inicial()
    encabezados = ["Gen", "Mejor cromosoma", "x", "Máx fit", "Mín fit", "Prom fit", "Desv.Std"]
    filas = []

    mejor_global_fit  = -1.0
    mejor_global_crom = None
    mejor_global_x    = None

    for gen in range(NUM_GEN):
        fits = [fitness(decodificar(c)) for c in pob]
        est  = estadisticas(fits)
        idx_best = fits.index(max(fits))
        best_crom = pob[idx_best]
        best_x    = decodificar(best_crom)

        if fits[idx_best] > mejor_global_fit:
            mejor_global_fit  = fits[idx_best]
            mejor_global_crom = best_crom
            mejor_global_x    = best_x

        filas.append([
            gen + 1,
            best_crom,
            best_x,
            f"{est['max']:.6f}",
            f"{est['min']:.6f}",
            f"{est['prom']:.6f}",
            f"{est['std']:.6f}",
        ])

        # nueva generación
        nueva_pob = []
        if metodo == 'elitismo':
            elite_idx = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)[:ELITE_SIZE]
            for i in elite_idx:
                nueva_pob.append(pob[i])

        while len(nueva_pob) < TAM_POB:
            if metodo == 'torneo':
                p1 = seleccion_torneo(pob, fits)
                p2 = seleccion_torneo(pob, fits)
            else:
                p1 = seleccion_ruleta(pob, fits)
                p2 = seleccion_ruleta(pob, fits)
            h1, h2 = crossover(p1, p2)
            h1 = mutar(h1)
            h2 = mutar(h2)
            nueva_pob.append(h1)
            if len(nueva_pob) < TAM_POB:
                nueva_pob.append(h2)
        pob = nueva_pob

    print(tabulate(filas, headers=encabezados, tablefmt="rounded_outline"))
    print(f"\n  ★ MEJOR SOLUCIÓN ENCONTRADA:")
    print(f"     Cromosoma : {mejor_global_crom}")
    print(f"     x         : {mejor_global_x}")
    print(f"     f(x)      : {mejor_global_fit:.8f}")


# ─────────────────────────────────── MAIN ────────────────────────────────────

def main():
    CORRIDAS_LISTA = [20, 100, 200]
    METODOS = ['ruleta', 'torneo', 'elitismo']

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        TP N°1 — ALGORITMO GENÉTICO CANÓNICO             ║")
    print("║  f(x) = (x/coef)²   dominio [0, 2^30-1]                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Parámetros:")
    print(f"    Bits          : {BITS}")
    print(f"    coef          : {COEF}")
    print(f"    PC            : {PC}")
    print(f"    PM            : {PM}")
    print(f"    Tamaño pob.   : {TAM_POB}")
    print(f"    Generaciones  : {NUM_GEN}")
    print(f"    Elite size    : {ELITE_SIZE}")
    print(f"    Torneo K      : {TORNEO_K}")

    # ── DEMO: una sola corrida por método ──────────────────────────────────
    for metodo in METODOS:
        demo_corrida_unica(metodo)

    # ── MÚLTIPLES CORRIDAS ─────────────────────────────────────────────────
    resultados_por_metodo = {m: [] for m in METODOS}

    for metodo in METODOS:
        print(f"\n{'─'*60}")
        print(f"  Ejecutando múltiples corridas — Método: {metodo.upper()}")
        print(f"{'─'*60}")
        for nc in CORRIDAS_LISTA:
            print(f"  → {nc} corridas ... ", end='', flush=True)
            t0  = time.perf_counter()
            res = multiples_corridas(metodo=metodo, n_corridas=nc)
            t1  = time.perf_counter()
            print(f"listo en {t1-t0:.2f}s")
            resultados_por_metodo[metodo].append(res)
            imprimir_tabla_generaciones(res, metodo, nc)
            graficar_corridas(res, metodo, nc)

    # ── TABLA RESUMEN ──────────────────────────────────────────────────────
    imprimir_tabla_resumen(resultados_por_metodo, CORRIDAS_LISTA)

    # ── GRÁFICAS COMPARATIVAS ─────────────────────────────────────────────
    for idx, nc in enumerate(CORRIDAS_LISTA):
        res_r = resultados_por_metodo['ruleta'][idx]
        res_t = resultados_por_metodo['torneo'][idx]
        res_e = resultados_por_metodo['elitismo'][idx]
        graficar_comparacion(res_r, res_t, res_e, nc)

    # ── TABLA TIEMPO DE EJECUCIÓN ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TABLA — Tiempo de ejecución promedio por corrida (ms)")
    print(f"{'='*60}")
    encabezados = ["Método"] + [f"{nc} corridas" for nc in CORRIDAS_LISTA]
    filas = []
    for metodo in METODOS:
        fila = [metodo.capitalize()]
        for res in resultados_por_metodo[metodo]:
            fila.append(f"{res['tiempo_prom']*1000:.4f} ms")
        filas.append(fila)
    print(tabulate(filas, headers=encabezados, tablefmt="rounded_outline"))

    print("\n  ✓ TP finalizado. Revise los archivos PNG generados.\n")


if __name__ == "__main__":
    main()
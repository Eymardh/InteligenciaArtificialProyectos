import heapq
import copy

class Estado:
    def __init__(self, tablero, g, h, padre=None):
        self.tablero = tablero
        self.g = g
        self.h = h 
        self.padre = padre

    def __lt__(self, otro):
        return (self.g + self.h) < (otro.g + otro.h)

    def __eq__(self, otro):
        return self.tablero == otro.tablero

    def __hash__(self):
        return hash(tuple(map(tuple, self.tablero)))

def distancia_manhattan(tablero, estado_final):
    distancia = 0
    for i in range(3):
        for j in range(3):
            valor = tablero[i][j]
            if valor != 0:
                fila_objetivo, col_objetivo = divmod(valor - 1, 3)
                distancia += abs(i - fila_objetivo) + abs(j - col_objetivo)
    return distancia

def encontrar_vacio(tablero):
    for i in range(3):
        for j in range(3):
            if tablero[i][j] == 0:
                return (i, j)
    return None

def generar_movimientos(estado):
    movimientos = []
    i, j = encontrar_vacio(estado.tablero)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            nuevo_tablero = copy.deepcopy(estado.tablero)
            nuevo_tablero[i][j], nuevo_tablero[ni][nj] = nuevo_tablero[ni][nj], nuevo_tablero[i][j]
            movimientos.append(nuevo_tablero)
    return movimientos

def resolver_puzzle(estado_inicial, estado_final):
    cola_prioridad = []
    heapq.heappush(cola_prioridad, estado_inicial)
    visitados = set()

    while cola_prioridad:
        estado_actual = heapq.heappop(cola_prioridad)

        if estado_actual.tablero == estado_final.tablero:
            return estado_actual

        visitados.add(estado_actual)

        for movimiento in generar_movimientos(estado_actual):
            nuevo_estado = Estado(movimiento, estado_actual.g + 1, distancia_manhattan(movimiento, estado_final.tablero), estado_actual)
            if nuevo_estado not in visitados:
                heapq.heappush(cola_prioridad, nuevo_estado)

    return None

def imprimir_camino(estado):
    camino = []
    while estado:
        camino.append(estado.tablero)
        estado = estado.padre
    camino.reverse()
    for paso in camino:
        for fila in paso:
            print(fila)
        print("-----")

estado_inicial_tablero = [
    [1, 2, 3],
    [4, 6, 0],
    [7, 5, 8]
    
    # [1, 2, 3]
    # [4, 5, 6]
    # [0, 7, 8]

    # [1, 2, 3]
    # [5, 0, 6]
    # [4, 7, 8]

    # [1, 2, 3]
    # [4, 0, 5]
    # [7, 8, 6]

]

estado_final_tablero = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

estado_inicial = Estado(estado_inicial_tablero, 0, distancia_manhattan(estado_inicial_tablero, estado_final_tablero))
estado_final = Estado(estado_final_tablero, 0, 0)

solucion = resolver_puzzle(estado_inicial, estado_final)

if solucion:
    print("Camino más corto:")
    imprimir_camino(solucion)
else:
    print("No se encontró solución.")
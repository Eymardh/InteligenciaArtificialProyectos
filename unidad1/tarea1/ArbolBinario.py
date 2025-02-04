class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.izquierda = None
        self.derecha = None

class ArbolBinario:
    def __init__(self):
        self.raiz = None
    
    def insertar(self, valor):
        if self.raiz is None:
            self.raiz = Nodo(valor)
        else:
            self._insertar_recursivo(self.raiz, valor)
    
    def _insertar_recursivo(self, actual, valor):
        if valor < actual.valor:
            if actual.izquierda is None:
                actual.izquierda = Nodo(valor)
            else:
                self._insertar_recursivo(actual.izquierda, valor)
        else:
            if actual.derecha is None:
                actual.derecha = Nodo(valor)
            else:
                self._insertar_recursivo(actual.derecha, valor)
    
    def enorden_transversal(self):
        self._enorder_recursivo(self.raiz)
        print()
    
    def _enorder_recursivo(self, actual):
        if actual is not None:
            self._enorder_recursivo(actual.izquierda)
            print(actual.valor, end=" ")
            self._enorder_recursivo(actual.derecha)
    
    def buscar(self, valor):
        return self._buscar_recursivo(self.raiz, valor)
    
    def _buscar_recursivo(self, actual, valor):
        if actual is None:
            return False
        if actual.valor == valor:
            return True
        elif valor < actual.valor:
            return self._buscar_recursivo(actual.izquierda, valor)
        else:
            return self._buscar_recursivo(actual.derecha, valor)

# Ejemplo de uso
bst = ArbolBinario()
bst.insertar(50)
bst.insertar(30)
bst.insertar(70)
bst.insertar(20)
bst.insertar(40)
bst.insertar(60)
bst.insertar(80)

print("Recorrido en orden:")
bst.enorden_transversal()

# Búsqueda de elementos
print("Buscar 40:", bst.buscar(40))  # True
print("Buscar 100:", bst.buscar(100))  # False


# Búsqueda de elementos
print("Buscar 40:", bst.buscar(80))  # True
print("Buscar 100:", bst.buscar(50))  # True

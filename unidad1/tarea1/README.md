# Árbol Binario de Búsqueda en Python

Este proyecto implementa un Árbol Binario de Búsqueda (ABB) en Python, permitiendo insertar elementos, realizar búsquedas y recorrer el árbol en orden.

## Implementación

La implementación consta de dos clases:

- **Nodo**: Representa cada nodo del árbol con un valor y referencias a los nodos izquierdo y derecho.
- **ArbolBinario**: Contiene la raíz del árbol y métodos para insertar, buscar y recorrer el árbol en orden.

## Métodos Principales

- `insertar(valor)`: Inserta un nuevo valor en el árbol.
- `buscar(valor)`: Devuelve `True` si el valor está en el árbol, `False` en caso contrario.
- `enorden_transversal()`: Muestra los valores del árbol en orden ascendente.

## Ejemplo de Uso / Prueba

```python
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

print("Buscar 40:", bst.buscar(40))  # True
print("Buscar 100:", bst.buscar(100))  # False
```

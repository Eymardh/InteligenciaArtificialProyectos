# Lógica Proposicional para el Sistema Experto de Recomendación de Ejercicios

## Tabla de Variables Proposicionales
| Variable | Descripción                                  |
|----------|----------------------------------------------|
| G1       | Meta: Perder peso                            |
| G2       | Meta: Ganar músculo                          |
| G3       | Meta: Mantenerse en forma                    |
| N1       | Nivel: Principiante                          |
| N2       | Nivel: Intermedio                            |
| N3       | Nivel: Avanzado                              |
| T1       | Tiempo: 20-30 minutos                        |
| T2       | Tiempo: 30-45 minutos                        |
| T3       | Tiempo: 45-60 minutos                        |
| E        | Equipamiento básico disponible               |

| Salida   | Recomendación                                |
|----------|----------------------------------------------|
| R1       | Cardio de bajo impacto                        |
| R2       | HIIT + Fuerza básica                          |
| R3       | Entrenamiento de fuerza completo             |
| R4       | Yoga + Cardio moderado                        |
| R5       | Cardio suave + Ejercicios posturales (seguridad) |

---

## Reglas de Lógica Proposicional

### Reglas Principales
**Regla 1** (Para principiantes que buscan perder peso):  
`G1 ∧ N1 ∧ T1 → R1`  
*Ejemplo:* Si el usuario quiere perder peso (G1), es principiante (N1) y tiene 20-30 minutos (T1), recomendar R1.

**Regla 2** (Para intermedios con meta de perder peso):  
`G1 ∧ N2 ∧ T2 → R2`  
*Ejemplo:* Meta = perder peso (G1), nivel intermedio (N2), tiempo 30-45 min (T2) → HIIT + fuerza.

**Regla 3** (Para avanzados que buscan ganar músculo):  
`G2 ∧ N3 ∧ T3 ∧ E → R3`  
*Ejemplo:* Ganar músculo (G2), avanzado (N3), 45-60 min (T3), con equipamiento (E) → Rutina de fuerza completa.

**Regla 4** (Para mantenerse en forma con poco tiempo):  
`G3 ∧ T1 → R4`  
*Ejemplo:* Mantenerse (G3) + 20-30 min (T1) → Yoga + cardio moderado.

---

### Reglas de Seguridad
**Regla 5** (Prevención para principiantes en ganancia muscular):  
`N1 ∧ G2 → R5`  
*Ejemplo:* Si es principiante (N1) pero selecciona ganar músculo (G2), recomendar R5 en lugar de rutinas intensas.

---

## Ejemplo de Aplicación
**Caso**:  
- `G1 = Verdadero` (Perder peso)  
- `N1 = Verdadero` (Principiante)  
- `T1 = Verdadero` (20-30 min)  
- `E = Falso` (Sin equipamiento)  

**Aplicación de reglas**:  
1. `G1 ∧ N1 ∧ T1 → R1` (Se activa)  
2. `N1 ∧ G2 → R5` (No aplica, G2 es falso)  

**Conclusión**:  
`R1 = Verdadero` → Recomendar "Cardio de bajo impacto".

---

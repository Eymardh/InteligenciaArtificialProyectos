:- dynamic meta/1, nivel/1, tiempo/1, equipamiento/1.
:- use_module(library(readutil)).

% PREDICADOS PARA VALIDACIÓN DE ENTRADAS
leer_entrada_numero(Mensaje, Min, Max, Opcion) :-
    repeat,
    format('~w (~w-~w): ', [Mensaje, Min, Max]),
    read_line_to_string(user_input, Linea),
    (atom_number(Linea, Opcion), 
     integer(Opcion), 
     Opcion >= Min, 
     Opcion =< Max -> 
        true
    ;
        format('Opción inválida. Ingrese un número entre ~w y ~w.~n', [Min, Max]),
        fail
    ).

leer_entrada_sn(Pregunta, Respuesta) :-
    repeat,
    format('~w (s/n): ', [Pregunta]),
    read_line_to_string(user_input, Linea),
    string_lower(Linea, Lower),
    (sub_string(Lower, 0, 1, _, "s") -> Respuesta = si;
     sub_string(Lower, 0, 1, _, "n") -> Respuesta = no;
     (writeln('Respuesta no válida, responda "s" o "n"'), fail)).

% PREDICADO PRINCIPAL
iniciar :-
    retractall(meta(_)),
    retractall(nivel(_)),
    retractall(tiempo(_)),
    retractall(equipamiento(_)),
    
    writeln('=== SISTEMA EXPERTO PARA RECOMENDAR EJERCICIOS ==='),
    obtener_datos_usuario,
    generar_recomendacion,
    finalizar_consulta.

% OBTENER DATOS DEL USUARIO
obtener_datos_usuario :-
    writeln('\nSeleccione su meta principal:'),
    writeln('1. Perder peso'),
    writeln('2. Ganar músculo'),
    writeln('3. Mantenerse en forma'),
    leer_entrada_numero('Opción', 1, 3, OpcionMeta),
    assert_meta(OpcionMeta),

    writeln('\nNivel de condición física:'),
    writeln('1. Principiante'),
    writeln('2. Intermedio'),
    writeln('3. Avanzado'),
    leer_entrada_numero('Opción', 1, 3, OpcionNivel),
    assert_nivel(OpcionNivel),

    writeln('\nTiempo disponible por sesión:'),
    writeln('1. 20-30 minutos'),
    writeln('2. 30-45 minutos'),
    writeln('3. 45-60 minutos'),
    leer_entrada_numero('Opción', 1, 3, OpcionTiempo),
    assert_tiempo(OpcionTiempo),

    preguntar_equipamiento.

assert_meta(1) :- assertz(meta(perder_peso)).
assert_meta(2) :- assertz(meta(ganar_musculo)).
assert_meta(3) :- assertz(meta(mantenerse)).

assert_nivel(1) :- assertz(nivel(principiante)).
assert_nivel(2) :- assertz(nivel(intermedio)).
assert_nivel(3) :- assertz(nivel(avanzado)).

assert_tiempo(1) :- assertz(tiempo('20-30')).
assert_tiempo(2) :- assertz(tiempo('30-45')).
assert_tiempo(3) :- assertz(tiempo('45-60')).

preguntar_equipamiento :-
    leer_entrada_sn('¿Tiene acceso a equipamiento básico? (mancuernas, banda elástica)', Resp),
    (Resp = si -> assertz(equipamiento(si)); true.

% REGLAS DE RECOMENDACIÓN
generar_recomendacion :-
    writeln('\n=== RECOMENDACIÓN PERSONALIZADA ==='),
    (recomendacion(Rutina, Explicacion) -> 
        format('Rutina sugerida: ~w~n~n', [Rutina]),
        explicacion(Explicacion)
    ;
        writeln('No se encontró rutina adecuada para sus características.')
    ).

% REGLAS PRINCIPALES
recomendacion('Cardio de bajo impacto', 1) :-
    meta(perder_peso),
    nivel(principiante),
    tiempo('20-30').

recomendacion('HIIT + Fuerza básica', 2) :-
    meta(perder_peso),
    nivel(intermedio),
    tiempo('30-45').

recomendacion('Entrenamiento de fuerza completo', 3) :-
    meta(ganar_musculo),
    nivel(avanzado),
    tiempo('45-60'),
    equipamiento(si).

recomendacion('Yoga + Cardio moderado', 4) :-
    meta(mantenerse),
    tiempo('20-30').

% REGLAS DE SEGURIDAD
recomendacion('Cardio suave + Ejercicios posturales', 5) :-
    nivel(principiante),
    meta(ganar_musculo).

% EXPLICACIONES
explicacion(1) :-
    writeln('Detalles:'),
    writeln('- Caminata rápida 25 minutos'),
    writeln('- Saltos suaves: 3 series de 10 repeticiones'),
    writeln('- Estiramientos finales 5 minutos'),
    writeln('Precaución: Evitar impactos en articulaciones').

explicacion(2) :-
    writeln('Detalles:'),
    writeln('- HIIT: 20 minutos (30s sprint/1min descanso)'),
    writeln('- Sentadillas: 3x12'),
    writeln('- Flexiones modificadas: 3x10'),
    writeln('- Descanso entre series: 45 segundos').

explicacion(3) :-
    writeln('Detalles:'),
    writeln('- Press banca: 4x8 (70% 1RM)'),
    writeln('- Peso muerto: 4x6'),
    writeln('- Dominadas asistidas: 4x8'),
    writeln('- Descanso entre series: 90 segundos').

explicacion(4) :-
    writeln('Detalles:'),
    writeln('- Saludo al sol: 10 minutos'),
    writeln('- Elíptica/bicicleta: 15 minutos'),
    writeln('- Ejercicios de movilidad articular: 5 minutos').

explicacion(5) :-
    writeln('Detalles:'),
    writeln('- Caminata en inclinación: 20 minutos'),
    writeln('- Plancha estática: 3 series de 20 segundos'),
    writeln('- Ejercicios con banda elástica: 2x15'),
    writeln('Precaución: Enfoque en técnica correcta').

% FINALIZAR CONSULTA
finalizar_consulta :-
    writeln('\n¿Desea otra recomendación? (s/n)'),
    leer_entrada_sn('', Respuesta),
    (Respuesta = si -> iniciar; 
     Respuesta = no -> writeln('¡Buena rutina!')).

% INICIAR SISTEMA
:- iniciar.
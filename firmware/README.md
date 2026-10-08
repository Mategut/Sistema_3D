# Control de la plataforma

El [firmware del Arduino Uno](control_plataforma_2055_pasos/control_plataforma_2055_pasos.ino) controla la plataforma mediante un motor 28BYJ-48 unipolar y un módulo ULN2003. Arduino se comunica con el computador por USB a 115200 baudios. Las dos cámaras utilizan sus propias conexiones USB al mismo equipo.

![Diagrama de conexiones](imagenes/conexiones_plataforma.png)

Diagrama funcional: los terminales se identifican por sus etiquetas; la posición gráfica no reproduce la distribución física de las placas y los colores son ilustrativos.

| Arduino Uno | ULN2003 |
| --- | --- |
| D8 | IN1 |
| D9 | IN2 |
| D10 | IN3 |
| D11 | IN4 |
| 5 V | + / VCC |
| GND | − / GND |

Conecte el cable de cinco hilos del motor al conector del módulo ULN2003. En este montaje, Arduino suministra 5 V y comparte masa con el controlador. El diagrama identifica las conexiones por sus etiquetas; los colores dibujados no indican un orden ni colores reales para los hilos del motor.

`Stepper(..., 8, 10, 9, 11)` establece el orden de accionamiento IN1, IN3, IN2, IN4. El cableado físico sigue la tabla, sin intercambiar IN2 e IN3. La biblioteca utiliza 2048 pasos para temporización; el recorrido calibrado es de 2055 pasos por vuelta. La secuencia de 25 movimientos es `[82, 82, 83, 82, 82]` repetida cinco veces.

Durante la adquisición se realizan 24 avances `NEXT` y `CLOSE` completa los 82 pasos restantes. Como el montaje no dispone de sensor de origen, `RESET` reinicia los contadores sin modificar la posición física. `STOP` libera las bobinas una vez concluido el movimiento bloqueante en curso. Consulte el [manual de operación](../documentacion/instalacion_y_uso.md#plataforma-y-firmware) y la [recuperación de trabajos](../documentacion/operacion_y_cierre.md).

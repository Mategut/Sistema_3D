/*
 * Plataforma giratoria: 2055 pasos calibrados por vuelta y 25 poses.
 *
 * La secuencia [82, 82, 83, 82, 82] repetida cinco veces suma 2055 pasos.
 * Los 2048 pasos de Stepper configuran la temporizacion de la biblioteca;
 * los movimientos de captura siguen la secuencia calibrada del montaje.
 *
 * Protocolo serie: 115200 baudios, comandos en mayusculas terminados en '\n'.
 *   PING    -> identidad ROT2055_V7_2.
 *   STATUS  -> fase, pasos acumulados y parametros de la vuelta.
 *   RESET   -> reinicia contadores, sin movimiento ni busqueda fisica de origen.
 *   NEXT    -> ejecuta una de las primeras 24 transiciones.
 *   CLOSE   -> completa los ultimos 82 pasos desde fase 24 y acumulado 1973.
 *   RELEASE -> desenergiza las bobinas, sin cambiar los contadores.
 *   STOP    -> desenergiza las bobinas, sin cambiar los contadores.
 *
 * NEXT y CLOSE mantienen el motor energizado durante el asentamiento y la
 * captura. Al arrancar, las bobinas quedan libres hasta el primer movimiento.
 * Stepper.step() es bloqueante: otro comando se atiende cuando termina el
 * movimiento en curso. STOP no interrumpe una llamada ya iniciada a step().
 *
 * Un CLOSE correcto responde:
 *   CLOSE_DONE delta=82 total=2055 phase=0 cumulative=0
 * Los contadores representan el estado logico; no miden la posicion con sensores.
 */

#include <Stepper.h>

// Temporizacion nominal de Stepper y recorrido calibrado del montaje.
const long STEPPER_LIBRARY_STEPS_PER_REV = 2048L;
const long CALIBRATED_STEPS_PER_REV = 2055L;
const int MOVES_PER_REV = 25;
const int MOTOR_RPM = 8;

const long EXPECTED_BEFORE_CLOSE = 1973L;
const int FINAL_CLOSE_STEPS = 82;

const int STEP_SEQUENCE[MOVES_PER_REV] = {
  82, 82, 83, 82, 82,
  82, 82, 83, 82, 82,
  82, 82, 83, 82, 82,
  82, 82, 83, 82, 82,
  82, 82, 83, 82, 82
};

// Orden de excitacion de las bobinas: conservar la asignacion 8, 10, 9, 11.
Stepper motor(
  STEPPER_LIBRARY_STEPS_PER_REV,
  8, 10, 9, 11
);

// Estado logico compartido por NEXT, CLOSE, RESET y STATUS.
int phaseIndex = 0;
long cumulativeSteps = 0;

// Desenergiza las cuatro salidas sin modificar la fase ni los pasos acumulados.
void releaseMotor() {
  digitalWrite(8, LOW);
  digitalWrite(9, LOW);
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);
}

// Publica el estado con los nombres de campo que interpreta la aplicacion Python.
void printStatus() {
  Serial.print("STATUS phase=");
  Serial.print(phaseIndex);
  Serial.print(" cumulative=");
  Serial.print(cumulativeSteps);
  Serial.print(" revsteps=");
  Serial.print(CALIBRATED_STEPS_PER_REV);
  Serial.print(" moves=");
  Serial.println(MOVES_PER_REV);
}

// Inicializa el puerto, la velocidad nominal y el estado libre del motor.
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(150);

  motor.setSpeed(MOTOR_RPM);

  // Al arrancar, las bobinas permanecen libres hasta el primer movimiento.
  releaseMotor();

  Serial.println("READY ROT2055_V7_2");
}

// Valida y ejecuta un comando; conserva las respuestas del protocolo de captura.
void handleCommand(String cmd) {
  cmd.trim();

  if (cmd.length() == 0) {
    return;
  }

  if (cmd == "PING") {
    Serial.println("PONG ROT2055_V7_2");
    return;
  }

  if (cmd == "STATUS") {
    printStatus();
    return;
  }

  if (cmd == "RESET") {
    // NO mueve físicamente el motor.
    phaseIndex = 0;
    cumulativeSteps = 0;

    Serial.println("RESET_OK phase=0 cumulative=0");
    return;
  }

  if (cmd == "NEXT") {
    // NEXT solo puede ejecutar las primeras 24 transiciones.
    // La transición número 25 está reservada exclusivamente para CLOSE.
    if (phaseIndex >= 24) {
      Serial.println("ERR NEXT_REQUIRES_CLOSE");
      return;
    }

    int delta = STEP_SEQUENCE[phaseIndex];

    motor.step(delta);

    // Conservar las bobinas energizadas para sostener la pose durante la captura.

    cumulativeSteps += delta;
    phaseIndex++;

    Serial.print("DONE delta=");
    Serial.print(delta);
    Serial.print(" phase=");
    Serial.print(phaseIndex);
    Serial.print(" cumulative=");
    Serial.print(cumulativeSteps);
    Serial.println(" revolution=0");

    return;
  }

  if (cmd == "CLOSE") {
    // CLOSE es deliberadamente estricto.
    // Solo puede ejecutarse después de las 24 transiciones normales.
    if (phaseIndex != 24) {
      Serial.print("ERR CLOSE_PHASE phase=");
      Serial.println(phaseIndex);
      return;
    }

    if (cumulativeSteps != EXPECTED_BEFORE_CLOSE) {
      Serial.print("ERR CLOSE_CUMULATIVE cumulative=");
      Serial.println(cumulativeSteps);
      return;
    }

    int delta = STEP_SEQUENCE[24];

    if (delta != FINAL_CLOSE_STEPS) {
      Serial.println("ERR CLOSE_INTERNAL_DELTA");
      return;
    }

    motor.step(delta);

    // Conservar energizada la posicion alcanzada para el inicio de otra sesion.

    long total = cumulativeSteps + delta;

    if (total != CALIBRATED_STEPS_PER_REV) {
      Serial.print("ERR CLOSE_TOTAL total=");
      Serial.println(total);
      return;
    }

    // El total ya esta guardado; reiniciar contadores y publicar el cierre.
    phaseIndex = 0;
    cumulativeSteps = 0;

    Serial.print("CLOSE_DONE delta=");
    Serial.print(delta);
    Serial.print(" total=");
    Serial.print(total);
    Serial.print(" phase=");
    Serial.print(phaseIndex);
    Serial.print(" cumulative=");
    Serial.println(cumulativeSteps);

    return;
  }

  if (cmd == "RELEASE") {
    releaseMotor();
    Serial.println("RELEASED");
    return;
  }

  if (cmd == "STOP") {
    releaseMotor();
    Serial.println("STOPPED");
    return;
  }

  Serial.print("ERR UNKNOWN ");
  Serial.println(cmd);
}

// Lee una linea del puerto y la entrega al despachador de comandos.
void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }
}

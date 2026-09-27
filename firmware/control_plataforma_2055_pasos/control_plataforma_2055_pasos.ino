/*
 * Plataforma: 2055 pasos por vuelta, 25 poses; Stepper usa 2048 para temporizacion.
 * Secuencia: [82, 82, 83, 82, 82] repetida cinco veces.
 * Serie: 115200 baudios, comandos en mayusculas terminados en '\n'.
 * NEXT ejecuta 24 transiciones; CLOSE completa 82 pasos desde phase=24,
 * cumulative=1973 y responde total=2055, phase=0, cumulative=0.
 * PING identifica el firmware; STATUS informa contadores; RESET no mueve el motor.
 * RELEASE y STOP liberan bobinas. STOP no interrumpe Stepper.step(), que es bloqueante.
 * NEXT/CLOSE mantienen la pose energizada. Los contadores no usan sensores de posicion.
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

// Orden de bobinas: 8, 10, 9, 11.
Stepper motor(
  STEPPER_LIBRARY_STEPS_PER_REV,
  8, 10, 9, 11
);

// Contadores logicos de la vuelta.
int phaseIndex = 0;
long cumulativeSteps = 0;

// Libera bobinas y conserva los contadores.
void releaseMotor() {
  digitalWrite(8, LOW);
  digitalWrite(9, LOW);
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);
}

// Nombres de campo usados por el cliente Python.
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

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(150);

  motor.setSpeed(MOTOR_RPM);

  // Al arrancar, las bobinas permanecen libres hasta el primer movimiento.
  releaseMotor();

  Serial.println("READY ROT2055_V7_2");
}

// Despacho del protocolo serie.
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
    // La ultima transicion se ejecuta con CLOSE.
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
    // Exigir las 24 transiciones previas antes del cierre.
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

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }
}

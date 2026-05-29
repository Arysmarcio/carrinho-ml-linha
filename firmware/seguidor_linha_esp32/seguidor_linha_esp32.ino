#include "modelo_arvore.h"

#define PINO_SENSOR_ESQ  4
#define PINO_SENSOR_DIR  5

#define MOTOR_ESQ_IN1   16
#define MOTOR_ESQ_IN2   17
#define MOTOR_ESQ_EN    18

#define MOTOR_DIR_IN1   19
#define MOTOR_DIR_IN2   20
#define MOTOR_DIR_EN    21

#define PWM_FREQ      1000
#define PWM_RES          8

#define CH_ESQ           0
#define CH_DIR           1

#define VEL_MAX        200
#define VEL_MEDIA      150
#define VEL_BAIXA       90
#define VEL_BUSCA       70

float Kp_reta = 0.8;
float Ki_reta = 0.0;
float Kd_reta = 0.1;
int vel_reta = VEL_MAX;

float Kp_curva = 1.8;
float Ki_curva = 0.0;
float Kd_curva = 0.3;
int vel_curva = VEL_MEDIA;

int vel_cruzamento = VEL_BAIXA;
int tempo_cruzamento_ms = 250;
int vel_busca = VEL_BUSCA;

float erro_anterior = 0.0;
float integral = 0.0;

unsigned long t_anterior = 0;

bool em_cruzamento = false;
unsigned long t_cruzamento = 0;

void configurar_motores() {
  pinMode(MOTOR_ESQ_IN1, OUTPUT);
  pinMode(MOTOR_ESQ_IN2, OUTPUT);

  pinMode(MOTOR_DIR_IN1, OUTPUT);
  pinMode(MOTOR_DIR_IN2, OUTPUT);

  ledcSetup(CH_ESQ, PWM_FREQ, PWM_RES);
  ledcSetup(CH_DIR, PWM_FREQ, PWM_RES);

  ledcAttachPin(MOTOR_ESQ_EN, CH_ESQ);
  ledcAttachPin(MOTOR_DIR_EN, CH_DIR);
}

void mover(int vel_esq, int vel_dir) {

  if (vel_esq >= 0) {
    digitalWrite(MOTOR_ESQ_IN1, HIGH);
    digitalWrite(MOTOR_ESQ_IN2, LOW);
    ledcWrite(CH_ESQ, constrain(vel_esq, 0, 255));
  } else {
    digitalWrite(MOTOR_ESQ_IN1, LOW);
    digitalWrite(MOTOR_ESQ_IN2, HIGH);
    ledcWrite(CH_ESQ, constrain(-vel_esq, 0, 255));
  }

  if (vel_dir >= 0) {
    digitalWrite(MOTOR_DIR_IN1, HIGH);
    digitalWrite(MOTOR_DIR_IN2, LOW);
    ledcWrite(CH_DIR, constrain(vel_dir, 0, 255));
  } else {
    digitalWrite(MOTOR_DIR_IN1, LOW);
    digitalWrite(MOTOR_DIR_IN2, HIGH);
    ledcWrite(CH_DIR, constrain(-vel_dir, 0, 255));
  }
}

void parar() {
  ledcWrite(CH_ESQ, 0);
  ledcWrite(CH_DIR, 0);
}

float calcular_pid(float erro, float Kp, float Ki, float Kd) {

  unsigned long t_atual = millis();

  float dt = (t_atual - t_anterior) / 1000.0;

  if (dt <= 0) {
    dt = 0.001;
  }

  float P = Kp * erro;

  integral += erro * dt;
  integral = constrain(integral, -500, 500);

  float I = Ki * integral;

  float D = Kd * (erro - erro_anterior) / dt;

  erro_anterior = erro;
  t_anterior = t_atual;

  return P + I + D;
}

void setup() {

  Serial.begin(115200);

  delay(500);

  analogReadResolution(10);
  analogSetAttenuation(ADC_11db);

  configurar_motores();

  t_anterior = millis();

  delay(2000);
}

void loop() {

  int s_esq = analogRead(PINO_SENSOR_ESQ);
  int s_dir = analogRead(PINO_SENSOR_DIR);

  int acao = predict(s_esq, s_dir);

  Serial.printf(
    "S_ESQ=%4d  S_DIR=%4d  ACAO=%d\n",
    s_esq,
    s_dir,
    acao
  );

  float erro = (float)(s_dir - s_esq) / 1023.0;

  switch (acao) {

    case ACAO_RETO: {

      integral = 0;

      float correcao = calcular_pid(
        erro,
        Kp_reta,
        Ki_reta,
        Kd_reta
      );

      int v_esq = vel_reta - (int)correcao;
      int v_dir = vel_reta + (int)correcao;

      mover(
        constrain(v_esq, 0, 255),
        constrain(v_dir, 0, 255)
      );

      break;
    }

    case ACAO_CURVA_ESQ: {

      float correcao = calcular_pid(
        erro,
        Kp_curva,
        Ki_curva,
        Kd_curva
      );

      int v_esq = vel_curva - (int)correcao;
      int v_dir = vel_curva + (int)correcao;

      mover(
        constrain(v_esq, 0, 255),
        constrain(v_dir, 0, 255)
      );

      break;
    }

    case ACAO_CURVA_DIR: {

      float correcao = calcular_pid(
        erro,
        Kp_curva,
        Ki_curva,
        Kd_curva
      );

      int v_esq = vel_curva - (int)correcao;
      int v_dir = vel_curva + (int)correcao;

      mover(
        constrain(v_esq, 0, 255),
        constrain(v_dir, 0, 255)
      );

      break;
    }

    case ACAO_CRUZAMENTO: {

      if (!em_cruzamento) {
        em_cruzamento = true;
        t_cruzamento = millis();
        integral = 0;
      }

      if (millis() - t_cruzamento < tempo_cruzamento_ms) {

        mover(
          vel_cruzamento,
          vel_cruzamento
        );

      } else {

        em_cruzamento = false;
      }

      break;
    }

    case ACAO_BUSCAR_LINHA: {

      integral = 0;

      if (erro_anterior >= 0) {

        mover(
          vel_busca,
          -vel_busca
        );

      } else {

        mover(
          -vel_busca,
          vel_busca
        );
      }

      break;
    }

    default:
      parar();
      break;
  }

  delayMicroseconds(500);
}
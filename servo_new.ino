

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <MPU6050_light.h>
#include <ESP32Servo.h>
#include <WiFiUdp.h>

// WIFI
const char* ssid = "iPhone";
const char* password = "12345678";

WebServer server(80);
WiFiUDP udp;

const char* laptopIP = "172.20.10.3";  
const int udpPort = 5005;

MPU6050 mpu(Wire);
Servo tail;

int servoPin = 13;
int angle = 90;

#define SERVO_FREQ 50

// ---------------- SERVO ----------------
void handleServo() {

  if (server.hasArg("angle")) {

    angle = server.arg("angle").toInt();
    angle = constrain(angle,60,120);
    tail.write(angle);
  }

  server.send(200,"text/plain","OK");
}

void setup() {

  Serial.begin(115200);

  Wire.begin(21,22);

  mpu.begin();
  delay(1000);
  mpu.calcOffsets(true,true);

  ESP32PWM::allocateTimer(0);
  tail.setPeriodHertz(SERVO_FREQ);
  tail.attach(servoPin,500,2400);
  tail.write(90);

  WiFi.begin(ssid,password);

  while(WiFi.status()!=WL_CONNECTED)
    delay(500);

  Serial.println("Connected");
  Serial.println(WiFi.localIP());

  server.on("/servo",handleServo);
  server.begin();
}

void loop() {

  server.handleClient();

  mpu.update();

  float yaw = mpu.getAngleZ();

  String msg = String(yaw,2);

  udp.beginPacket(laptopIP,udpPort);
  udp.print(msg);
  udp.endPacket();

  delay(40);  // 
}
#include <PubSubClient.h>
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <time.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "IsolationForest.h"

#define PIN_ULTRASONIC_TRIG 4
#define PIN_ULTRASONIC_ECHO 16
#define PIN_BUZZER          0
#define PIN_FLOW_SENSOR     35

#define PIN_SD_CS           5
#define PIN_SD_SCK          18
#define PIN_SD_MISO         19
#define PIN_SD_MOSI         23
#define PIN_SD_CD           34 // Card Detect (Wired, but optional for basic SD usage)

#define LOOP_DURATION_MS    5000

// WiFi
const char* ssid = "Wokwi-GUEST";
const char* password = "";

// MQTT
const char* mqttServer = "broker.hivemq.com"; // Refer to https://www.mqtt-dashboard.com/
const char* mqttTopic = "esp32/water/sensors";

// NTP
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 0; // Use 0 offset and let the dashboard handle it
const int daylightOffset_sec = 0; // Use 0 offset and let the dashboard handle it

// Directories
const char* aiToggleDir = "/ai_toggle.txt";
const char* sensorLogDir = "/sensor_log.csv";
const char* trainedModelDir = "/trained_model.bin";
const char* trainingWindowDir = "/training_window.txt";

WiFiClient espClient;
PubSubClient client(espClient);

IsolationForest iforest(10, 64);

void setup() {
  Serial.begin(115200);
  
  pinMode(PIN_ULTRASONIC_TRIG, OUTPUT);
  pinMode(PIN_ULTRASONIC_ECHO, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_FLOW_SENSOR, INPUT); 
  pinMode(PIN_SD_CD, INPUT);
  
  SPI.begin(PIN_SD_SCK, PIN_SD_MISO, PIN_SD_MOSI, PIN_SD_CS);
  Serial.println("[SD] Mounting SD Card...");

  if (SD.begin(PIN_SD_CS, SPI)) {
    Serial.println("[SD] SD Card mounted successfully!");
    
    initFiles();

    if (getAiToggleValue()) {
        applyIsolationForestTraining();
    }
  } else {
    Serial.println("[ERROR] SD Card mount failed!");
  }

  Serial.print("[WIFI] Connecting to WiFi");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("\n[WIFI] Connected to WiFi!");

  initTime();
  client.setServer(mqttServer, 1883);
}

void loop() {
  delay(LOOP_DURATION_MS);
  
  time_t epochTime = getTime();
  float distance = getDistanceToWater();
  int flowLPM = getFlowRate();
  
  sendAlert(distance, flowLPM);
  logToSD(epochTime, distance, flowLPM);
  mqttPublish(epochTime, distance, flowLPM);
}

void mqttPublish(time_t epochTime, float distance, int flow) {
  if (!client.connected()) {
    mqttReconnect();
  }

  String payloadStr = "{ \"timestamp\": " + String(epochTime) + 
                      ", \"distance_mm\": " + String(distance) + 
                      ", \"flow_lpm\": " + String(flow) + " }";
  const uint8_t* payload = (uint8_t*) payloadStr.c_str();
  
  // QoS 1
  if (client.publish(mqttTopic, payload, payloadStr.length(), false)) {
    Serial.print("[MQTT] Published payload: ");
    Serial.println(payloadStr);
  } else {
    Serial.println("[ERROR] Publish failed or timed out.");
  }
}

void mqttReconnect() {
  client.loop();

  while (!client.connected()) {
    Serial.println("[MQTT] Attempting MQTT connection...");

    if (client.connect(mqttTopic)) {
      Serial.println("[MQTT] MQTT connected!");
    } else {
      Serial.printf("[ERROR] MQTT failed, rc=%d try again in 3 seconds.\n", client.state());
      delay(3000);
    }
  }
}

bool getAiToggleValue() {
  File file = SD.open(aiToggleDir, FILE_READ);

  if (!file) {
    Serial.println("[ERROR] Failed to open file, using 'false' as fallback.");
    return false;
  }

  String fileContent = "";
  
  while (file.available()) {
    fileContent += (char)file.read();
  }

  file.close();
  fileContent.trim();

  return fileContent.equals("true");
}

float getDistanceToWater() {
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  
  long duration = pulseIn(PIN_ULTRASONIC_ECHO, HIGH);

  if (duration == 0) {
    return 0.0;
  }
  
  return (duration * 0.034) / 2.0;
}

time_t getTime() {
  struct tm timeinfo;

  if (getLocalTime(&timeinfo)) {
    return mktime(&timeinfo);
  }

  return 0; // Fallback
}

void logToSD(time_t epochTime, float distance, int flow) {
  struct tm timeinfo;

  File logFile = SD.open(sensorLogDir, FILE_APPEND);

  if (logFile) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%ld,%.2f,%d", epochTime, distance, flow);
    logFile.println(buffer);
    logFile.close();
    
    Serial.printf("[LOG] Timestamp: %ld | Distance: %.2f mm | Flow LPM: %d\n", epochTime, distance, flow);
  } else {
    Serial.println("[ERROR] Failed to open file.");
  }
}

void sendAlert(float distance, int flowLPM) {
  if (!getAiToggleValue()) { return; }

  std::vector<float> sample = {distance, (float)flowLPM};
  float score = iforest.score(sample);

  Serial.printf("[IFOREST] Anomaly Score: %.3f\n", score);

  if (score > 0.65) {
    Serial.println("[ALERT] Anomaly detected!");
    tone(PIN_BUZZER, 1000, 1000); 
  }
}

void ensureFileExists(const char* path, const char* content) {
  if (SD.exists(path)) {
    Serial.printf("[FILE] %s already exists.\n", path);
    return;
  }

  File file = SD.open(path, FILE_WRITE);

  if (file) {
    if (content && strlen(content) > 0) {
      file.println(content);
    }

    file.close();
    Serial.printf("[FILE] Created %s.\n", path);
  } else {
    Serial.printf("[ERROR] Failed to create %s!\n", path);
  }
}

void initFiles() {
  ensureFileExists(aiToggleDir, "false");
  ensureFileExists(sensorLogDir, "Timestamp,Distance(mm),Flow(LPM)");
  ensureFileExists(trainingWindowDir, "0,0");
}

void initTime() {
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  
  struct tm timeinfo;

  if (!getLocalTime(&timeinfo)) {
    Serial.println("[ERROR] Failed to obtain time from NTP. Retrying in 3 seconds");
    delay(3000);
  }

  Serial.println("[NTP] Time synchronized successfully!");
}

void applyIsolationForestTraining() {
  File configFile = SD.open(trainingWindowDir, FILE_READ);

  if (!configFile) {
    Serial.println("[INFO] No training window file found. Skipping training.");
    return;
  }

  long long start = 0;
  long long end = 0;
  sscanf(configFile.readString().c_str(), "%lld,%lld", &start, &end);
  configFile.close();

  if (start == 0 || end == 0) {
    Serial.println("[ERROR] Training window timestamps invalid or 0.");
    return;
  }

  time_t startEpoch = (time_t)start;
  time_t endEpoch = (time_t)end;

  Serial.printf("[IFOREST] Training window: %lld to %lld\n", (long long)startEpoch, (long long)endEpoch);
  trainFromLog(startEpoch, endEpoch);
}

void trainFromLog(time_t startEpoch, time_t endEpoch) {
  File logFile = SD.open(sensorLogDir, FILE_READ);

  if (!logFile) {
    Serial.println("[ERROR] Can't open log file for training.");
    return;
  }

  std::vector<std::vector<float>> dataset;
  logFile.readStringUntil('\n'); // Skip CSV header

  while (logFile.available()) {
    String line = logFile.readStringUntil('\n');
    if (line.length() < 10) continue;

    int firstComma = line.indexOf(',');
    int secondComma = line.indexOf(',', firstComma + 1);

    if (firstComma > 0 && secondComma > 0) {
      time_t ts = line.substring(0, firstComma).toInt();
      float distance = line.substring(firstComma + 1, secondComma).toFloat();
      float flow = line.substring(secondComma + 1).toFloat();

      if (ts >= startEpoch && ts <= endEpoch) {
        dataset.push_back({distance, flow});
      }
    }
  }

  logFile.close();

  if (dataset.empty()) {
    Serial.println("[ERROR] No data found in the training window.");
    return;
  }

  iforest.train(dataset);
  Serial.printf("[IFOREST] Trained successfully on %d samples ([Distance, Flow]).\n", dataset.size());
  iforest.save(trainedModelDir);
}


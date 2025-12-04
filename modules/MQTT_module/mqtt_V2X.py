# modules/mqtt_module.py
import json
import paho.mqtt.client as mqtt
import random
import time
import uuid
from threading import Thread

class MQTTClient:
    def __init__(self, broker="localhost", port=1883):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.message_times = {}
        self.sla_stats = {
            "latency_violations": 0,
            "total_messages": 0
        }
        self.incident_active = False
        self.traffic_active = False
        self.road_close_active = False
        self.incident_time = 0
        self.traffic_time = 0
        self.road_close_time = 0

        # Configura callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        """Callback per la connessione MQTT."""
        client.subscribe("traffic/incidents", qos=1)

    def on_message(self, client, userdata, msg):
        """Callback per la ricezione di messaggi MQTT."""
        try:
            payload = json.loads(msg.payload)
            recv_time = time.time()
            message_id = payload.get("message_id")
            
            # SLA: misurazione latenza
            if message_id in self.message_times:
                sent_time = self.message_times.pop(message_id)
                latency = recv_time - sent_time
                print(f"⏱ Latenza messaggio: {latency:.3f} secondi")
                self.sla_stats["total_messages"] += 1

                if latency > 1:
                    self.sla_stats["latency_violations"] += 1

                # QoE ogni 10 messaggi
                if self.sla_stats["total_messages"] % 10 == 0:
                    rate = self.sla_stats["latency_violations"] / self.sla_stats["total_messages"]
                    if rate > 0.2:
                        print("😕 QoE bassa: troppi ritardi")
                    else:
                        print("😊 QoE buona")
            
            incident_type = payload.get("type", "")

            if "incident" in incident_type:
                self.incident_active = True
                self.incident_time = recv_time
            elif "traffic" in incident_type:
                self.traffic_active = True
                self.traffic_time = recv_time
            elif "road closed" in incident_type:
                self.road_close_active = True
                self.road_close_time = recv_time
            elif "clear" in incident_type:
                if "incident" in payload.get("clear_type", ""):
                    self.incident_active = False
                if "traffic" in payload.get("clear_type", ""):
                    self.traffic_active = False

        except Exception as e:
            print(f"Errore nell'elaborazione del messaggio: {e}")

    def transmit(self, topic, data):
        """Trasmette dati via MQTT con QoS."""
        try:
            message_id = str(uuid.uuid4())
            data["message_id"] = message_id
            data["timestamp"] = time.time()
            self.message_times[message_id] = data["timestamp"]

            message = json.dumps(data)
            self.client.publish(topic, message, qos=1)
        except Exception as e:
            print(f"Errore nella trasmissione MQTT: {e}")

    def simulate_incoming_data(self, topic="traffic/incidents"):
        """Simula l'invio di dati MQTT per test."""
        incident_types = ["incident", "traffic", "road closed", "blockage"]
        location = ["A1", "A2", "B3", "C4"]
        
        while True:
            incident = random.choice(incident_types)
            place = random.choice(location)
            severity = random.randint(1, 5)
            
            data = {
                "type": incident,
                "location": place,
                "severity": severity
            }
            
            self.transmit(topic, data)
            time.sleep(3)

    def start(self):
        """Avvia il client MQTT e la simulazione in thread separati."""
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()
        
        # Avvia la simulazione in un thread separato
        simulator_thread = Thread(target=self.simulate_incoming_data)
        simulator_thread.daemon = True
        simulator_thread.start()

    def get_incident_status(self):
        """Restituisce lo stato corrente degli incidenti"""
        return {
            'incident_active': self.incident_active,
            'traffic_active': self.traffic_active,
            'road_close_active': self.road_close_active,
            'incident_time': self.incident_time,
            'traffic_time': self.traffic_time,
            'road_close_time': self.road_close_time
        }
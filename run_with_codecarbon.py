import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from camera_calibration import calib
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from parameter import ProcessingParams
import pandas as pd
import time
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
import csv
from detect_line import Line
import json
import os
import psutil
import threading
from datetime import datetime
import tempfile
import shutil

from modules.Object_detection_module.lane_detection_ultrafast import detect_lanes_in_frame
from modules.Object_detection_module.object_detection import load_yolo_model,detect_objects,draw_rectangles_and_text
from modules.Reconstruction_module.reconstruction import overlay_png,overlay_fixed_car_image
from modules.Object_detection_module.lane_detection_pipline import pipeline
from modules.MQTT_module.mqtt_V2X import MQTTClient
from modules.Interface_module.interface_user import add_to_frames_to_save, run_tkinter

# Classe migliorata per gestire il monitoraggio energetico dei moduli
# Aggiungi queste importazioni all'inizio del tuo file
try:
    import pynvml
    NVIDIA_GPU_AVAILABLE = True
except ImportError:
    NVIDIA_GPU_AVAILABLE = False
    print("pynvml non disponibile. Installa con: pip install pynvml")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("GPUtil non disponibile. Installa con: pip install GPUtil")

# Sostituisci la classe ImprovedModuleEnergyTracker esistente con questa versione estesa
class ImprovedModuleEnergyTracker:
    def __init__(self, output_dir="energy_reports"):
        self.output_dir = output_dir
        self.module_data = {}
        self.total_emissions = 0
        self.total_energy = 0
        self.lock = Lock()
        self.process = psutil.Process()
        self.baseline_cpu_percent = None
        self.baseline_memory_mb = None
        
        # Variabili per GPU tracking
        self.gpu_available = False
        self.gpu_count = 0
        self.baseline_gpu_utilization = []
        self.baseline_gpu_memory_mb = []
        self.baseline_gpu_power_w = []
        
        # Variabili per il tracking manuale
        self.current_module = None
        self.current_frame = None
        self.module_start_time = None
        self.start_cpu_times = None
        self.start_memory_info = None
        self.start_gpu_utilization = []
        self.start_gpu_memory_mb = []
        self.start_gpu_power_w = []
        
        # Crea directory per i report se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        # Inizializza GPU monitoring
        self._initialize_gpu_monitoring()
        
        # Inizializza il file CSV per i dati dettagliati (con colonne GPU)
        self.csv_filename = os.path.join(output_dir, f"module_energy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.csv_filename, "w", newline="") as file:
            writer = csv.writer(file)
            header = [
                "Frame", "Module", "Start_Time", "End_Time", "Duration_s", 
                "CPU_Usage_%", "Memory_Usage_MB", "Memory_Delta_MB", 
                "CPU_Time_User_s", "CPU_Time_System_s"
            ]
            
            # Aggiungi colonne GPU se disponibili
            if self.gpu_available:
                for gpu_id in range(self.gpu_count):
                    header.extend([
                        f"GPU_{gpu_id}_Utilization_%", 
                        f"GPU_{gpu_id}_Memory_MB", 
                        f"GPU_{gpu_id}_Memory_Delta_MB",
                        f"GPU_{gpu_id}_Power_W",
                        f"GPU_{gpu_id}_Power_Delta_W"
                    ])
                header.append("Total_GPU_Energy_Wh")
            
            header.append("Estimated_Total_Energy_Wh")
            writer.writerow(header)
        
        # Inizializza baseline delle risorse
        self._initialize_baseline()
    
    def _initialize_gpu_monitoring(self):
        """Inizializza il monitoraggio GPU"""
        try:
            if NVIDIA_GPU_AVAILABLE:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = True
                print(f"GPU NVIDIA rilevate: {self.gpu_count}")
                
                # Inizializza liste per ogni GPU
                for gpu_id in range(self.gpu_count):
                    self.baseline_gpu_utilization.append(0)
                    self.baseline_gpu_memory_mb.append(0)
                    self.baseline_gpu_power_w.append(0)
                    
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                self.gpu_count = len(gpus)
                self.gpu_available = self.gpu_count > 0
                print(f"GPU rilevate con GPUtil: {self.gpu_count}")
                
                # Inizializza liste per ogni GPU
                for gpu_id in range(self.gpu_count):
                    self.baseline_gpu_utilization.append(0)
                    self.baseline_gpu_memory_mb.append(0)
                    self.baseline_gpu_power_w.append(0)
            else:
                print("Nessuna libreria GPU disponibile")
                
        except Exception as e:
            print(f"Errore nell'inizializzazione GPU: {e}")
            self.gpu_available = False
    
    def _get_gpu_stats(self):
        """Ottieni statistiche GPU correnti"""
        if not self.gpu_available:
            return [], [], []
        
        utilizations = []
        memory_usage_mb = []
        power_usage_w = []
        
        try:
            if NVIDIA_GPU_AVAILABLE and self.gpu_count > 0:
                for gpu_id in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    
                    # Utilizzo GPU
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        utilizations.append(util.gpu)
                    except:
                        utilizations.append(0)
                    
                    # Memoria GPU
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_usage_mb.append(mem_info.used / 1024 / 1024)  # MB
                    except:
                        memory_usage_mb.append(0)
                    
                    # Potenza GPU
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        power_usage_w.append(power_mw / 1000.0)  # Converti mW in W
                    except:
                        power_usage_w.append(0)
                        
            elif GPUTIL_AVAILABLE and self.gpu_count > 0:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    utilizations.append(gpu.load * 100)  # GPUtil restituisce 0-1
                    memory_usage_mb.append(gpu.memoryUsed)  # Già in MB
                    # GPUtil non fornisce dati di potenza, usiamo stima
                    power_usage_w.append(gpu.load * 300)  # Stima: 300W max per GPU
                    
        except Exception as e:
            print(f"Errore nel leggere stats GPU: {e}")
            # Ritorna liste vuote della dimensione corretta
            for _ in range(self.gpu_count):
                utilizations.append(0)
                memory_usage_mb.append(0)
                power_usage_w.append(0)
        
        return utilizations, memory_usage_mb, power_usage_w
    
    def _initialize_baseline(self):
        """Inizializza i valori baseline per CPU, memoria e GPU"""
        try:
            # Baseline CPU e memoria (codice esistente)
            cpu_measurements = []
            memory_measurements = []
            
            for _ in range(5):
                cpu_measurements.append(self.process.cpu_percent())
                memory_measurements.append(self.process.memory_info().rss / 1024 / 1024)  # MB
                time.sleep(0.1)
            
            self.baseline_cpu_percent = sum(cpu_measurements) / len(cpu_measurements)
            self.baseline_memory_mb = sum(memory_measurements) / len(memory_measurements)
            
            # Baseline GPU
            if self.gpu_available:
                gpu_util_measurements = [[] for _ in range(self.gpu_count)]
                gpu_mem_measurements = [[] for _ in range(self.gpu_count)]
                gpu_power_measurements = [[] for _ in range(self.gpu_count)]
                
                for _ in range(5):
                    utils, mems, powers = self._get_gpu_stats()
                    for gpu_id in range(self.gpu_count):
                        if gpu_id < len(utils):
                            gpu_util_measurements[gpu_id].append(utils[gpu_id])
                            gpu_mem_measurements[gpu_id].append(mems[gpu_id])
                            gpu_power_measurements[gpu_id].append(powers[gpu_id])
                    time.sleep(0.1)
                
                # Calcola medie baseline per ogni GPU
                for gpu_id in range(self.gpu_count):
                    if gpu_util_measurements[gpu_id]:
                        self.baseline_gpu_utilization[gpu_id] = sum(gpu_util_measurements[gpu_id]) / len(gpu_util_measurements[gpu_id])
                        self.baseline_gpu_memory_mb[gpu_id] = sum(gpu_mem_measurements[gpu_id]) / len(gpu_mem_measurements[gpu_id])
                        self.baseline_gpu_power_w[gpu_id] = sum(gpu_power_measurements[gpu_id]) / len(gpu_power_measurements[gpu_id])
                
                print(f"Baseline - CPU: {self.baseline_cpu_percent:.2f}%, Memory: {self.baseline_memory_mb:.2f} MB")
                for gpu_id in range(self.gpu_count):
                    print(f"GPU {gpu_id} - Util: {self.baseline_gpu_utilization[gpu_id]:.2f}%, "
                          f"Mem: {self.baseline_gpu_memory_mb[gpu_id]:.2f} MB, "
                          f"Power: {self.baseline_gpu_power_w[gpu_id]:.2f} W")
            else:
                print(f"Baseline - CPU: {self.baseline_cpu_percent:.2f}%, Memory: {self.baseline_memory_mb:.2f} MB, GPU: N/A")
            
        except Exception as e:
            print(f"Errore nell'inizializzazione baseline: {e}")
            self.baseline_cpu_percent = 0
            self.baseline_memory_mb = 0
            self.baseline_gpu_utilization = [0] * self.gpu_count
            self.baseline_gpu_memory_mb = [0] * self.gpu_count
            self.baseline_gpu_power_w = [0] * self.gpu_count
    
    def start_module_tracking(self, module_name, frame_number=None):
        """Inizia il tracking manuale per un modulo specifico (esteso con GPU)"""
        with self.lock:
            if self.current_module is not None:
                print(f"Warning: Tracking già attivo per {self.current_module}. Fermando tracker precedente.")
                self._stop_current_tracking()
            
            try:
                self.current_module = module_name
                self.current_frame = frame_number
                self.module_start_time = time.time()
                
                # Cattura stato iniziale delle risorse CPU/memoria
                self.start_cpu_times = self.process.cpu_times()
                self.start_memory_info = self.process.memory_info()
                
                # Cattura stato iniziale GPU
                if self.gpu_available:
                    self.start_gpu_utilization, self.start_gpu_memory_mb, self.start_gpu_power_w = self._get_gpu_stats()
                else:
                    self.start_gpu_utilization = []
                    self.start_gpu_memory_mb = []
                    self.start_gpu_power_w = []
                
                return True
                
            except Exception as e:
                print(f"Errore nell'avvio del tracking per {module_name}: {e}")
                self._reset_tracking_state()
                return False
    
    def stop_module_tracking(self):
        """Ferma il tracking corrente e calcola i consumi (esteso con GPU)"""
        with self.lock:
            if self.current_module is None:
                return None
            
            try:
                end_time = time.time()
                duration = end_time - self.module_start_time
                
                # Cattura stato finale delle risorse CPU/memoria
                end_cpu_times = self.process.cpu_times()
                end_memory_info = self.process.memory_info()
                
                # Calcola differenze CPU/memoria (codice esistente)
                cpu_user_time = end_cpu_times.user - self.start_cpu_times.user
                cpu_system_time = end_cpu_times.system - self.start_cpu_times.system
                total_cpu_time = cpu_user_time + cpu_system_time
                
                current_memory_mb = end_memory_info.rss / 1024 / 1024  # MB
                start_memory_mb = self.start_memory_info.rss / 1024 / 1024  # MB
                memory_delta = current_memory_mb - start_memory_mb
                
                cpu_percent = (total_cpu_time / duration) * 100 if duration > 0 else 0
                
                # Cattura e calcola differenze GPU
                gpu_data = {}
                total_gpu_energy_wh = 0
                
                if self.gpu_available:
                    end_gpu_utilization, end_gpu_memory_mb, end_gpu_power_w = self._get_gpu_stats()
                    
                    for gpu_id in range(min(self.gpu_count, len(end_gpu_utilization))):
                        # Utilizzo GPU per questo modulo
                        start_util = self.start_gpu_utilization[gpu_id] if gpu_id < len(self.start_gpu_utilization) else 0
                        end_util = end_gpu_utilization[gpu_id]
                        avg_gpu_util = (start_util + end_util) / 2
                        
                        # Memoria GPU
                        start_mem = self.start_gpu_memory_mb[gpu_id] if gpu_id < len(self.start_gpu_memory_mb) else 0
                        end_mem = end_gpu_memory_mb[gpu_id]
                        gpu_memory_delta = end_mem - start_mem
                        
                        # Potenza GPU
                        start_power = self.start_gpu_power_w[gpu_id] if gpu_id < len(self.start_gpu_power_w) else 0
                        end_power = end_gpu_power_w[gpu_id]
                        avg_gpu_power = (start_power + end_power) / 2
                        gpu_power_delta = end_power - start_power
                        
                        # Calcola energia GPU per questa GPU
                        gpu_energy_wh = avg_gpu_power * (duration / 3600.0)  # Wh
                        total_gpu_energy_wh += gpu_energy_wh
                        
                        gpu_data[f"gpu_{gpu_id}"] = {
                            "utilization_percent": avg_gpu_util,
                            "memory_mb": end_mem,
                            "memory_delta_mb": gpu_memory_delta,
                            "power_w": avg_gpu_power,
                            "power_delta_w": gpu_power_delta,
                            "energy_wh": gpu_energy_wh
                        }
                
                # Stima consumo energetico CPU/memoria (codice esistente)
                cpu_memory_energy_wh = self._estimate_cpu_memory_energy_consumption(
                    cpu_percent, current_memory_mb, duration
                )
                
                # Energia totale
                total_estimated_energy_wh = cpu_memory_energy_wh + total_gpu_energy_wh
                
                # Salva i dati del modulo
                module_data = {
                    "module": self.current_module,
                    "frame": self.current_frame,
                    "start_time": self.module_start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_mb": current_memory_mb,
                    "memory_delta_mb": memory_delta,
                    "cpu_time_user": cpu_user_time,
                    "cpu_time_system": cpu_system_time,
                    "cpu_memory_energy_wh": cpu_memory_energy_wh,
                    "gpu_data": gpu_data,
                    "total_gpu_energy_wh": total_gpu_energy_wh,
                    "estimated_energy_wh": total_estimated_energy_wh
                }
                
                # Aggiungi ai dati totali
                self.total_energy += total_estimated_energy_wh / 1000  # Converti in kWh
                
                # Salva nel CSV
                self._save_to_csv_with_gpu(module_data)
                
                # Reset dello stato di tracking
                self._reset_tracking_state()
                
                return module_data
                
            except Exception as e:
                print(f"Errore nel fermare il tracker: {e}")
                self._reset_tracking_state()
                return None
    
    def _estimate_cpu_memory_energy_consumption(self, cpu_percent, memory_mb, duration_s):
        """Stima il consumo energetico basato su CPU e memoria (metodo esistente rinominato)"""
        try:
            # Parametri di stima (possono essere calibrati)
            cpu_cores = psutil.cpu_count()
            watts_per_cpu_core_100_percent = 4.0  # Watt per core al 100%
            watts_per_gb_memory = 0.3  # Watt per GB di memoria
            
            # Calcola consumo CPU
            cpu_power_w = (cpu_percent / 100.0) * (watts_per_cpu_core_100_percent * cpu_cores / 100.0)
            
            # Calcola consumo memoria
            memory_gb = memory_mb / 1024
            memory_power_w = memory_gb * watts_per_gb_memory
            
            # Consumo totale in Wh
            total_power_w = cpu_power_w + memory_power_w
            energy_wh = total_power_w * (duration_s / 3600.0)  # Converti secondi in ore
            
            return energy_wh
            
        except Exception as e:
            print(f"Errore nel calcolo energetico CPU/memoria: {e}")
            return 0.0
    
    def _save_to_csv_with_gpu(self, data):
        """Salva i dati nel file CSV inclusi dati GPU"""
        try:
            with open(self.csv_filename, "a", newline="") as file:
                writer = csv.writer(file)
                
                # Dati base
                row = [
                    data["frame"],
                    data["module"],
                    datetime.fromtimestamp(data["start_time"]).strftime('%H:%M:%S.%f'),
                    datetime.fromtimestamp(data["end_time"]).strftime('%H:%M:%S.%f'),
                    f"{data['duration']:.6f}",
                    f"{data['cpu_usage_percent']:.2f}",
                    f"{data['memory_usage_mb']:.2f}",
                    f"{data['memory_delta_mb']:.2f}",
                    f"{data['cpu_time_user']:.6f}",
                    f"{data['cpu_time_system']:.6f}"
                ]
                
                # Aggiungi dati GPU se disponibili
                if self.gpu_available and data.get("gpu_data"):
                    for gpu_id in range(self.gpu_count):
                        gpu_key = f"gpu_{gpu_id}"
                        if gpu_key in data["gpu_data"]:
                            gpu_info = data["gpu_data"][gpu_key]
                            row.extend([
                                f"{gpu_info['utilization_percent']:.2f}",
                                f"{gpu_info['memory_mb']:.2f}",
                                f"{gpu_info['memory_delta_mb']:.2f}",
                                f"{gpu_info['power_w']:.2f}",
                                f"{gpu_info['power_delta_w']:.2f}"
                            ])
                        else:
                            # GPU non disponibile per questo measurement
                            row.extend(["0.00", "0.00", "0.00", "0.00", "0.00"])
                    
                    row.append(f"{data['total_gpu_energy_wh']:.8f}")
                
                row.append(f"{data['estimated_energy_wh']:.8f}")
                writer.writerow(row)
                
        except Exception as e:
            print(f"Errore nel salvare i dati CSV: {e}")
    
    def _stop_current_tracking(self):
        """Helper method per fermare il tracking corrente senza lock"""
        if self.current_module is not None:
            self.stop_module_tracking()
    
    def _reset_tracking_state(self):
        """Reset dello stato di tracking (esteso con GPU)"""
        self.current_module = None
        self.current_frame = None
        self.module_start_time = None
        self.start_cpu_times = None
        self.start_memory_info = None
        self.start_gpu_utilization = []
        self.start_gpu_memory_mb = []
        self.start_gpu_power_w = []
    
    def get_summary(self):
        """Ritorna un riassunto del consumo energetico (esteso con GPU)"""
        summary = {
            "total_energy_kwh": self.total_energy,
            "total_emissions_kg": self.total_emissions,
            "csv_report": self.csv_filename,
            "baseline_cpu_percent": self.baseline_cpu_percent,
            "baseline_memory_mb": self.baseline_memory_mb,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count
        }
        
        if self.gpu_available:
            summary["baseline_gpu_utilization"] = self.baseline_gpu_utilization
            summary["baseline_gpu_memory_mb"] = self.baseline_gpu_memory_mb
            summary["baseline_gpu_power_w"] = self.baseline_gpu_power_w
        
        return summary
    
    def save_summary_report(self):
        """Salva un report riassuntivo in JSON (esteso con GPU)"""
        summary = self.get_summary()
        summary["timestamp"] = datetime.now().isoformat()
        summary["total_frames_processed"] = self._count_frames_in_csv()
        
        json_filename = os.path.join(self.output_dir, f"energy_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            with open(json_filename, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Report riassuntivo salvato in: {json_filename}")
            return json_filename
        except Exception as e:
            print(f"Errore nel salvare il report riassuntivo: {e}")
            return None
    
    def _count_frames_in_csv(self):
        """Conta il numero di frame processati nel CSV"""
        try:
            with open(self.csv_filename, "r") as file:
                return sum(1 for line in file) - 1  # -1 per l'header
        except Exception as e:
            print(f"Errore nel contare i frame: {e}")
            return 0
    
    def cleanup(self):
        """Pulisce le risorse e finalizza il tracking (esteso con GPU)"""
        with self.lock:
            if self.current_module is not None:
                print(f"Finalizing tracking for {self.current_module}")
                self.stop_module_tracking()
            
            # Cleanup GPU monitoring
            if self.gpu_available and NVIDIA_GPU_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass


# Classe alternativa usando CodeCarbon con gestione migliorata dei lock
class SafeCodeCarbonTracker:
    def __init__(self, output_dir="energy_reports"):
        self.output_dir = output_dir
        self.lock = Lock()
        self.trackers = {}
        self.current_tracker_id = None
        
        # Crea directory temporanea unica per evitare conflitti
        self.temp_dir = tempfile.mkdtemp(prefix="codecarbon_", dir=output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV per raccogliere tutti i risultati
        self.master_csv = os.path.join(output_dir, f"codecarbon_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.master_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Frame", "Module", "Duration_s", "Energy_kWh", 
                "Emissions_kg", "CPU_Count", "RAM_Total_GB"
            ])
    
    def start_module_tracking(self, module_name, frame_number=None):
        """Avvia tracking con CodeCarbon in modo sicuro"""
        with self.lock:
            tracker_id = f"{module_name}_{frame_number}_{int(time.time()*1000)}"
            
            try:
                # Pulisci eventuali tracker precedenti
                self._cleanup_previous_tracker()
                
                # Crea tracker con directory temporanea unica
                tracker = OfflineEmissionsTracker(
                    project_name=f"module_{module_name}",
                    output_dir=self.temp_dir,
                    country_iso_code="ITA",
                    log_level="ERROR",
                    save_to_file=True,
                    tracking_mode="process"  # Track only this process
                )
                
                self.trackers[tracker_id] = {
                    "tracker": tracker,
                    "module": module_name,
                    "frame": frame_number,
                    "start_time": time.time()
                }
                
                self.current_tracker_id = tracker_id
                tracker.start()
                
                return True
                
            except Exception as e:
                print(f"Errore CodeCarbon per {module_name}: {e}")
                # Fallback al tracker manuale
                return False
    
    def stop_module_tracking(self):
        """Ferma il tracking CodeCarbon corrente"""
        with self.lock:
            if self.current_tracker_id is None:
                return None
            
            try:
                tracker_info = self.trackers.get(self.current_tracker_id)
                if not tracker_info:
                    return None
                
                tracker = tracker_info["tracker"]
                end_time = time.time()
                duration = end_time - tracker_info["start_time"]
                
                # Ferma il tracker
                emissions_data = tracker.stop()
                
                # Estrai i dati
                energy_kwh = getattr(emissions_data, 'energy_consumed', 0) or 0
                emissions_kg = getattr(emissions_data, 'emissions', 0) or 0
                cpu_count = getattr(emissions_data, 'cpu_count', 0) or 0
                ram_total = getattr(emissions_data, 'ram_total_size', 0) or 0
                
                result = {
                    "module": tracker_info["module"],
                    "frame": tracker_info["frame"],
                    "duration": duration,
                    "energy_kwh": energy_kwh,
                    "emissions_kg": emissions_kg,
                    "cpu_count": cpu_count,
                    "ram_total_gb": ram_total
                }
                
                # Salva nel CSV master
                self._save_to_master_csv(result)
                
                # Pulisci il tracker
                del self.trackers[self.current_tracker_id]
                self.current_tracker_id = None
                
                return result
                
            except Exception as e:
                print(f"Errore nel fermare CodeCarbon: {e}")
                self._cleanup_previous_tracker()
                return None
    
    def _cleanup_previous_tracker(self):
        """Pulisce il tracker precedente se esiste"""
        if self.current_tracker_id and self.current_tracker_id in self.trackers:
            try:
                tracker_info = self.trackers[self.current_tracker_id]
                tracker_info["tracker"].stop()
                del self.trackers[self.current_tracker_id]
            except Exception as e:
                print(f"Errore nella pulizia tracker: {e}")
        
        self.current_tracker_id = None
    
    def _save_to_master_csv(self, data):
        """Salva nel CSV master"""
        try:
            with open(self.master_csv, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    data["frame"],
                    data["module"],
                    f"{data['duration']:.6f}",
                    f"{data['energy_kwh']:.8f}",
                    f"{data['emissions_kg']:.8f}",
                    data["cpu_count"],
                    f"{data['ram_total_gb']:.2f}"
                ])
        except Exception as e:
            print(f"Errore nel salvare CSV master: {e}")
    
    def cleanup(self):
        """Pulisce tutti i tracker e rimuove directory temporanea"""
        with self.lock:
            # Ferma tutti i tracker attivi
            for tracker_id, tracker_info in list(self.trackers.items()):
                try:
                    tracker_info["tracker"].stop()
                except:
                    pass
            
            self.trackers.clear()
            self.current_tracker_id = None
            
            # Rimuovi directory temporanea
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass

# Variabili globali (mantieni le tue variabili esistenti)
type_of_inconvenient = None
last_overlay_time = 0
incident_active = False
traffic_active = False
road_close_active = False
road_close_time = 0
incident_time = 0
traffic_time = 0
icon_width = None
spacing = 20
total_icons_width = None
background_color = (200, 200, 200)
background_height = None
background_width = None
x_start = None
x_end = None
center_x = 0
img_back = True
icon = True
gray_background = None
original_gray_background = None
pipeline_chek = False
frames_to_save = []
MAX_FRAMES_TO_SAVE = 50
detected_front_vehicles = []
detected_vehicles = {}
vehicle_counter = 0
current_offsets = {}
target_offsets = {}
destra = False
sinistra = False
use_car_fix = False
performance_data = pd.DataFrame(columns=['Frame', 'Processing Time'])

# Inizializza il tracker energetico migliorato
energy_tracker = ImprovedModuleEnergyTracker()

# Se vuoi provare CodeCarbon con gestione migliorata, decommenta questa riga:
# energy_tracker = SafeCodeCarbonTracker()

def process_frame(params, mqtt_client=None, frame_number=None):
    """Funzione di processamento frame con tracking energetico migliorato"""
    global vehicle_counter, detected_front_vehicles, detected_vehicles, destra, sinistra, energy_tracker

    # Estrai parametri (mantieni il tuo codice esistente)
    frame = params.frame_resized
    yolo_model = params.yolo_model
    window_scale_factor = params.window_scale_factor
    car_fix = params.car_fix
    car_fix2 = params.car_fix2
    car_back_img = params.car_back_img
    car_back_imgS = params.car_back_imgS
    car_front_imgS = params.car_front_imgS
    car_front_img = params.car_front_img
    stop_img = params.stop_img
    mtx = params.mtx
    dist = params.dist
    focal_length_px = params.focal_length_px
    vehicle_height_m = params.vehicle_height_m
    moto_back_img = params.moto_back_img
    moto_back_imgS = params.moto_back_imgS
    car_fix_curve_left = params.car_fix_curve_left
    car_fix_curve_right = params.car_fix_curve_right
    car_fix_move = params.car_fix_move
    car_back_imgM = params.car_back_imgM
    car_front_imgM = params.car_front_imgM
    moto_back_imgM = params.moto_back_imgM
    car_fix2_move = params.car_fix2_move
    car_fix_curve_left_move = params.car_fix_curve_left_move
    car_fix_curve_right_move = params.car_fix_curve_right_move
    truck_back_img = params.truck_back_img
    truck_back_imgS = params.truck_back_imgS
    truck_back_imgM = params.truck_back_imgM
    traffic = params.traffic
    accident = params.accident
    road_close = params.road_close

    detected_vehicles.clear()
    vehicle_counter = 0
    
    global icon_width, total_icons_width, background_height, background_width, x_start, x_end, gray_background, img_back, original_gray_background, background_color, center_x
    global incident_active, traffic_active, road_close_active, road_close_time, incident_time, traffic_time
    global icon_width, total_icons_width, background_width, icon, center_x, spacing, destra, sinistra, use_car_fix, use_car_fix

    # MODULO 1: Lane Detection
    energy_tracker.start_module_tracking("lane_detection", frame_number)
    
    if (pipeline_chek):
        img, lane, parameters = pipeline(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mtx, dist, "giorno") #giorno,video
    else:
        img, lane, parameters, mask = detect_lanes_in_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    lane_detection_data = energy_tracker.stop_module_tracking()
    if lane_detection_data:
        print(f"Frame {frame_number} - Lane Detection:")
        print(f"  CPU/Memory: {lane_detection_data.get('cpu_memory_energy_wh', 0):.6f} Wh")
        if lane_detection_data.get('gpu_data'):
            print(f"  GPU Total: {lane_detection_data.get('total_gpu_energy_wh', 0):.6f} Wh")
            for gpu_id, gpu_info in lane_detection_data['gpu_data'].items():
                print(f"  {gpu_id.upper()}: {gpu_info['utilization_percent']:.1f}% util, {gpu_info['energy_wh']:.6f} Wh")
        print(f"  Total: {lane_detection_data['estimated_energy_wh']:.6f} Wh, {lane_detection_data['duration']:.4f}s")

    # MODULO 2: Car Overlay and Reconstruction
    energy_tracker.start_module_tracking("car_overlay_reconstruction", frame_number)
    
    final_image, background_height, x_start, x_end, original_gray_background, fixed_image, destra, sinistra, use_car_fix = overlay_fixed_car_image(
        car_fix, car_fix2, car_fix_curve_left, car_fix_curve_right,
        window_scale_factor, parameters, car_fix_move, car_fix2_move,
        car_fix_curve_left_move, car_fix_curve_right_move, traffic, accident, road_close,
        mqtt_client.get_incident_status()['incident_active'],
        mqtt_client.get_incident_status()['traffic_active'],
        mqtt_client.get_incident_status()['road_close_active'],
        mqtt_client.get_incident_status()['road_close_time'],
        mqtt_client.get_incident_status()['incident_time'],
        mqtt_client.get_incident_status()['traffic_time'],
        icon_width, total_icons_width, background_width, icon, center_x, spacing,
        destra, sinistra, use_car_fix, original_gray_background,
        background_height, x_start, x_end, frame
    )
    
    overlay_data = energy_tracker.stop_module_tracking()
    if overlay_data:
        print(f"Frame {frame_number} - Car Overlay:")
        print(f"  CPU/Memory: {overlay_data.get('cpu_memory_energy_wh', 0):.6f} Wh")
        if overlay_data.get('gpu_data'):
            print(f"  GPU Total: {overlay_data.get('total_gpu_energy_wh', 0):.6f} Wh")
            for gpu_id, gpu_info in overlay_data['gpu_data'].items():
                print(f"  {gpu_id.upper()}: {gpu_info['utilization_percent']:.1f}% util, {gpu_info['energy_wh']:.6f} Wh")
        print(f"  Total: {overlay_data['estimated_energy_wh']:.6f} Wh, {overlay_data['duration']:.4f}s")

    # MODULO 3: Object Detection (YOLO)
    energy_tracker.start_module_tracking("object_detection_yolo", frame_number)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = detect_objects(img, yolo_model)
    
    yolo_data = energy_tracker.stop_module_tracking()
    if yolo_data:
        print(f"Frame {frame_number} - YOLO Detection:")
        print(f"  CPU/Memory: {yolo_data.get('cpu_memory_energy_wh', 0):.6f} Wh")
        if yolo_data.get('gpu_data'):
            print(f"  GPU Total: {yolo_data.get('total_gpu_energy_wh', 0):.6f} Wh")
            for gpu_id, gpu_info in yolo_data['gpu_data'].items():
                print(f"  {gpu_id.upper()}: {gpu_info['utilization_percent']:.1f}% util, {gpu_info['energy_wh']:.6f} Wh")
        print(f"  Total: {yolo_data['estimated_energy_wh']:.6f} Wh, {yolo_data['duration']:.4f}s")

    # MODULO 4: Post-processing and Vehicle Analysis
    energy_tracker.start_module_tracking("vehicle_analysis_postprocessing", frame_number)
    
    names = yolo_model.names
    distance_factor = vehicle_height_m * focal_length_px

    rectangles = []
    texts = []
    detected_vehicles = {}

    mqtt_data = {
        "frame_id": params.frame_id if hasattr(params, "frame_id") else None,
        "vehicles": []
    }

    for j in range(len(output)):
        output_j = output[j]
        label = names[int(output_j[5])]
        confidence = round(output_j[4].item(), 2)
        coordinates = output_j[:4].int().tolist()
        xmin, ymin, xmax, ymax = coordinates

        rectangles.append(((xmin, ymin), (xmax, ymax)))
        texts.append(((xmin, ymin - 10), f'{label} {confidence}'))

        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        bbox_height_px = ymax - ymin
        distance_m = distance_factor / bbox_height_px
        distance_text = f'{distance_m:.2f}m'
        texts.append(((center_x, center_y), distance_text))

        vehicle_counter = len(detected_vehicles) + 1
        detected_vehicles[vehicle_counter] = (xmin, ymin, xmax, ymax)

        overlay_png(
            final_image, coordinates, label, window_scale_factor, car_back_img, car_back_imgS,
            car_front_imgS, car_front_img, stop_img, confidence, moto_back_img, moto_back_imgS,
            distance_m, car_back_imgM, car_front_imgM, moto_back_imgM, truck_back_img, truck_back_imgS, truck_back_imgM,
            destra, sinistra
        )

        mqtt_data["vehicles"].append({
            "id": vehicle_counter,
            "label": label,
            "confidence": confidence,
            "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "center": {"x": center_x, "y": center_y},
            "distance_m": round(distance_m, 2)
        })

    draw_rectangles_and_text(img, rectangles, texts)
    
    postprocessing_data = energy_tracker.stop_module_tracking()
    if postprocessing_data:
        print(f"Frame {frame_number} - Post-processing:")
        print(f"  CPU/Memory: {postprocessing_data.get('cpu_memory_energy_wh', 0):.6f} Wh")
        if postprocessing_data.get('gpu_data'):
            print(f"  GPU Total: {postprocessing_data.get('total_gpu_energy_wh', 0):.6f} Wh")
            for gpu_id, gpu_info in postprocessing_data['gpu_data'].items():
                print(f"  {gpu_id.upper()}: {gpu_info['utilization_percent']:.1f}% util, {gpu_info['energy_wh']:.6f} Wh")
        print(f"  Total: {postprocessing_data['estimated_energy_wh']:.6f} Wh, {postprocessing_data['duration']:.4f}s")

    # MODULO 5: MQTT Communication
    energy_tracker.start_module_tracking("mqtt_communication", frame_number)
    
    rows, cols = img.shape[:2]

    if mqtt_client:
        mqtt_client.transmit("intelligent-driving/vehicles", mqtt_data)
    
    mqtt_comm_data = energy_tracker.stop_module_tracking()
    if mqtt_comm_data:
        print(f"Frame {frame_number} - MQTT Comm:")
        print(f"  CPU/Memory: {mqtt_comm_data.get('cpu_memory_energy_wh', 0):.6f} Wh")
        if mqtt_comm_data.get('gpu_data'):
            print(f"  GPU Total: {mqtt_comm_data.get('total_gpu_energy_wh', 0):.6f} Wh")
            for gpu_id, gpu_info in mqtt_comm_data['gpu_data'].items():
                print(f"  {gpu_id.upper()}: {gpu_info['utilization_percent']:.1f}% util, {gpu_info['energy_wh']:.6f} Wh")
        print(f"  Total: {mqtt_comm_data['estimated_energy_wh']:.6f} Wh, {mqtt_comm_data['duration']:.4f}s")
    return final_image, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_performance_measure(frame_number, processing_time):
    global performance_data
    new_row = pd.DataFrame({'Frame': [frame_number], 'Processing Time': [processing_time]})
    performance_data = pd.concat([performance_data, new_row], ignore_index=True)


def main():
    global energy_tracker
    
    try:
        # Il tuo codice main esistente con piccole modifiche...
        yolo_model = load_yolo_model()

        video_paths = [
            'test_video/project_video.mp4',
            'Lane_detect/advanced-lane-detection-for-self-driving-cars-master/harder_challenge_video.mp4',
            'test_video/prova1.mp4',
            '/media/vrlab/video/video/output_compressed_video.mp4',
            '/media/vrlab/video/video/normal.mp4',
            '/media/vrlab/video/video/pioggia_2224x1080.mp4',
            '/media/vrlab/video/video/output_2224x1080.mp4',
            'Lane_detect/advanced-lane-detection-for-self-driving-cars-master/challenge_video.mp4',
            'videoStrada2.mp4'
        ]

        video_path = video_paths[6]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Errore nell'apertura del video.")
            return

        # Configurazioni video
        window_scale_factor = 1/2
        original_width = 960
        original_height = 540
        new_width = 960
        new_height = 540

        scale_x = new_width / original_width
        scale_y = new_height / original_height
        window_scale_factor = (scale_x + scale_y) / 2

        # ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=4)
        window_scale_factor = 1/2

        # Carica e ridimensiona tutte le immagini (mantieni il tuo codice esistente)
        car_back_img = cv2.imread('img_Project/car_back2.png', cv2.IMREAD_UNCHANGED)
        car_front_img = cv2.imread('img_Project/car_front2.png', cv2.IMREAD_UNCHANGED)
        stop_img = cv2.imread('img_Project/stop.png', cv2.IMREAD_UNCHANGED)
        moto_back = cv2.imread('img_Project/moto_back.png', cv2.IMREAD_UNCHANGED)
        truck_back = cv2.imread('img_Project/truck_back.png', cv2.IMREAD_UNCHANGED)

        car_back_img_original = car_back_img.copy()
        car_front_img_original = car_front_img.copy()
        truck_back_img_original = truck_back.copy()

        # Ridimensionamenti (mantieni il tuo codice)
        car_back_img = cv2.resize(car_back_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        car_back_imgM = cv2.resize(car_back_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        car_back_imgS = cv2.resize(car_back_img_original, (int(170 * window_scale_factor), int(170 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        car_front_img = cv2.resize(car_front_img_original, (int(80 * window_scale_factor), int(80 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        car_front_imgM = cv2.resize(car_front_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        car_front_imgS = cv2.resize(car_front_img_original, (int(170 * window_scale_factor), int(170 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        moto_back_img = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        moto_back_imgM = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        moto_back_imgS = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        truck_back_img = cv2.resize(truck_back_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        truck_back_imgM = cv2.resize(truck_back_img_original, (int(160 * window_scale_factor), int(160 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        truck_back_imgS = cv2.resize(truck_back_img_original, (int(200 * window_scale_factor), int(200 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        # Carica immagini auto fisse
        car_fix = cv2.imread('img_Project/carline.png', cv2.IMREAD_UNCHANGED)
        car_fix = cv2.resize(car_fix, (int(450 * window_scale_factor), int(450 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        car_fix_move = cv2.imread('img_Project/carline2.png', cv2.IMREAD_UNCHANGED)
        car_fix_move = cv2.resize(car_fix_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        car_fix2 = cv2.imread('img_Project/no_carline.png', cv2.IMREAD_UNCHANGED)
        car_fix2 = cv2.resize(car_fix2, (int(450 * window_scale_factor), int(450 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        car_fix2_move = cv2.imread('img_Project/no_carline2.png', cv2.IMREAD_UNCHANGED)
        car_fix2_move = cv2.resize(car_fix2_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        car_fix_curve_left = cv2.imread('img_Project/carline_left.png', cv2.IMREAD_UNCHANGED)
        car_fix_curve_left = cv2.resize(car_fix_curve_left, (int(450 * window_scale_factor), int(450 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        car_fix_curve_left_move = cv2.imread('img_Project/carline_left2.png', cv2.IMREAD_UNCHANGED)
        car_fix_curve_left_move = cv2.resize(car_fix_curve_left_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        car_fix_curve_right = cv2.imread('img_Project/carline_right.png', cv2.IMREAD_UNCHANGED)
        car_fix_curve_right = cv2.resize(car_fix_curve_right, (int(450 * window_scale_factor), int(450 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        car_fix_curve_right_move = cv2.imread('img_Project/carline_right2.png', cv2.IMREAD_UNCHANGED)
        car_fix_curve_right_move = cv2.resize(car_fix_curve_right_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        # Icone
        accident = cv2.imread('img_Project/accident.png', cv2.IMREAD_UNCHANGED)
        accident = cv2.resize(accident, (int(100 * window_scale_factor), int(100 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        traffic = cv2.imread('img_Project/traffic-jam.png', cv2.IMREAD_UNCHANGED)
        traffic = cv2.resize(traffic, (int(100 * window_scale_factor), int(100 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        road_close = cv2.imread('img_Project/road.png', cv2.IMREAD_UNCHANGED)
        road_close = cv2.resize(road_close, (int(100 * window_scale_factor), int(100 * window_scale_factor)), interpolation=cv2.INTER_AREA)

        # Calibrazione camera
        mtx, dist = calib()
        focal_length_px = mtx[1, 1]
        vehicle_height_m = 1.5

        # Thread Tkinter
        tkinter_thread = Thread(target=run_tkinter)
        tkinter_thread.start()
        
        frame_number = 0

        # CSV per label manuali
        csv_filename = "manual_labels.csv"
        with open(csv_filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Label"])

        # MQTT Client
        mqtt_client = MQTTClient(broker="127.0.0.1", port=1883)
        mqtt_client.start()

        print("=== Avvio monitoraggio energetico migliorato ===")
        print(f"Report CSV: {energy_tracker.csv_filename}")

        # Loop principale per processamento frame
        while True:
            if video_path is not None:
                ret, frame = cap.read()
                if not ret:
                    print("Fine del video raggiunta")
                    break
            else:
                frame = cv2.imread("runs/detect/predict5/t.png")
                if frame is None:
                    print("Errore nel caricamento dell'immagine")
                    break

            # Ridimensiona frame
            add_to_frames_to_save(frames_to_save)
            frame_resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)

            start_time = time.time()

            # Crea parametri per il processamento
            params = ProcessingParams(
                cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), yolo_model, window_scale_factor, 
                car_fix, car_fix2, car_back_img, car_back_imgS, car_front_imgS, car_front_img, 
                stop_img, mtx, dist, focal_length_px, vehicle_height_m, moto_back_img,
                moto_back_imgS, car_fix_curve_left, car_fix_curve_right, car_fix_move, 
                car_back_imgM, car_front_imgM, moto_back_imgM, car_fix2_move, 
                car_fix_curve_left_move, car_fix_curve_right_move, truck_back_img, 
                truck_back_imgM, truck_back_imgS, traffic, accident, road_close
            )

            # Processa frame
            future = executor.submit(process_frame, params, mqtt_client, frame_number)
            gray_background, img = future.result()

            # Concatena immagini per visualizzazione
            height = gray_background.shape[0]
            width = gray_background.shape[1]
            img_resized = cv2.resize(img, (width, height))
            concatenated_img = np.concatenate((gray_background, img_resized), axis=1)
            
            frames_to_save.append(concatenated_img.copy())
            
            end_time = time.time()
            processing_time = end_time - start_time

            # Visualizza risultati
            cv2.namedWindow('Object Detection Overlay', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Object Detection Overlay', int(img.shape[1]), int(img.shape[0]))
            cv2.imshow('Object Detection Overlay', concatenated_img)

            # Aggiungi misura prestazionale
            add_performance_measure(frame_number, processing_time)

            # Stampa statistiche ogni 10 frame
            if frame_number % 10 == 0:
                summary = energy_tracker.get_summary()
                print(f"\n=== Frame {frame_number} - Statistiche Energetiche ===")
                print(f"Energia totale: {summary['total_energy_kwh']:.6f} kWh")
                
                # Statistiche CPU/Memoria
                if hasattr(energy_tracker, 'baseline_cpu_percent'):
                    print(f"CPU baseline: {summary.get('baseline_cpu_percent', 0):.2f}%")
                    print(f"Memory baseline: {summary.get('baseline_memory_mb', 0):.2f} MB")
                
                # Statistiche GPU se disponibili
                if summary.get('gpu_available', False):
                    print(f"GPU disponibili: {summary.get('gpu_count', 0)}")
                    for gpu_id in range(summary.get('gpu_count', 0)):
                        if gpu_id < len(summary.get('baseline_gpu_utilization', [])):
                            print(f"GPU {gpu_id} baseline - "
                                  f"Util: {summary['baseline_gpu_utilization'][gpu_id]:.2f}%, "
                                  f"Mem: {summary['baseline_gpu_memory_mb'][gpu_id]:.2f} MB, "
                                  f"Power: {summary['baseline_gpu_power_w'][gpu_id]:.2f} W")
                else:
                    print("GPU: Non disponibile")

            frame_number += 1

            # Esci con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Uscita richiesta dall'utente")
                break

            # Limita per test (opzionale)
            # if frame_number >= 100:  # Limita a 100 frame per test
            #     print("Limite frame raggiunto per test")
            #     break

    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

    finally:
            # Cleanup finale
            print("\n=== Cleanup e generazione report finali ===")
            
            try:
                # Salva performance data
                performance_data.to_csv('performance_data_video.csv', index=False)
                print("Dati delle performance salvati in performance_data_video.csv")
                
                # Cleanup energy tracker
                energy_tracker.cleanup()
                
                # Salva report finale
                energy_tracker.save_summary_report()
                summary = energy_tracker.get_summary()
                
                print(f"\n=== RIASSUNTO FINALE CONSUMI ENERGETICI ===")
                print(f"Energia totale consumata: {summary['total_energy_kwh']:.6f} kWh")
                print(f"Emissioni totali CO2: {summary['total_emissions_kg']:.6f} kg")
                print(f"Report dettagliato CSV: {summary['csv_report']}")
                print(f"Frame totali processati: {frame_number}")
                
                # Statistiche GPU
                if summary.get('gpu_available', False):
                    print(f"GPU monitorate: {summary.get('gpu_count', 0)}")
                    if summary.get('gpu_count', 0) > 0:
                        print("Baseline GPU:")
                        for gpu_id in range(summary.get('gpu_count', 0)):
                            if gpu_id < len(summary.get('baseline_gpu_utilization', [])):
                                print(f"  GPU {gpu_id}: {summary['baseline_gpu_utilization'][gpu_id]:.2f}% util, "
                                      f"{summary['baseline_gpu_memory_mb'][gpu_id]:.2f} MB mem, "
                                      f"{summary['baseline_gpu_power_w'][gpu_id]:.2f} W power")
                else:
                    print("GPU: Monitoraggio non disponibile")
                
                if frame_number > 0:
                    print(f"Energia media per frame: {(summary['total_energy_kwh']/frame_number)*1000:.3f} Wh/frame")
                
                # Cleanup risorse
                if 'cap' in locals():
                    cap.release()
                cv2.destroyAllWindows()
                
                if 'mqtt_client' in locals():
                    mqtt_client.stop()
                
                if 'executor' in locals():
                    executor.shutdown(wait=True)
                    
            except Exception as cleanup_error:
                print(f"Errore durante il cleanup: {cleanup_error}")


if __name__ == "__main__":
    main()
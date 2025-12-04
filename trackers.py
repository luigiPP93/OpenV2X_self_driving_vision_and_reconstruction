import pandas as pd

def genera_sunto_codecarbon_dettagliato(master_csv):
    df = pd.read_csv(master_csv)

    # Converte l'energia da Wh a kWh (corretto: /1000, non /100)
    df["Estimated_Total_Energy_kWh"] = df["Estimated_Total_Energy_Wh"] / 1000.0
    df["Total_GPU_Energy_kWh"] = df["Total_GPU_Energy_Wh"] / 1000.0

    # Raggruppa per modulo
    summary = df.groupby("Module").agg({
        "Duration_s": "sum",
        "CPU_Usage_%": "mean",
        "Memory_Usage_MB": "mean",
        "Memory_Delta_MB": "sum",
        "CPU_Time_User_s": "sum",
        "CPU_Time_System_s": "sum",
        "Total_GPU_Energy_kWh": "sum",
        "Estimated_Total_Energy_kWh": "sum"
    }).reset_index()

    # Totale finale
    total = pd.DataFrame([{
        "Module": "TOTAL",
        "Duration_s": summary["Duration_s"].sum(),
        "CPU_Usage_%": summary["CPU_Usage_%"].mean(),
        "Memory_Usage_MB": summary["Memory_Usage_MB"].mean(),
        "Memory_Delta_MB": summary["Memory_Delta_MB"].sum(),
        "CPU_Time_User_s": summary["CPU_Time_User_s"].sum(),
        "CPU_Time_System_s": summary["CPU_Time_System_s"].sum(),
        "Total_GPU_Energy_kWh": summary["Total_GPU_Energy_kWh"].sum(),
        "Estimated_Total_Energy_kWh": summary["Estimated_Total_Energy_kWh"].sum()
    }])

    summary = pd.concat([summary, total], ignore_index=True)
    return summary


tabella_moduli = genera_sunto_codecarbon_dettagliato(
    "/home/vrlab/Scaricati/self_driving_vision_and_reconstruction/energy_reports/module_energy_report_20250902_135223.csv"
)
print(tabella_moduli)

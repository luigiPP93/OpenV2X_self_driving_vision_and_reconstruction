import os
import cv2
import tkinter as tk
from tkinter import messagebox


frames_to_save = []

def add_to_frames_to_save(frame_to_save):
    global frames_to_save
    frames_to_save=frame_to_save
    max_frames = 50
    excess_frames = len(frames_to_save) - max_frames
    if excess_frames > 0:
        del frames_to_save[:excess_frames]

    return frames_to_save


def save_text(text_entry_widget):
    global frames_to_save
    text_to_save = text_entry_widget.get("1.0", "end-1c")  # Ottieni il testo inserito dall'utente
    if text_to_save.strip():
        # Salva il testo
        text_file_path = os.path.abspath("testo_salvato.txt")
        with open(text_file_path, "w") as file:
            file.write(text_to_save)
        print(f"[📝] Testo salvato in: {text_file_path}")

        # Salva i frame
        frames_folder = "frame_video"
        os.makedirs(frames_folder, exist_ok=True)
        abs_frames_folder = os.path.abspath(frames_folder)
        print(f"[📁] Salvo i frame in: {abs_frames_folder}")

        for idx, frame in enumerate(frames_to_save):
            filename = os.path.join(frames_folder, f"frame_{idx}.png")
            success = cv2.imwrite(filename, frame)
            if success:
                print(f"[✅] Frame salvato: {os.path.abspath(filename)}")
            else:
                print(f"[❌] Errore nel salvataggio del frame: {os.path.abspath(filename)}")

        messagebox.showinfo("Salvataggio completato", "Il testo e i frame sono stati salvati correttamente!")
    else:
        messagebox.showwarning("Nessun testo", "Inserisci del testo prima di salvare.")

    return frames_to_save


def run_tkinter():
    text_save_window = tk.Tk()
    text_save_window.title("Interfaccia utente per segnalzioni")

    text_frame = tk.Frame(text_save_window, padx=20, pady=20)
    text_frame.pack()

    text_label = tk.Label(text_frame, text="Inserisci una descrizione dell'anomalia:")
    text_label.pack()

    text_entry = tk.Text(text_frame, height=10, width=50)
    text_entry.pack(pady=10)

    save_button = tk.Button(text_frame, text="Invia", command=lambda: save_text(text_entry))
    save_button.pack()

    text_save_window.mainloop()

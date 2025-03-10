import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

class MentalHealthSurvey:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mental Health Survey")
        self.root.configure(bg="white")

        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Load the updated multi-class model
        try:
            self.model = joblib.load("../models/_random_forest_model_multiclass.pkl")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.quit()

        # Variables to store user input
        self.age = tk.StringVar()
        self.gender = tk.StringVar()
        self.family_history = tk.StringVar()
        self.benefits = tk.StringVar()
        self.care_options = tk.StringVar()
        self.anonymity = tk.StringVar()
        self.leave = tk.StringVar()
        self.work_interfere = tk.StringVar()
        self.flashbacks = tk.StringVar()
        self.social_withdrawal = tk.StringVar()

        self.create_page1()

    def create_page1(self):
        """Create first page of the survey."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.main_frame, text="1. What is your age?").grid(row=0, column=0, pady=5, sticky=tk.W)
        self.age_entry = ttk.Entry(self.main_frame, textvariable=self.age, width=30)
        self.age_entry.grid(row=1, column=0, pady=5, sticky=tk.W)

        ttk.Label(self.main_frame, text="2. What is your Gender?").grid(row=2, column=0, pady=5, sticky=tk.W)
        for i, option in enumerate(["Male", "Female", "Transgender"]):
            ttk.Radiobutton(self.main_frame, text=option, variable=self.gender, value=option).grid(row=3 + i, column=0, pady=2, sticky=tk.W)

        questions = [
            ("3. Do you have a family history of mental health conditions?", self.family_history),
            ("4. Does your company provide mental health benefits?", self.benefits),
            ("5. Are mental health care options available at work?", self.care_options),
        ]

        row_index = 6
        for question, var in questions:
            ttk.Label(self.main_frame, text=question).grid(row=row_index, column=0, pady=5, sticky=tk.W)
            for i, option in enumerate(["Yes", "No"]):
                ttk.Radiobutton(self.main_frame, text=option, variable=var, value=option).grid(row=row_index + 1 + i, column=0, pady=2, sticky=tk.W)
            row_index += 3

        self.next_button = ttk.Button(self.main_frame, text="Next", command=self.create_page2)
        self.next_button.grid(row=row_index, column=0, pady=20, sticky=tk.W)

    def create_page2(self):
        """Create second page of the survey."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        questions = [
            ("6. Is anonymity protected when discussing mental health issues at work?", self.anonymity),
            ("7. How easy is it to take leave for mental health reasons?", self.leave, ["Easy", "Difficult"]),
            ("8. How often does mental health interfere with your work?", self.work_interfere, ["Easy", "Difficult"]),
            ("9. Have you experienced flashbacks?", self.flashbacks),
            ("10. Do you often withdraw from social situations?", self.social_withdrawal),
        ]

        row_index = 0
        for question in questions:
            ttk.Label(self.main_frame, text=question[0]).grid(row=row_index, column=0, pady=5, sticky=tk.W)
            if len(question) == 3:
                for i, option in enumerate(question[2]):
                    ttk.Radiobutton(self.main_frame, text=option, variable=question[1], value=option).grid(row=row_index + 1 + i, column=0, pady=2, sticky=tk.W)
            else:
                for i, option in enumerate(["Yes", "No"]):
                    ttk.Radiobutton(self.main_frame, text=option, variable=question[1], value=option).grid(row=row_index + 1 + i, column=0, pady=2, sticky=tk.W)
            row_index += 3

        self.submit_button = ttk.Button(self.main_frame, text="Submit", command=self.submit_survey)
        self.submit_button.grid(row=row_index, column=0, pady=20, sticky=tk.W)

    def submit_survey(self):
        """Process input, predict using the model, and show the result in a message box."""
        try:
            leave_mapping = {"Difficult": 0, "Easy": 1}
            work_interfere_mapping = {"Difficult": 0, "Easy": 1}
            processed_data = np.array([
                int(self.age.get()) if self.age.get().isdigit() else 0,
                1 if self.gender.get() == "Female" else (2 if self.gender.get() == "Transgender" else 0),
                *[1 if var.get() == "Yes" else 0 for var in [
                    self.family_history, self.benefits, self.care_options, self.anonymity,
                    self.flashbacks, self.social_withdrawal
                ]],
                leave_mapping.get(self.leave.get(), 0),
                work_interfere_mapping.get(self.work_interfere.get(), 0)
            ]).reshape(1, -1)

            prediction = self.model.predict(processed_data)[0]
            result_map = {0: "No significant mental health concerns.", 1: "You may need to seek the help of a mental health professional.", 2: "You may have PTSD.Please seek the help of mental health professional."}
            messagebox.showinfo("Prediction Result", result_map.get(prediction, "Unknown result"))
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()    

if __name__ == "__main__":
    app = MentalHealthSurvey()
    app.run()

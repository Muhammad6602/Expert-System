import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import PySimpleGUI as sg
import base64
# Define the PySimpleGUI layout
layout = [
    [sg.Text("Select a CSV file:")],
    [sg.Input(key="-FILE-", enable_events=True), sg.FileBrowse()],
    [sg.Text("Enter the label for the decision tree:")],
    [sg.Input(key="-LABEL-")],
    [sg.Button("Generate Decision Tree"), sg.Button("Predict"), sg.Button("Quit")],
    [sg.Text("Enter the data you want to predict:"),sg.Text("  hint:A B C")],
    [sg.Input(key="-PREDICT-")],
    [sg.Multiline(size = (40,10), key = "-FINAL-")]
]

# Create the PySimpleGUI window
window = sg.Window("Decision Tree Generator", layout)

# Event loop to process PySimpleGUI events
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Quit":
        break
    if event == "-FILE-":
        file_path = values["-FILE-"]
        try:
            df = pd.read_csv(file_path)
            sg.popup("CSV file loaded successfully!")
        except Exception as e:
            sg.popup_error(f"Error loading CSV file: {str(e)}")
    if event == "Generate Decision Tree":

            label = values["-LABEL-"]
            x = df.drop(columns=[label])
            y = df[label]
            clf = DecisionTreeClassifier()
            clf.fit(x, y)
            dot_data = export_graphviz(clf, out_file=None,
                                       feature_names=x.columns,
                                       class_names=y.unique().astype(str),
                                       filled=True, rounded=True,
                                       special_characters=True)
            graph = graphviz.Source(dot_data)
            png_bytes = graph.pipe(format='png')
            # Convert the PNG image to a base64-encoded string
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            # Create a PySimpleGUI Image element with the base64 string
            image_elem = sg.Image(data=b64, subsample = 2)
            # Create a new window to display the image
            layout2 = [[image_elem]]
            window2 = sg.Window("Decision Tree", layout2)
            while True:

                event, values = window2.read(timeout=20)
                if event == sg.WIN_CLOSED:
                    break
    if event == "Predict":
          label = values["-LABEL-"]
          x = df.drop(columns=[label])
          y = df[label]
          clf = DecisionTreeClassifier()
          clf.fit(x, y)
          Z = values["-PREDICT-"]
          Z = Z.split()
          input_df = pd.DataFrame([Z], columns=x.columns)
          prediction = clf.predict(input_df)
          window["-FINAL-"].update(prediction)
#['5.1', '3.4', '1.4', '0.2']        
# Close the PySimpleGUI window
window.close()
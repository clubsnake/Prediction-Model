# Prediction Model Architecture Documentation

## Viewing Mermaid Diagrams

### Using VS Code MermaidChart Extension

1. Install the "Mermaid Chart" extension in VS Code
   - Search for "Mermaid Chart" in the extensions marketplace
   - Install the extension by Mermaid Chart Inc.

2. Open the workspace file
   - Open VS Code
   - Go to File > Open Workspace from File...
   - Select `prediction-model.code-workspace`

3. Open diagram files
   - Open `architecture.mmd` or `function_call_graph.mmd`
   - VS Code should recognize these as Mermaid diagrams

4. To preview the diagram:
   - Press Ctrl+Shift+P to open the command palette
   - Type "Mermaid" and select "Mermaid: Preview"
   - Alternatively, you may see a "Preview" button in the editor

### Using the HTML Viewer

1. Double-click on `view_diagrams.bat` to launch the HTML viewer
2. When the browser opens:
   - Click the "Choose File" button
   - Navigate to the Prediction Model folder
   - Select either `architecture.mmd` or `function_call_graph.mmd`
3. The diagram will display in the browser
4. Use the zoom buttons to adjust the size

## Troubleshooting

If diagrams aren't displaying in VS Code:

1. Make sure you have the latest version of the Mermaid Chart extension
2. Try reloading VS Code (File > Reload Window)
3. Check if the extension recognized the `.mmd` files correctly
4. Try renaming files to `.md` and wrapping the mermaid code in markdown code blocks if needed

If the HTML viewer shows "Error loading diagram":
1. Make sure you're selecting the correct file type (.mmd)
2. Check that the file exists in the expected location
3. Try using a different browser (Chrome or Edge recommended)

import os

def summarize_project(root_dir):
    summary = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py"):
                path = os.path.join(dirpath, file)
                with open(path, 'r') as f:
                    try:
                        content = f.read()
                        lines = len(content.split('\n'))
                        imports = [line for line in content.split('\n') 
                                   if line.strip().startswith('import ') 
                                   or line.strip().startswith('from ')]
                        summary.append({
                            'path': path,
                            'lines': lines,
                            'imports': imports
                        })
                    except UnicodeDecodeError:
                        summary.append({
                            'path': path,
                            'lines': '?',
                            'imports': []
                        })
    return summary

# Run it
summary = summarize_project("C:/Users/clubs/Desktop/Prediction Model")
with open("project_summary.txt", "w") as f:
    for item in summary:
        f.write(f"File: {item['path']}\n")
        f.write(f"Lines: {item['lines']}\n")
        f.write("Imports:\n")
        for imp in item['imports']:
            f.write(f"  {imp}\n")
        f.write("\n")
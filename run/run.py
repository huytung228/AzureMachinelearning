from azureml.core import Workspace, Run
ws = Workspace.from_config()
print(ws)
run = Run.get_context()
print(run)
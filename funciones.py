from langsmith import traceable
@traceable
def debug(debug):
    showLogs = True
    if(showLogs):
        print(debug)
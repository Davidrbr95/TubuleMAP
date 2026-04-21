import psutil
import platform

def normal_round(value):
    """Round a value using half-up behavior."""
    return int(value + 0.5)

def is_excel_running():
    # List of possible process names for Excel
    """Return whether excel is running."""
    excel_process_names = {
        'Windows': 'EXCEL.EXE',
        'Darwin': 'Microsoft Excel',
        'Linux': 'excel'  # or use specific process name if different
    }
    
    # Get the current system platform
    platform_system = platform.system()

    if platform_system not in excel_process_names:
        raise RuntimeError("Unsupported OS")

    # Get the specific process name for Excel on the current platform
    process_name = excel_process_names[platform_system]
    
    # Iterate through all running processes
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False
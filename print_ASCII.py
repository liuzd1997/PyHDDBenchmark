import sys
from packaging import version

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def print_banner(version):
    banner = r"""
     ______          _     _  _____    _____   
    (_____ \        | |   | |(____ \  (____ \  
     _____) ) _   _ | |__ | | _   \ \  _   \ \ 
    |  ____/ | | | ||  __)| || |   | || |   | |
    | |      | |_| || |   | || |__/ / | |__/ / 
    |_|       \__  ||_|   |_||_____/  |_____/  
            (____/                            
    """
    colored_banner = color_text(banner, '36')  # Cyan color
    version_text = f"(v{version})"
    centered_version = version_text.center(len(banner.splitlines()[-1]))
    colored_version = color_text(centered_version, '33')  # Yellow color
    
    print(colored_banner)
    print(colored_version)

def check_version(current_version, min_version):
    if version.parse(current_version) < version.parse(min_version):
        print(color_text(f"Warning: Your version ({current_version}) is outdated. Please upgrade to at least version {min_version}.", '31'))

def print_system(current_version='0.1.0', min_version='0.1.0'):
    print_banner(current_version)
    check_version(current_version, min_version)

if __name__ == "__main__":
    print_system()
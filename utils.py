from colorama import init, Fore, Style

init(autoreset=True)

def info(msg): 
    print(Fore.CYAN + msg)

def success(msg): 
    print(Fore.GREEN + msg)

def warn(msg): 
    print(Fore.YELLOW + msg)

def error(msg): 
    print(Fore.RED + msg)

def bold(msg): 
    print(Style.BRIGHT + msg)

def divider(char='-', length=60): 
    print(Fore.MAGENTA + char * length)

# Style constants
bright = Style.BRIGHT
reset = Style.RESET_ALL 
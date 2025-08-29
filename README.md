# Dedimo - ***Ded***alus and mar***imo***
[Dedalus](https://dedalus-project.org/) examples adopted for [marimo](https://marimo.io)

Once I discovered the amazing package Dedalus, but I could not run it under Windows. 
The problem was solved with the help of WSL.
But to make it convenient to work, I decided to add Marimo notebooks 

Now one command is enough:
```cmd
wsl --cd ~/dedimo -e bash -c "source ~/.bashrc && marimo edit"
```

or simple:
```bash
marimo edit
```

## Installation on WSL

1. install miniforge from https://github.com/conda-forge/miniforge
2. install marimo `conda install marimo`
3. install dedalus `conda install dedalus`
4. clone this rep
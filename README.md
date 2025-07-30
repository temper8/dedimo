# Dedimo - ***Ded***alus and mar***imo***
Dedalus examples adopted for marimo

Once I discovered the amazing package Dedalus, but I could not run it under Windows. 
The problem was solved with the help of WSL.
But to make it convenient to work, I decided to add Marimo notebooks 

Now one command is enough:
```cmd
wsl --cd ~/dedimo -e bash -c "source ~/.bashrc && conda run marimo edit"
```
